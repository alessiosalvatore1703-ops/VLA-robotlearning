#!/usr/bin/env python3
"""
Augment a LeRobot v3 dataset on the HuggingFace Hub with one brightness-augmented
copy of the source, where the source is split into N contiguous partitions
(N = number of brightness levels) and each partition is rendered at a single
brightness multiplier.

For S source episodes and L brightness levels:
  - Output episodes 0..S-1            = originals (videos copied unchanged)
  - Output episodes S..2*S-1          = augmented copies
        Each source episode i is assigned brightness levels[(i * L) // S].
        Final size = 2 * source.

All non-image fields (action, observation.state, timestamps, task_index,
per-episode stats) are copied from the source — only the pixels change.

Usage:
    python augment_brightness_push.py --src user/my_dataset --dst user/my_dataset_bright
    python augment_brightness_push.py --src user/my_dataset --dst user/my_dataset_bright \
        --brightness-levels 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_CHUNK_SIZE = 1000   # rows per output data parquet file
VIDEO_CRF = 30           # SVT-AV1 quality (lower = better)
VIDEO_PRESET = 8         # SVT-AV1 speed preset (0=slowest/best, 12=fastest)


def discover_video_keys(info: dict) -> list[str]:
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def episode_partition(ep_idx: int, n_episodes: int, n_levels: int) -> int:
    """Assign each source episode to one of n_levels contiguous partitions."""
    return (ep_idx * n_levels) // n_episodes


def reencode_per_frame_brightness(
    src_video: Path,
    out_video: Path,
    frame_brightness: dict[int, float],
    fps: float,
):
    """Re-encode src_video into out_video, multiplying each frame by its assigned brightness."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    codec_name = "libsvtav1" if "libsvtav1" in av.codecs_available else "libx264"
    rate = int(round(fps))

    with av.open(str(src_video)) as src:
        in_stream = src.streams.video[0]
        w, h = in_stream.width, in_stream.height

    with av.open(str(out_video), mode="w") as dst:
        if codec_name == "libsvtav1":
            out_stream = dst.add_stream(codec_name, rate=rate)
            out_stream.options = {"crf": str(VIDEO_CRF), "preset": str(VIDEO_PRESET)}
        else:
            out_stream = dst.add_stream(codec_name, rate=rate)
            out_stream.options = {"crf": "18", "preset": "fast", "bf": "0"}
        out_stream.width = w
        out_stream.height = h
        out_stream.pix_fmt = "yuv420p"

        out_pts = 0
        with av.open(str(src_video)) as src:
            for frame_idx, frame in enumerate(src.decode(video=0)):
                b = frame_brightness.get(frame_idx, 1.0)
                arr = frame.to_ndarray(format="rgb24").astype(np.float32)
                if b != 1.0:
                    arr = np.clip(arr * b, 0, 255)
                arr = arr.astype(np.uint8)
                out_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                out_frame = out_frame.reformat(format="yuv420p")
                out_frame.pts = out_pts
                out_pts += 1
                for pkt in out_stream.encode(out_frame):
                    dst.mux(pkt)

        for pkt in out_stream.encode():
            dst.mux(pkt)

    return out_pts


def build_output_data_parquets(src_df, out_dir: Path):
    """Concat originals followed by one augmented block (episode_index and index shifted)."""
    import pandas as pd

    n_src_episodes = int(src_df["episode_index"].max()) + 1
    n_src_frames = len(src_df)

    blocks = [src_df.copy()]
    aug = src_df.copy()
    aug["episode_index"] = aug["episode_index"] + n_src_episodes
    aug["index"] = aug["index"] + n_src_frames
    blocks.append(aug)

    out_df = pd.concat(blocks, ignore_index=True)

    ep_data_loc: dict[int, tuple[int, int]] = {}
    file_idx = 0
    for start in range(0, len(out_df), DATA_CHUNK_SIZE):
        chunk = out_df.iloc[start : start + DATA_CHUNK_SIZE]
        out_path = out_dir / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(chunk, preserve_index=False), out_path)
        for ep_id in chunk["episode_index"].unique():
            if int(ep_id) not in ep_data_loc:
                ep_data_loc[int(ep_id)] = (0, file_idx)
        file_idx += 1

    return out_df, ep_data_loc, file_idx


def build_episodes_meta(
    orig_ep_meta,
    video_keys: list[str],
    src_video_file_counts: dict[str, int],
    out_df,
    ep_data_loc: dict[int, tuple[int, int]],
    out_dir: Path,
):
    """Build 2*S episode rows: S originals followed by S augmented copies."""
    import pandas as pd

    n_src_episodes = len(orig_ep_meta)
    new_rows = []

    # ---- Originals ----
    for _, orig_row in orig_ep_meta.iterrows():
        row = orig_row.copy()
        ep_id = int(orig_row["episode_index"])
        ep_data = out_df[out_df["episode_index"] == ep_id]
        if "dataset_from_index" in row.index:
            row["dataset_from_index"] = int(ep_data["index"].iloc[0])
            row["dataset_to_index"] = int(ep_data["index"].iloc[-1]) + 1
        chunk_idx, file_idx = ep_data_loc[ep_id]
        if "data/chunk_index" in row.index:
            row["data/chunk_index"] = chunk_idx
        if "data/file_index" in row.index:
            row["data/file_index"] = file_idx
        new_rows.append(row)

    # ---- Augmented copies (same per-episode video position, shifted file_index) ----
    for _, orig_row in orig_ep_meta.iterrows():
        row = orig_row.copy()
        src_ep_id = int(orig_row["episode_index"])
        new_ep_id = src_ep_id + n_src_episodes
        row["episode_index"] = new_ep_id

        ep_data = out_df[out_df["episode_index"] == new_ep_id]
        if "dataset_from_index" in row.index:
            row["dataset_from_index"] = int(ep_data["index"].iloc[0])
            row["dataset_to_index"] = int(ep_data["index"].iloc[-1]) + 1
        chunk_idx, file_idx = ep_data_loc[new_ep_id]
        if "data/chunk_index" in row.index:
            row["data/chunk_index"] = chunk_idx
        if "data/file_index" in row.index:
            row["data/file_index"] = file_idx

        for vk in video_keys:
            fcol = f"videos/{vk}/file_index"
            if fcol in row.index:
                row[fcol] = int(orig_row[fcol]) + src_video_file_counts[vk]

        new_rows.append(row)

    new_meta = pd.DataFrame(new_rows).reset_index(drop=True)
    out_path = out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(new_meta, preserve_index=False), out_path)
    return new_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source HF repo (e.g. user/my_dataset)")
    parser.add_argument("--dst", required=True, help="Destination HF repo (e.g. user/my_dataset_bright)")
    parser.add_argument(
        "--brightness-levels", type=float, nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        help="List of brightness multipliers. Source episodes are split into "
             "len(levels) contiguous partitions; each partition gets one level.",
    )
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create destination repo as private")
    args = parser.parse_args()

    levels: list[float] = list(args.brightness_levels)
    n_levels = len(levels)
    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Brightness levels ({n_levels}): {levels}")

    # ------------------------------------------------------------------ #
    # 1. Download meta + data parquets (no videos yet)
    # ------------------------------------------------------------------ #
    print("\n[1/5] Downloading dataset metadata and parquets...")
    local_dir = Path(
        snapshot_download(
            repo_id=args.src,
            repo_type="dataset",
            ignore_patterns=["videos/*"],
            token=token,
        )
    )
    print(f"  Cached at: {local_dir}")

    # ------------------------------------------------------------------ #
    # 2. Read source structure
    # ------------------------------------------------------------------ #
    print("\n[2/5] Reading source structure...")
    import pandas as pd

    with open(local_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = float(info["fps"])
    video_keys = discover_video_keys(info)
    print(f"  fps={fps}  video_keys={video_keys}")

    src_parquets = sorted((local_dir / "data" / "chunk-000").glob("*.parquet"))
    src_df = pd.concat([pq.read_table(p).to_pandas() for p in src_parquets], ignore_index=True)
    n_src_episodes = int(src_df["episode_index"].max()) + 1
    n_src_frames = len(src_df)
    print(f"  source episodes: {n_src_episodes}  frames: {n_src_frames}")

    orig_ep_meta = pq.read_table(
        local_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pandas()

    src_video_file_counts: dict[str, int] = {}
    for vk in video_keys:
        fcol = f"videos/{vk}/file_index"
        if fcol in orig_ep_meta.columns:
            src_video_file_counts[vk] = int(orig_ep_meta[fcol].max()) + 1
        else:
            src_video_file_counts[vk] = 1
    print(f"  source video files per key: {src_video_file_counts}")

    # ---- Assign brightness per source episode ----
    ep_brightness: dict[int, float] = {}
    partition_counts = [0] * n_levels
    for ep_id in range(n_src_episodes):
        p = episode_partition(ep_id, n_src_episodes, n_levels)
        ep_brightness[ep_id] = levels[p]
        partition_counts[p] += 1
    print("  partition sizes: " + ", ".join(
        f"b={levels[p]:.2f}:{partition_counts[p]}" for p in range(n_levels)
    ))

    # ------------------------------------------------------------------ #
    # 3. Build output data parquets and episode metadata
    # ------------------------------------------------------------------ #
    print("\n[3/5] Building output data parquets and episode metadata...")
    work_dir = Path(tempfile.mkdtemp(prefix="lerobot_brightness_"))
    print(f"  Working directory: {work_dir}")

    out_df, ep_data_loc, n_data_files = build_output_data_parquets(src_df, work_dir)
    print(f"  wrote {n_data_files} data parquet file(s), {len(out_df)} total rows")

    new_ep_meta = build_episodes_meta(
        orig_ep_meta,
        video_keys=video_keys,
        src_video_file_counts=src_video_file_counts,
        out_df=out_df,
        ep_data_loc=ep_data_loc,
        out_dir=work_dir,
    )
    print(f"  wrote episodes meta: {len(new_ep_meta)} episodes")

    # ------------------------------------------------------------------ #
    # 4. For each source video file: copy original, re-encode augmented
    # ------------------------------------------------------------------ #
    print(f"\n[4/5] Processing videos ({len(video_keys)} key(s))...")
    for vk in video_keys:
        n_files = src_video_file_counts[vk]
        chunk_col = f"videos/{vk}/chunk_index"
        file_col = f"videos/{vk}/file_index"
        from_col = f"videos/{vk}/from_timestamp"
        to_col = f"videos/{vk}/to_timestamp"

        print(f"  Video key '{vk}': {n_files} source file(s)")
        for f in range(n_files):
            video_repo_path = f"videos/{vk}/chunk-000/file-{f:03d}.mp4"
            src_video = Path(
                hf_hub_download(
                    repo_id=args.src,
                    repo_type="dataset",
                    filename=video_repo_path,
                    token=token,
                )
            )

            # Copy original to the corresponding output slot
            out_orig = work_dir / "videos" / vk / "chunk-000" / f"file-{f:03d}.mp4"
            out_orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, out_orig)
            print(f"    copied original  videos/{vk}/chunk-000/file-{f:03d}.mp4")

            # Build frame-index -> brightness map for the augmented copy of this file
            file_eps = orig_ep_meta[
                (orig_ep_meta[chunk_col] == 0) & (orig_ep_meta[file_col] == f)
            ] if chunk_col in orig_ep_meta.columns else orig_ep_meta
            frame_brightness: dict[int, float] = {}
            covered = []
            for _, row in file_eps.iterrows():
                ep_id = int(row["episode_index"])
                from_ts = float(row[from_col])
                to_ts = float(row[to_col])
                start = round(from_ts * fps)
                end = round(to_ts * fps) + 1
                b = ep_brightness[ep_id]
                for fi in range(start, end):
                    frame_brightness[fi] = b
                covered.append((ep_id, b, start, end))

            new_file_idx = n_files + f
            out_aug = work_dir / "videos" / vk / "chunk-000" / f"file-{new_file_idx:03d}.mp4"
            print(f"    encoding augmented copy -> file-{new_file_idx:03d}.mp4 "
                  f"({len(covered)} episodes, brightness range "
                  f"{min(b for _, b, _, _ in covered):.2f}..{max(b for _, b, _, _ in covered):.2f})")
            n_written = reencode_per_frame_brightness(src_video, out_aug, frame_brightness, fps)
            print(f"      wrote {n_written} frames")

    # ------------------------------------------------------------------ #
    # 5. Finalize info.json, copy tasks.parquet, push
    # ------------------------------------------------------------------ #
    print("\n[5/5] Finalizing and pushing...")

    new_info = dict(info)
    new_info["total_episodes"] = n_src_episodes * 2
    new_info["total_frames"] = n_src_frames * 2
    if "total_videos" in info:
        new_info["total_videos"] = info["total_videos"] * 2
    new_info["augmentation"] = (
        f"Brightness-augmented from {args.src}: source episodes split into "
        f"{n_levels} partitions, brightness levels {levels}. Output = original + "
        f"one augmented copy per source episode (2x source size)."
    )
    (work_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(work_dir / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    shutil.copy(local_dir / "meta" / "tasks.parquet", work_dir / "meta" / "tasks.parquet")

    try:
        api.repo_info(repo_id=args.dst, repo_type="dataset")
        print("  Repo exists, uploading files...")
    except Exception:
        print("  Creating new repo...")
        api.create_repo(repo_id=args.dst, repo_type="dataset", private=args.private)

    api.upload_folder(
        folder_path=str(work_dir),
        repo_id=args.dst,
        repo_type="dataset",
        commit_message=(
            f"Add 1 brightness-augmented copy per episode "
            f"(partition-assigned over levels {levels})"
        ),
    )

    print(f"\nDone. Dataset pushed to https://huggingface.co/datasets/{args.dst}")
    print(f"  Episodes: {n_src_episodes} -> {new_info['total_episodes']}")
    print(f"  Frames:   {n_src_frames} -> {new_info['total_frames']}")

    shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
