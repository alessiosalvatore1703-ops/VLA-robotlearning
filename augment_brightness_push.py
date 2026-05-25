#!/usr/bin/env python3
"""
Augment a LeRobot v3 dataset on the HuggingFace Hub *in place*: each source
episode's video is re-encoded at a single brightness multiplier. The dataset
keeps exactly the same number of episodes — no episodes are duplicated.

For S source episodes and L brightness levels:
  - Each source episode is assigned one brightness level chosen uniformly at
    random (reproducible via --seed).
  - Output has exactly S episodes (same count as the source).

All non-image fields (action, observation.state, timestamps, task_index,
per-episode stats) are copied from the source unchanged — only the pixels change.

Usage:
    python augment_brightness_push.py --src user/my_dataset --dst user/my_dataset_bright
    python augment_brightness_push.py --src user/my_dataset --dst user/my_dataset_bright \
        --brightness-levels 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
"""

import argparse
import json
import os
import random
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
    """Write the source data parquets unchanged — no episodes are duplicated.

    Only the video pixels are augmented; every non-image field (and therefore
    episode_index / index) is identical to the source.
    """
    out_df = src_df.copy()

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


def dedup_episodes_meta(orig_ep_meta, src_df):
    """Drop stale/duplicate rows from a corrupt episodes-metadata table.

    Some datasets ship an episodes parquet with more rows than real episodes
    (leftover phantom rows from earlier edits/merges). The data parquets are
    the source of truth: for each episode_index we keep the meta row whose
    [dataset_from_index, dataset_to_index) matches the episode's actual frame
    range in src_df, and drop any episode_index not present in the data.
    """
    actual = {}
    for ep_id, grp in src_df.groupby("episode_index"):
        actual[int(ep_id)] = (int(grp["index"].min()), int(grp["index"].max()) + 1)

    kept_rows = []
    dropped = 0
    for ep_id, grp in orig_ep_meta.groupby("episode_index"):
        ep_id = int(ep_id)
        if ep_id not in actual:
            dropped += len(grp)
            continue
        if len(grp) == 1:
            kept_rows.append(grp.iloc[0])
            continue
        from_idx, to_idx = actual[ep_id]
        match = grp[
            (grp["dataset_from_index"] == from_idx)
            & (grp["dataset_to_index"] == to_idx)
        ]
        if len(match) >= 1:
            kept_rows.append(match.iloc[0])
        else:
            kept_rows.append(grp.iloc[0])  # no match: keep first as fallback
        dropped += len(grp) - 1

    import pandas as pd

    deduped = (
        pd.DataFrame(kept_rows)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )
    if dropped:
        print(f"  WARNING: source episodes meta had {dropped} stale/duplicate "
              f"row(s); kept {len(deduped)} episode(s) matching the data parquets")
    return deduped


def build_episodes_meta(
    orig_ep_meta,
    out_df,
    ep_data_loc: dict[int, tuple[int, int]],
    out_dir: Path,
):
    """Rebuild the S episode rows in place — same episodes, same video positions.

    Videos are re-encoded under their original file index, so no video file
    columns need shifting; only the data-parquet locations are refreshed.
    """
    import pandas as pd

    new_rows = []
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

    new_meta = pd.DataFrame(new_rows).reset_index(drop=True)
    out_path = out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(new_meta, preserve_index=False), out_path)
    return new_meta


def write_readme(out_dir: Path, repo_id: str | None = None):
    """Write a minimal LeRobot dataset card.

    Only the YAML front-matter matters for the Hugging Face dataset page to embed
    the LeRobot visualizer: the ``LeRobot`` tag (capital L and R) is what makes HF
    recognise it as a LeRobot dataset, and the ``configs`` block points the viewer
    at the parquet files. This matches the card LeRobot itself generates.
    """
    frontmatter = (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "  - robotics\n"
        "tags:\n"
        "  - LeRobot\n"
        "configs:\n"
        "  - config_name: default\n"
        "    data_files: data/*/*.parquet\n"
        "---\n"
    )

    body = "\nThis dataset was created using [LeRobot](https://github.com/huggingface/lerobot).\n"
    if repo_id:
        body += (
            f'\n<a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path={repo_id}">\n'
            '  <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>\n'
            "</a>\n"
        )

    (out_dir / "README.md").write_text(frontmatter + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source HF repo (e.g. user/my_dataset)")
    parser.add_argument("--dst", required=True, help="Destination HF repo (e.g. user/my_dataset_bright)")
    parser.add_argument(
        "--brightness-levels", type=float, nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        help="List of brightness multipliers. Each source episode is assigned "
             "one level chosen uniformly at random.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the per-episode brightness assignment (reproducible).",
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
    orig_ep_meta = dedup_episodes_meta(orig_ep_meta, src_df)

    src_video_file_counts: dict[str, int] = {}
    for vk in video_keys:
        fcol = f"videos/{vk}/file_index"
        if fcol in orig_ep_meta.columns:
            src_video_file_counts[vk] = int(orig_ep_meta[fcol].max()) + 1
        else:
            src_video_file_counts[vk] = 1
    print(f"  source video files per key: {src_video_file_counts}")

    # ---- Assign a random brightness level to each source episode ----
    rng = random.Random(args.seed)
    ep_brightness: dict[int, float] = {}
    level_counts = {lvl: 0 for lvl in levels}
    for ep_id in range(n_src_episodes):
        lvl = rng.choice(levels)
        ep_brightness[ep_id] = lvl
        level_counts[lvl] += 1
    print(f"  random brightness assignment (seed={args.seed}); level counts: "
          + ", ".join(f"b={lvl:.2f}:{level_counts[lvl]}" for lvl in levels))

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
        out_df=out_df,
        ep_data_loc=ep_data_loc,
        out_dir=work_dir,
    )
    print(f"  wrote episodes meta: {len(new_ep_meta)} episodes")

    # ---- Verify in place: episode count must be unchanged ----
    n_out_episodes = len(new_ep_meta)
    if n_out_episodes != n_src_episodes:
        raise RuntimeError(
            f"Episode count changed during augmentation: "
            f"{n_src_episodes} (source) -> {n_out_episodes} (output). "
            f"Augmentation must be in place."
        )
    print(f"  verified episode count unchanged: {n_src_episodes} == {n_out_episodes}")

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

            # Build frame-index -> brightness map for this file's episodes
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

            # Re-encode in place: same file index, brightness applied to pixels
            out_aug = work_dir / "videos" / vk / "chunk-000" / f"file-{f:03d}.mp4"
            print(f"    re-encoding in place -> file-{f:03d}.mp4 "
                  f"({len(covered)} episodes, brightness range "
                  f"{min(b for _, b, _, _ in covered):.2f}..{max(b for _, b, _, _ in covered):.2f})")
            n_written = reencode_per_frame_brightness(src_video, out_aug, frame_brightness, fps)
            print(f"      wrote {n_written} frames")

    # ------------------------------------------------------------------ #
    # 5. Finalize info.json, copy tasks.parquet, push
    # ------------------------------------------------------------------ #
    print("\n[5/5] Finalizing and pushing...")

    new_info = dict(info)
    # In-place augmentation: episode/frame/video counts are unchanged.
    new_info["total_episodes"] = n_src_episodes
    new_info["total_frames"] = n_src_frames
    new_info["augmentation"] = (
        f"Brightness-augmented in place from {args.src}: each episode assigned a "
        f"random brightness level from {levels} (seed={args.seed}). Each episode's "
        f"video is re-encoded at its brightness; episode count unchanged "
        f"({n_src_episodes})."
    )
    (work_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(work_dir / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    shutil.copy(local_dir / "meta" / "tasks.parquet", work_dir / "meta" / "tasks.parquet")

    # README.md — the `LeRobot` tag is what makes the HF dataset page embed the
    # visualizer; without a card the augmented dataset cannot be visualized.
    write_readme(work_dir, repo_id=args.dst)

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
            f"Brightness-augment episodes in place "
            f"(random levels {levels}, seed={args.seed})"
        ),
    )

    print(f"\nDone. Dataset pushed to https://huggingface.co/datasets/{args.dst}")
    print(f"  Episodes: {n_src_episodes} (unchanged)")
    print(f"  Frames:   {n_src_frames} (unchanged)")

    shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
