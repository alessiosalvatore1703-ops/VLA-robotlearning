#!/usr/bin/env python3
"""
Trim leading/trailing frozen action frames from a LeRobot v3 dataset on HuggingFace Hub
and push the cleaned dataset to a new HF repo.

Usage:
    python trim_and_push.py --src Alessio03/task1dataset --dst Alessio03/task1dataset_clean
    python trim_and_push.py --src Alessio03/task2dataset --dst Alessio03/task2dataset_clean
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
MIN_EPISODE_FRAMES = 50   # never trim below this length
VIDEO_KEY = "observation.images.front"
VIDEO_CRF = 35            # SVT-AV1 quality (lower = better, 35 ≈ visually good)
VIDEO_PRESET = 8          # SVT-AV1 speed preset (0=slowest/best, 12=fastest)


def detect_bounds(actions: np.ndarray) -> tuple[int, int]:
    """Return (lead, trail) frozen frame counts at start/end of episode."""
    n = len(actions)
    lead = 0
    for i in range(1, n):
        if np.all(actions[i] == actions[0]):
            lead += 1
        else:
            break
    trail = 0
    for i in range(n - 2, lead, -1):
        if np.all(actions[i] == actions[-1]):
            trail += 1
        else:
            break
    # Safety: never trim so much that the episode falls below minimum length
    while lead + trail > n - MIN_EPISODE_FRAMES and (lead > 0 or trail > 0):
        if trail >= lead:
            trail -= 1
        else:
            lead -= 1
    return lead, trail


def recompute_episode_stats(ep_df: "pd.DataFrame") -> dict:
    """Compute per-episode stats columns from trimmed episode dataframe."""
    stats = {}
    numeric_cols = {
        "action": ep_df["action"].tolist(),
        "observation.state": ep_df["observation.state"].tolist(),
        "timestamp": ep_df["timestamp"].tolist(),
        "frame_index": ep_df["frame_index"].tolist(),
        "episode_index": ep_df["episode_index"].tolist(),
        "index": ep_df["index"].tolist(),
        "task_index": ep_df["task_index"].tolist(),
    }
    for col_name, values in numeric_cols.items():
        arr = np.array(values, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        stats[f"stats/{col_name}/min"] = arr.min(axis=0).tolist()
        stats[f"stats/{col_name}/max"] = arr.max(axis=0).tolist()
        stats[f"stats/{col_name}/mean"] = arr.mean(axis=0).tolist()
        stats[f"stats/{col_name}/std"] = arr.std(axis=0).tolist()
        stats[f"stats/{col_name}/count"] = [len(arr)] * arr.shape[1]
        stats[f"stats/{col_name}/q01"] = np.quantile(arr, 0.01, axis=0).tolist()
        stats[f"stats/{col_name}/q10"] = np.quantile(arr, 0.10, axis=0).tolist()
        stats[f"stats/{col_name}/q50"] = np.quantile(arr, 0.50, axis=0).tolist()
        stats[f"stats/{col_name}/q90"] = np.quantile(arr, 0.90, axis=0).tolist()
        stats[f"stats/{col_name}/q99"] = np.quantile(arr, 0.99, axis=0).tolist()
    # image stats: copy from original (we can't recompute without decoding every frame)
    return stats


def build_trimmed_parquets(df, trim_map: dict[int, tuple[int, int]], out_dir: Path, fps: float = 10.0):
    """
    Rebuild data parquet files after trimming.
    trim_map: {episode_index: (lead, trail)}
    Returns the trimmed dataframe for use in metadata rebuild.
    """
    import pandas as pd

    rows = []
    global_idx = 0
    for ep_id in sorted(df["episode_index"].unique()):
        ep = df[df["episode_index"] == ep_id].copy().reset_index(drop=True)
        lead, trail = trim_map[ep_id]
        end = len(ep) - trail if trail > 0 else len(ep)
        ep = ep.iloc[lead:end].copy().reset_index(drop=True)

        ep["frame_index"] = np.arange(len(ep), dtype=np.int64)
        ep["timestamp"] = np.round(np.arange(len(ep)) / fps, 1).astype(np.float32)
        ep["index"] = np.arange(global_idx, global_idx + len(ep), dtype=np.int64)
        global_idx += len(ep)
        rows.append(ep)

    trimmed_df = pd.concat(rows, ignore_index=True)

    # Write parquet files respecting original chunk/file structure (1 chunk, split by 1000)
    chunk_size = 1000
    file_idx = 0
    for start in range(0, len(trimmed_df), chunk_size):
        chunk = trimmed_df.iloc[start : start + chunk_size]
        out_path = out_dir / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, out_path)
        file_idx += 1

    return trimmed_df


def build_episode_metadata(trimmed_df, orig_ep_meta, trim_map: dict, fps: float, out_dir: Path):
    """Rebuild meta/episodes parquet with updated lengths, indices, timestamps, and stats."""
    import pandas as pd
    from collections import defaultdict

    # Track cumulative position within each video file so from/to timestamps are correct
    vid_chunk_col = f"videos/{VIDEO_KEY}/chunk_index"
    vid_file_col = f"videos/{VIDEO_KEY}/file_index"
    vid_from_col = f"videos/{VIDEO_KEY}/from_timestamp"
    vid_to_col = f"videos/{VIDEO_KEY}/to_timestamp"
    file_pos: dict[tuple, float] = defaultdict(float)

    new_rows = []
    for _, orig_row in orig_ep_meta.iterrows():
        ep_id = int(orig_row["episode_index"])
        ep_data = trimmed_df[trimmed_df["episode_index"] == ep_id].reset_index(drop=True)
        new_len = len(ep_data)

        row = orig_row.copy()
        row["length"] = new_len
        row["dataset_from_index"] = int(ep_data["index"].iloc[0])
        row["dataset_to_index"] = int(ep_data["index"].iloc[-1]) + 1

        # Accumulate timestamp within the video file (episodes are concatenated in one file)
        fkey = (int(orig_row[vid_chunk_col]), int(orig_row[vid_file_col]))
        new_from_ts = round(file_pos[fkey], 1)
        new_to_ts = round(new_from_ts + (new_len - 1) / fps, 1)
        file_pos[fkey] = new_to_ts + 1 / fps
        row[vid_from_col] = new_from_ts
        row[vid_to_col] = new_to_ts

        # Recompute stats for non-image features
        ep_stats = recompute_episode_stats(ep_data)
        for k, v in ep_stats.items():
            if k in row.index:
                row[k] = v

        new_rows.append(row)

    new_meta = pd.DataFrame(new_rows).reset_index(drop=True)
    out_path = out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(new_meta, preserve_index=False), out_path)
    return new_meta


def trim_video(src_video: Path, out_video: Path, keep_ranges: list[tuple[int, int]], fps: float):
    """
    Re-encode src_video keeping only frames whose 0-based index falls within keep_ranges.
    keep_ranges: list of (start_inclusive, end_exclusive) frame index pairs within this file.
    """
    keep_set = set()
    for s, e in keep_ranges:
        keep_set.update(range(s, e))

    out_video.parent.mkdir(parents=True, exist_ok=True)
    codec_name = "libsvtav1" if "libsvtav1" in av.codecs_available else "libx264"
    rate = int(fps)
    out_pts = 0

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

        with av.open(str(src_video)) as src:
            for frame_idx, frame in enumerate(src.decode(video=0)):
                if frame_idx not in keep_set:
                    continue
                # Deep-copy via numpy to avoid shared memory with the source decoder
                arr = frame.to_ndarray(format="yuv420p")
                out_frame = av.VideoFrame.from_ndarray(arr, format="yuv420p")
                out_frame.pts = out_pts
                out_pts += 1
                for pkt in out_stream.encode(out_frame):
                    dst.mux(pkt)

        for pkt in out_stream.encode():
            dst.mux(pkt)

    print(f"  {out_video.name}: {out_pts} frames written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source HF repo (e.g. Alessio03/task1dataset)")
    parser.add_argument("--dst", required=True, help="Destination HF repo (e.g. Alessio03/task1dataset_clean)")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")

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
    # 2. Load data and detect trim bounds per episode
    # ------------------------------------------------------------------ #
    print("\n[2/5] Computing trim bounds per episode...")
    import pandas as pd

    parquets = sorted((local_dir / "data" / "chunk-000").glob("*.parquet"))
    df = pd.concat([pq.read_table(p).to_pandas() for p in parquets], ignore_index=True)

    with open(local_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = float(info["fps"])

    orig_ep_meta = pq.read_table(
        local_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pandas()

    trim_map = {}
    total_trimmed = 0
    for ep_id in sorted(df["episode_index"].unique()):
        ep = df[df["episode_index"] == ep_id]
        actions = np.stack(ep["action"].values)
        lead, trail = detect_bounds(actions)
        trim_map[ep_id] = (lead, trail)
        trimmed = lead + trail
        total_trimmed += trimmed
        marker = "  >>>" if (lead > 10 or trail > 10) else ""
        print(f"  ep{ep_id:3d}: -{lead:3d} start / -{trail:3d} end  ({trimmed:3d} frames removed){marker}")

    print(f"\n  Total frames to remove: {total_trimmed} / {len(df)}")

    # ------------------------------------------------------------------ #
    # 3. Build trimmed parquets + episode metadata in temp dir
    # ------------------------------------------------------------------ #
    print("\n[3/5] Building trimmed parquets and metadata...")
    work_dir = Path(tempfile.mkdtemp(prefix="lerobot_trim_"))
    print(f"  Working directory: {work_dir}")

    trimmed_df = build_trimmed_parquets(df, trim_map, work_dir, fps=fps)
    new_ep_meta = build_episode_metadata(trimmed_df, orig_ep_meta, trim_map, fps, work_dir)

    # Update info.json
    new_info = dict(info)
    new_info["total_frames"] = len(trimmed_df)
    (work_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(work_dir / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    # Copy tasks.parquet unchanged
    shutil.copy(local_dir / "meta" / "tasks.parquet", work_dir / "meta" / "tasks.parquet")

    print(f"  New total frames: {len(trimmed_df)} (was {len(df)})")

    # ------------------------------------------------------------------ #
    # 4. Download, trim, and re-encode each video file
    # ------------------------------------------------------------------ #
    vid_file_col = f"videos/{VIDEO_KEY}/file_index"
    vid_chunk_col = f"videos/{VIDEO_KEY}/chunk_index"
    vid_from_col = f"videos/{VIDEO_KEY}/from_timestamp"
    vid_to_col = f"videos/{VIDEO_KEY}/to_timestamp"

    # Group episodes by their video file
    video_file_groups: dict[tuple[int,int], list[int]] = {}
    for _, row in orig_ep_meta.iterrows():
        key = (int(row[vid_chunk_col]), int(row[vid_file_col]))
        video_file_groups.setdefault(key, []).append(int(row["episode_index"]))

    n_video_files = len(video_file_groups)
    print(f"\n[4/5] Downloading and trimming {n_video_files} video file(s) ({VIDEO_KEY})...")

    for (chunk_idx, file_idx), ep_ids in sorted(video_file_groups.items()):
        video_repo_path = f"videos/{VIDEO_KEY}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
        src_video = Path(
            hf_hub_download(
                repo_id=args.src,
                repo_type="dataset",
                filename=video_repo_path,
                token=token,
            )
        )
        print(f"  chunk={chunk_idx} file={file_idx}: {src_video.stat().st_size/1e6:.1f} MB, eps {min(ep_ids)}-{max(ep_ids)}")

        # Build keep ranges as frame indices within this video file.
        # Each episode's from_timestamp marks where it starts in the file.
        # Frame index within the file = round(from_timestamp * fps) + frame_offset
        keep_ranges_local = []
        for ep_id in sorted(ep_ids):
            ep_rows = df[df["episode_index"] == ep_id]
            from_ts = float(orig_ep_meta.loc[orig_ep_meta["episode_index"] == ep_id, vid_from_col].iloc[0])
            file_frame_start = round(from_ts * fps)
            lead, trail = trim_map[ep_id]
            n = len(ep_rows)
            keep_start = file_frame_start + lead
            keep_end = file_frame_start + n - trail
            if keep_start < keep_end:
                keep_ranges_local.append((keep_start, keep_end))

        out_video = work_dir / "videos" / VIDEO_KEY / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        trim_video(src_video, out_video, keep_ranges_local, fps=fps)

    # ------------------------------------------------------------------ #
    # 5. Push everything to HF Hub
    # ------------------------------------------------------------------ #
    print(f"\n[5/5] Pushing to {args.dst}...")

    try:
        api.repo_info(repo_id=args.dst, repo_type="dataset")
        print("  Repo exists, uploading files...")
    except Exception:
        print("  Creating new repo...")
        api.create_repo(repo_id=args.dst, repo_type="dataset", private=False)

    api.upload_folder(
        folder_path=str(work_dir),
        repo_id=args.dst,
        repo_type="dataset",
        commit_message="Trim leading/trailing frozen action frames per episode",
    )

    print(f"\nDone! Dataset pushed to https://huggingface.co/datasets/{args.dst}")
    print(f"Frames removed: {len(df) - len(trimmed_df)} ({(len(df) - len(trimmed_df)) / len(df) * 100:.1f}%)")

    shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
