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


def detect_bounds(actions: np.ndarray, vel_threshold: float = 0.02) -> tuple[int, int]:
    """Return (lead, trail) frame counts to trim at start/end of episode.

    Trims all boundary frames where the per-frame action change (L2 norm)
    is below vel_threshold.  More aggressive than exact-equality matching:
    catches slow drift and gradual settling, not just perfectly frozen frames.
    Raise vel_threshold for heavier cutting.
    """
    n = len(actions)
    # deltas[i] = L2 norm of change from frame i-1 to frame i; deltas[0] = 0
    deltas = np.concatenate([[0.0], np.linalg.norm(np.diff(actions, axis=0), axis=1)])

    lead = 0
    for i in range(n):
        if deltas[i] > vel_threshold:
            lead = i
            break
    else:
        lead = n  # entire episode is below threshold

    trail = 0
    for i in range(n - 1, -1, -1):
        if deltas[i] > vel_threshold:
            trail = n - 1 - i
            break
    else:
        trail = n

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
    return stats


def parse_source_tasks(tasks_raw) -> dict[int, str]:
    """Parse a source meta/tasks.parquet into {task_index: task_string}.

    Tolerant of layout: the task string may be a column ('tasks' or 'task')
    or the DataFrame index; 'task_index' may be a column or the index.
    """
    if "task_index" in tasks_raw.columns:
        if "tasks" in tasks_raw.columns:
            strs = tasks_raw["tasks"]
        elif "task" in tasks_raw.columns:
            strs = tasks_raw["task"]
        else:
            strs = tasks_raw.index.to_series()
        return {int(i): str(s) for i, s in zip(tasks_raw["task_index"], strs)}
    # task_index stored as the DataFrame index
    col = "tasks" if "tasks" in tasks_raw.columns else "task"
    return {int(i): str(s) for s, i in zip(tasks_raw[col], tasks_raw.index)}


def build_task_mapping(orig_ep_meta, idx_to_str_src: dict[int, str]):
    """Build the authoritative task mapping from the source episodes metadata.

    The episodes-meta `tasks` field is treated as ground truth: it reflects any
    label corrections applied after data collection, whereas the per-frame
    `task_index` in the data parquet may be stale.  The source tasks.parquet
    numbering is preserved; task strings that appear only in the episodes
    metadata are appended with fresh indices.

    Returns (idx_to_str, ep_to_task_idx).
    """
    str_to_idx = {s: i for i, s in idx_to_str_src.items()}
    idx_to_str = dict(idx_to_str_src)
    next_idx = (max(idx_to_str) + 1) if idx_to_str else 0

    ep_to_task_idx: dict[int, int] = {}
    for _, row in orig_ep_meta.iterrows():
        ep_id = int(row["episode_index"])
        tasks_field = row["tasks"]
        task_list = list(tasks_field) if tasks_field is not None else []
        if len(task_list) != 1:
            raise ValueError(
                f"Episode {ep_id} has {len(task_list)} task(s) {task_list!r}; "
                "this script assumes exactly one task per episode."
            )
        task_str = str(task_list[0])
        if task_str not in str_to_idx:
            str_to_idx[task_str] = next_idx
            idx_to_str[next_idx] = task_str
            next_idx += 1
        ep_to_task_idx[ep_id] = str_to_idx[task_str]

    return idx_to_str, ep_to_task_idx


def format_image_stats(orig_row, raw: dict, video_key: str) -> dict:
    """Convert raw per-channel image stats accumulated over the trimmed video frames
    into LeRobot episode-stats columns, matching the key shapes and value scale used
    by the source dataset."""
    prefix = f"stats/{video_key}/"
    out = {}

    scale = 1.0
    max_key = prefix + "max"
    if max_key in orig_row.index:
        if float(np.asarray(orig_row[max_key], dtype=float).max()) <= 1.5:
            scale = 255.0

    channel_vals = {
        "mean": np.asarray(raw["mean"], dtype=float) / scale,
        "std": np.asarray(raw["std"], dtype=float) / scale,
        "min": np.asarray(raw["min"], dtype=float) / scale,
        "max": np.asarray(raw["max"], dtype=float) / scale,
    }
    for kind, val in channel_vals.items():
        key = prefix + kind
        if key not in orig_row.index:
            continue
        orig_shape = np.asarray(orig_row[key], dtype=float).shape
        out[key] = val.reshape(orig_shape).tolist()

    count_key = prefix + "count"
    if count_key in orig_row.index:
        orig_count = np.asarray(orig_row[count_key])
        out[count_key] = np.full(orig_count.shape, int(raw["count"]), dtype=np.int64).tolist()

    return out


def build_trimmed_parquets(df, trim_map: dict[int, tuple[int, int]], orig_ep_meta,
                           out_dir: Path, fps: float = 10.0,
                           ep_to_task_idx: dict[int, int] | None = None):
    """Rebuild data parquet files after trimming.

    If ep_to_task_idx is given, the per-frame `task_index` column is rewritten
    to the authoritative value derived from the episodes metadata (the data
    parquet's own task_index may be stale).
    """
    import pandas as pd

    ep_to_datafile = {
        int(r["episode_index"]): (int(r["data/chunk_index"]), int(r["data/file_index"]))
        for _, r in orig_ep_meta.iterrows()
    }
    file_to_eps: dict[tuple[int, int], list[int]] = {}
    for ep_id, loc in ep_to_datafile.items():
        file_to_eps.setdefault(loc, []).append(ep_id)

    trimmed_chunks = []
    global_idx = 0
    for chunk_idx, file_idx in sorted(file_to_eps):
        file_rows = []
        for ep_id in sorted(file_to_eps[(chunk_idx, file_idx)]):
            ep = (
                df[df["episode_index"] == ep_id]
                .sort_values("frame_index")
                .reset_index(drop=True)
            )
            lead, trail = trim_map[ep_id]
            end = len(ep) - trail if trail > 0 else len(ep)
            ep = ep.iloc[lead:end].copy().reset_index(drop=True)

            ep["frame_index"] = np.arange(len(ep), dtype=np.int64)
            ep["timestamp"] = np.round(np.arange(len(ep)) / fps, 1).astype(np.float32)
            ep["index"] = np.arange(global_idx, global_idx + len(ep), dtype=np.int64)
            if ep_to_task_idx is not None and "task_index" in ep.columns:
                ep["task_index"] = np.int64(ep_to_task_idx[ep_id])
            global_idx += len(ep)
            file_rows.append(ep)

        file_df = pd.concat(file_rows, ignore_index=True)
        out_path = out_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(file_df, preserve_index=False), out_path)
        trimmed_chunks.append(file_df)

    trimmed_df = pd.concat(trimmed_chunks, ignore_index=True)
    return trimmed_df, ep_to_datafile


def build_episode_metadata(trimmed_df, orig_ep_meta, trim_map: dict, fps: float, out_dir: Path,
                           orig_ep_files_per_chunk: int = 300,
                           ep_to_task_idx: dict | None = None,
                           ep_to_datafile: dict | None = None,
                           image_stats: dict | None = None):
    """Rebuild meta/episodes parquet with updated lengths, indices, timestamps, and stats.

    The per-episode `tasks` field is preserved verbatim from the source metadata
    (treated as ground truth).  Only a `task_index` column, if present, is
    realigned to ep_to_task_idx so it stays consistent with `tasks`.
    """
    import pandas as pd

    vid_file_col = f"videos/{VIDEO_KEY}/file_index"
    vid_chunk_col = f"videos/{VIDEO_KEY}/chunk_index"
    vid_from_col = f"videos/{VIDEO_KEY}/from_timestamp"
    vid_to_col = f"videos/{VIDEO_KEY}/to_timestamp"

    video_groups: dict[tuple[int, int], list] = {}
    for _, row in orig_ep_meta.iterrows():
        key = (int(row[vid_chunk_col]), int(row[vid_file_col]))
        video_groups.setdefault(key, []).append(row)

    ep_new_timestamps: dict[int, tuple[float, float]] = {}
    for rows in video_groups.values():
        rows_sorted = sorted(rows, key=lambda r: float(r[vid_from_col]))
        cumulative = 0
        for row in rows_sorted:
            ep_id = int(row["episode_index"])
            new_len = len(trimmed_df[trimmed_df["episode_index"] == ep_id])
            new_from_ts = round(cumulative / fps, 1)
            new_to_ts = round((cumulative + new_len - 1) / fps, 1)
            ep_new_timestamps[ep_id] = (new_from_ts, new_to_ts)
            cumulative += new_len

    new_rows = []
    for _, orig_row in orig_ep_meta.iterrows():
        ep_id = int(orig_row["episode_index"])
        ep_data = trimmed_df[trimmed_df["episode_index"] == ep_id].reset_index(drop=True)
        new_from_ts, new_to_ts = ep_new_timestamps[ep_id]

        row = orig_row.copy()
        row["length"] = len(ep_data)
        row["dataset_from_index"] = int(ep_data["index"].iloc[0])
        row["dataset_to_index"] = int(ep_data["index"].iloc[-1]) + 1
        row[vid_from_col] = new_from_ts
        row[vid_to_col] = new_to_ts

        if ep_to_datafile is not None and ep_id in ep_to_datafile:
            chunk_i, file_i = ep_to_datafile[ep_id]
            if "data/chunk_index" in row.index:
                row["data/chunk_index"] = chunk_i
            if "data/file_index" in row.index:
                row["data/file_index"] = file_i

        ep_stats = recompute_episode_stats(ep_data)
        for k, v in ep_stats.items():
            if k in row.index:
                row[k] = v

        if image_stats is not None and ep_id in image_stats:
            for k, v in format_image_stats(orig_row, image_stats[ep_id], VIDEO_KEY).items():
                if k in row.index:
                    row[k] = v

        # `tasks` is left exactly as in the source metadata (ground truth).
        if ep_to_task_idx is not None and "task_index" in row.index:
            row["task_index"] = ep_to_task_idx[ep_id]

        new_rows.append(row)

    new_meta = pd.DataFrame(new_rows).reset_index(drop=True)

    out_ep_dir = out_dir / "meta" / "episodes" / "chunk-000"
    out_ep_dir.mkdir(parents=True, exist_ok=True)
    for file_idx, start in enumerate(range(0, len(new_meta), orig_ep_files_per_chunk)):
        chunk = new_meta.iloc[start : start + orig_ep_files_per_chunk]
        out_path = out_ep_dir / f"file-{file_idx:03d}.parquet"
        pq.write_table(pa.Table.from_pandas(chunk, preserve_index=False), out_path)

    return new_meta


def trim_video(src_video: Path, out_video: Path,
               keep_ranges: list[tuple[int, int, int]], fps: float) -> dict:
    """Re-encode src_video keeping only requested frames and accumulate pixel statistics."""
    keep_set = set()
    frame_to_ep: dict[int, int] = {}
    for s, e, ep_id in keep_ranges:
        for f in range(s, e):
            keep_set.add(f)
            frame_to_ep[f] = ep_id

    acc = {
        ep_id: {
            "sum": np.zeros(3, dtype=np.float64),
            "sumsq": np.zeros(3, dtype=np.float64),
            "min": np.full(3, np.inf),
            "max": np.full(3, -np.inf),
            "npix": 0,
            "nframes": 0,
        }
        for _, _, ep_id in keep_ranges
    }

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

                rgb = frame.to_ndarray(format="rgb24").astype(np.float64)
                a = acc[frame_to_ep[frame_idx]]
                a["sum"] += rgb.sum(axis=(0, 1))
                a["sumsq"] += np.square(rgb).sum(axis=(0, 1))
                a["min"] = np.minimum(a["min"], rgb.min(axis=(0, 1)))
                a["max"] = np.maximum(a["max"], rgb.max(axis=(0, 1)))
                a["npix"] += rgb.shape[0] * rgb.shape[1]
                a["nframes"] += 1

                arr = frame.to_ndarray(format="yuv420p")
                out_frame = av.VideoFrame.from_ndarray(arr, format="yuv420p")
                out_frame.pts = out_pts
                out_pts += 1
                for pkt in out_stream.encode(out_frame):
                    dst.mux(pkt)

        for pkt in out_stream.encode():
            dst.mux(pkt)

    print(f"  {out_video.name}: {out_pts} frames written")

    ep_stats = {}
    for ep_id, a in acc.items():
        if a["npix"] == 0:
            continue
        mean = a["sum"] / a["npix"]
        var = np.maximum(a["sumsq"] / a["npix"] - np.square(mean), 0.0)
        ep_stats[ep_id] = {
            "mean": mean,
            "std": np.sqrt(var),
            "min": a["min"],
            "max": a["max"],
            "count": a["nframes"],
        }
    return ep_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source HF repo (e.g. Alessio03/task1dataset)")
    parser.add_argument("--dst", required=True, help="Destination HF repo (e.g. Alessio03/task1dataset_clean)")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--vel-threshold", type=float, default=2,
                        help="L2 action-velocity threshold below which boundary frames are trimmed (default 0.02; raise for heavier cuts)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")

    # ------------------------------------------------------------------ #
    # 1. Download meta + data parquets
    # ------------------------------------------------------------------ #
    print("\n[1/6] Downloading dataset metadata and parquets...")
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
    print("\n[2/6] Computing trim bounds per episode...")
    import pandas as pd

    parquets = sorted((local_dir / "data").rglob("*.parquet"))
    df = pd.concat([pq.read_table(p).to_pandas() for p in parquets], ignore_index=True)

    with open(local_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = float(info["fps"])

    ep_meta_files = sorted((local_dir / "meta" / "episodes" / "chunk-000").glob("*.parquet"))
    orig_ep_meta = pd.concat(
        [pq.read_table(p).to_pandas() for p in ep_meta_files], ignore_index=True
    )

    # Column name constants used throughout the rest of this function.
    vid_file_col = f"videos/{VIDEO_KEY}/file_index"
    vid_chunk_col = f"videos/{VIDEO_KEY}/chunk_index"
    vid_from_col = f"videos/{VIDEO_KEY}/from_timestamp"

    # ---------------------------------------------------------------------- #
    # Sanity-check and repair source meta/episodes before doing anything else.
    # The recording / push pipeline can leave duplicate episode_index rows and
    # corrupted from_timestamp values (pointing past the MP4 end).  Both bugs
    # propagate straight into the clean dataset if we don't catch them here.
    # ---------------------------------------------------------------------- #

    # 1. Deduplicate: keep the LAST occurrence per episode_index.
    #    The append/rewrite bug that creates duplicates appends corrected rows at
    #    the end, so the last row carries the most recently corrected task string,
    #    video file assignment, and from_timestamp.  (Any still-wrong from_timestamp
    #    in the last row is fixed by the recomputation in step 2 below.)
    n_meta_rows = len(orig_ep_meta)
    if orig_ep_meta["episode_index"].duplicated().any():
        n_unique = orig_ep_meta["episode_index"].nunique()
        print(f"  WARNING: meta/episodes has {n_meta_rows} rows for {n_unique} unique episodes.")
        orig_ep_meta = (
            orig_ep_meta
            .drop_duplicates(subset="episode_index", keep="last")
            .sort_values("episode_index")
            .reset_index(drop=True)
        )
        print(f"  Removed {n_meta_rows - len(orig_ep_meta)} duplicate rows; "
              f"kept {len(orig_ep_meta)} unique episodes.")

    # Warn if info.json disagrees with the actual unique episode count.
    if info.get("total_episodes") != len(orig_ep_meta):
        print(f"  WARNING: Source info.json total_episodes={info.get('total_episodes')} "
              f"but actual unique episodes in meta = {len(orig_ep_meta)}. Trusting meta count.")

    # 2. Recompute source from_timestamp from the data parquet frame counts.
    #    The source metadata's from_timestamp may be wrong (global vs. local
    #    timestamps, or stale after an append/rewrite bug).  We recompute it as
    #    cumulative frame counts of preceding episodes within the same video file,
    #    assuming episodes are stored in ascending episode_index order inside each
    #    file — which is always true for standard LeRobot recordings.
    ep_to_vidfile: dict[int, tuple[int, int]] = {
        int(r["episode_index"]): (int(r[vid_chunk_col]), int(r[vid_file_col]))
        for _, r in orig_ep_meta.iterrows()
    }
    vidfile_to_eps: dict[tuple[int, int], list[int]] = {}
    for ep_id, key in ep_to_vidfile.items():
        vidfile_to_eps.setdefault(key, []).append(ep_id)

    ep_corrected_from_ts: dict[int, float] = {}
    n_ts_corrections = 0
    for key, ep_ids in vidfile_to_eps.items():
        cumulative_frames = 0
        for ep_id in sorted(ep_ids):  # ascending episode_index = storage order in file
            expected_ts = cumulative_frames / fps
            orig_ts = float(
                orig_ep_meta.loc[orig_ep_meta["episode_index"] == ep_id, vid_from_col].iloc[0]
            )
            if abs(orig_ts - expected_ts) > 0.5 / fps:
                n_ts_corrections += 1
            ep_corrected_from_ts[ep_id] = expected_ts
            cumulative_frames += int((df["episode_index"] == ep_id).sum())

    if n_ts_corrections:
        print(f"  WARNING: {n_ts_corrections} episode(s) had inconsistent source from_timestamp.")
        print(f"  Recomputing all source from_timestamp values from data parquet frame counts.")
        for idx, row in orig_ep_meta.iterrows():
            orig_ep_meta.at[idx, vid_from_col] = ep_corrected_from_ts[int(row["episode_index"])]
    else:
        print("  Source from_timestamp values are consistent with data parquet frame counts.")

    # Authoritative task mapping: trust the episodes-meta `tasks` field.
    # The per-frame `task_index` in the data parquet may be stale (e.g. labels
    # corrected after collection); we rebuild it from the metadata below.
    tasks_raw = pd.read_parquet(local_dir / "meta" / "tasks.parquet")
    idx_to_str_src = parse_source_tasks(tasks_raw)
    idx_to_str, ep_to_task_idx = build_task_mapping(orig_ep_meta, idx_to_str_src)

    stale = []
    for ep_id in sorted(df["episode_index"].unique()):
        ep_id = int(ep_id)
        data_idxs = sorted(
            int(x) for x in df.loc[df["episode_index"] == ep_id, "task_index"].unique()
        )
        truth = ep_to_task_idx[ep_id]
        if data_idxs != [truth]:
            stale.append((ep_id, data_idxs, truth))
    if stale:
        print(f"  WARNING: {len(stale)} episode(s) have a per-frame task_index that "
              f"disagrees with the episodes-meta `tasks` field.")
        print(f"  Trusting the metadata; rewriting task_index for these episodes:")
        for ep_id, data_idxs, truth in stale:
            print(f"    ep{ep_id:3d}: data task_index={data_idxs} -> {truth} "
                  f"({idx_to_str[truth]!r})")
    else:
        print("  All per-frame task_index values agree with the episodes metadata.")

    trim_map = {}
    total_trimmed = 0
    for ep_id in sorted(df["episode_index"].unique()):
        ep = df[df["episode_index"] == ep_id]
        actions = np.stack(ep["action"].values)
        lead, trail = detect_bounds(actions, vel_threshold=args.vel_threshold)
        trim_map[ep_id] = (lead, trail)
        trimmed = lead + trail
        total_trimmed += trimmed
        marker = "  >>>" if (lead > 10 or trail > 10) else ""
        print(f"  ep{ep_id:3d}: -{lead:3d} start / -{trail:3d} end  ({trimmed:3d} frames removed){marker}")

    print(f"\n  Total frames to remove: {total_trimmed} / {len(df)}")

    # ------------------------------------------------------------------ #
    # 3. Build trimmed data parquets in temp dir
    # ------------------------------------------------------------------ #
    print("\n[3/6] Building trimmed data parquets...")
    work_dir = Path(tempfile.mkdtemp(prefix="lerobot_trim_"))
    print(f"  Working directory: {work_dir}")

    trimmed_df, ep_to_datafile = build_trimmed_parquets(
        df, trim_map, orig_ep_meta, work_dir, fps=fps, ep_to_task_idx=ep_to_task_idx)
    print(f"  New total frames: {len(trimmed_df)} (was {len(df)})")

    # ------------------------------------------------------------------ #
    # 4. Download, trim, and re-encode each video file.
    # ------------------------------------------------------------------ #
    video_file_groups: dict[tuple[int, int], list[int]] = {}
    for _, row in orig_ep_meta.iterrows():
        key = (int(row[vid_chunk_col]), int(row[vid_file_col]))
        video_file_groups.setdefault(key, []).append(int(row["episode_index"]))

    n_video_files = len(video_file_groups)
    print(f"\n[4/6] Downloading and trimming {n_video_files} video file(s) ({VIDEO_KEY})...")

    image_stats_raw: dict[int, dict] = {}
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

        keep_ranges_local = []
        ep_ids_sorted = sorted(ep_ids, key=lambda eid: float(
            orig_ep_meta.loc[orig_ep_meta["episode_index"] == eid, vid_from_col].iloc[0]))
        for ep_id in ep_ids_sorted:
            ep_rows = df[df["episode_index"] == ep_id]
            from_ts = float(orig_ep_meta.loc[orig_ep_meta["episode_index"] == ep_id, vid_from_col].iloc[0])
            file_frame_start = round(from_ts * fps)
            lead, trail = trim_map[ep_id]
            n = len(ep_rows)
            keep_start = file_frame_start + lead
            keep_end = file_frame_start + n - trail
            if keep_start < keep_end:
                keep_ranges_local.append((keep_start, keep_end, ep_id))

        out_video = work_dir / "videos" / VIDEO_KEY / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        image_stats_raw.update(trim_video(src_video, out_video, keep_ranges_local, fps=fps))

    # ------------------------------------------------------------------ #
    # 5. Build episode metadata (with recomputed stats) + remaining meta
    # ------------------------------------------------------------------ #
    print("\n[5/6] Building episode metadata and meta files...")

    rows_per_ep_file = max(len(pq.read_table(p).to_pandas()) for p in ep_meta_files)
    new_ep_meta = build_episode_metadata(trimmed_df, orig_ep_meta, trim_map, fps, work_dir,
                                         orig_ep_files_per_chunk=rows_per_ep_file,
                                         ep_to_task_idx=ep_to_task_idx,
                                         ep_to_datafile=ep_to_datafile,
                                         image_stats=image_stats_raw)

    new_info = dict(info)
    new_info["total_frames"] = len(trimmed_df)
    new_info["total_episodes"] = len(new_ep_meta)
    (work_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(work_dir / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    # Write meta/tasks.parquet in the canonical LeRobot v3 layout (matching the
    # source): index = task string (name "task"), single column "task_index".
    sorted_tasks = sorted(idx_to_str.items())
    tasks_out = pd.DataFrame(
        {"task_index": [idx for idx, _ in sorted_tasks]},
        index=pd.Index([task for _, task in sorted_tasks], name="task"),
    )
    tasks_out.to_parquet(work_dir / "meta" / "tasks.parquet")

    for src_meta_file in (local_dir / "meta").rglob("*"):
        if not src_meta_file.is_file():
            continue
        rel = src_meta_file.relative_to(local_dir / "meta")
        dst_meta_file = work_dir / "meta" / rel
        if not dst_meta_file.exists():
            dst_meta_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_meta_file, dst_meta_file)

    readme_src = local_dir / "README.md"
    if readme_src.exists():
        readme_text = readme_src.read_text().replace(args.src, args.dst)
        (work_dir / "README.md").write_text(readme_text)
        print("  README adapted from source repo.")

    # ------------------------------------------------------------------ #
    # 6. Push everything to HF Hub
    # ------------------------------------------------------------------ #
    print(f"\n[6/6] Pushing to {args.dst}...")

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
        delete_patterns=[
            "data/chunk-*/*.parquet",
            "meta/episodes/chunk-*/*.parquet",
        ],
    )

    print(f"\nDone! Dataset pushed to https://huggingface.co/datasets/{args.dst}")
    print(f"Frames removed: {len(df) - len(trimmed_df)} ({(len(df) - len(trimmed_df)) / len(df) * 100:.1f}%)")

    shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()