#!/usr/bin/env python3
"""Repair LeRobot datasets by clipping frames to actual MP4 timestamp ranges.

This handles cases where ``meta/episodes`` says a video segment extends past
the real frames contained in the referenced MP4. It may drop empty episodes
when no frame in an episode is decodable from the referenced video.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from move_episodes_between_datasets import _jsonify, _stats_from_data_parquets
from clip_lerobot_to_video_timestamps import _set_feature_stats


def _is_hf_repo_id(value: str) -> bool:
    path = Path(value)
    if path.exists():
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def _download_hf_dataset(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        force_download=True,
    )


def _push_to_hub(local_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading fixed dataset to {repo_id} ...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Clip frames to actual video timestamp ranges",
        delete_patterns="*",
    )
    print(f"Pushed -> https://huggingface.co/datasets/{repo_id}")


def _load_data(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No data parquet files found under {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")
    return (
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


def _video_keys(episodes: pd.DataFrame) -> list[str]:
    keys = []
    for col in episodes.columns:
        if col.startswith("videos/") and col.endswith("/from_timestamp"):
            keys.append(col[len("videos/") : -len("/from_timestamp")])
    return sorted(set(keys))


def _video_range(path: Path) -> tuple[float, float, int]:
    import av

    times: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is not None:
                times.append(float(frame.time))
    if not times:
        raise ValueError(f"No decodable video timestamps in {path}")
    return min(times), max(times), len(times)


def _collect_video_ranges(root: Path, episodes: pd.DataFrame, keys: list[str]) -> dict[tuple[str, int, int], tuple[float, float, int]]:
    ranges: dict[tuple[str, int, int], tuple[float, float, int]] = {}
    for key in keys:
        pairs = (
            episodes[[f"videos/{key}/chunk_index", f"videos/{key}/file_index"]]
            .drop_duplicates()
            .sort_values([f"videos/{key}/chunk_index", f"videos/{key}/file_index"])
        )
        for _, row in pairs.iterrows():
            chunk = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            path = root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
            ranges[(key, chunk, file_idx)] = _video_range(path)
    return ranges


def _rewrite(root: Path, tolerance_s: float) -> dict[str, Any]:
    data = _load_data(root)
    episodes = _load_episodes(root)
    video_keys = _video_keys(episodes)
    if not video_keys:
        raise ValueError("No video timestamp columns found in episode metadata")

    video_ranges = _collect_video_ranges(root, episodes, video_keys)

    new_data_frames: list[pd.DataFrame] = []
    new_rows: list[dict[str, Any]] = []
    dropped_rows = 0
    clipped_episodes = 0
    dropped_episodes: list[int] = []
    global_index = 0

    data_cols = set(data.columns)
    for new_ep_idx, row in enumerate([]):
        _ = new_ep_idx, row

    for _, raw_row in episodes.iterrows():
        old_ep_idx = int(raw_row["episode_index"])
        row = raw_row.to_dict()
        ep_data = data[data["episode_index"].astype(int) == old_ep_idx].copy()
        if ep_data.empty:
            dropped_episodes.append(old_ep_idx)
            continue

        keep = np.ones(len(ep_data), dtype=bool)
        timestamp = ep_data["timestamp"].astype(float).to_numpy()

        for key in video_keys:
            chunk = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            actual_min, actual_max, _ = video_ranges[(key, chunk, file_idx)]
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            to_ts = float(row[f"videos/{key}/to_timestamp"])
            abs_query = from_ts + timestamp
            keep &= abs_query >= actual_min - tolerance_s
            keep &= abs_query <= min(actual_max, to_ts) + tolerance_s

        kept = ep_data.loc[keep].copy()
        dropped_here = int((~keep).sum())
        dropped_rows += dropped_here
        if dropped_here:
            clipped_episodes += 1

        if kept.empty:
            dropped_episodes.append(old_ep_idx)
            continue

        first_ts = float(kept["timestamp"].iloc[0])
        if abs(first_ts) > tolerance_s:
            kept["timestamp"] = kept["timestamp"].astype(float) - first_ts
            for key in video_keys:
                row[f"videos/{key}/from_timestamp"] = float(row[f"videos/{key}/from_timestamp"]) + first_ts

        new_ep_idx = len(new_rows)
        kept = kept.reset_index(drop=True)
        kept["episode_index"] = np.int64(new_ep_idx)
        kept["index"] = np.arange(global_index, global_index + len(kept), dtype=np.int64)
        if "frame_index" in kept.columns:
            kept["frame_index"] = np.arange(len(kept), dtype=np.int64)

        row["episode_index"] = int(new_ep_idx)
        row["length"] = int(len(kept))
        row["dataset_from_index"] = int(global_index)
        row["dataset_to_index"] = int(global_index + len(kept))
        row["data/chunk_index"] = 0
        row["data/file_index"] = int(new_ep_idx)
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        for key in video_keys:
            row[f"videos/{key}/to_timestamp"] = (
                float(row[f"videos/{key}/from_timestamp"]) + float(kept["timestamp"].max())
            )

        for feature in [
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]:
            if feature in data_cols:
                _set_feature_stats(row, feature, kept[feature])

        new_data_frames.append(kept)
        new_rows.append(row)
        global_index += len(kept)

    if not new_rows:
        raise ValueError("Repair dropped every episode")

    data_dir = root / "data"
    shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for ep_idx, frame in enumerate(new_data_frames):
        out = data_dir / "chunk-000" / f"file-{ep_idx:03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)

    episodes_dir = root / "meta" / "episodes"
    shutil.rmtree(episodes_dir)
    out_ep = episodes_dir / "chunk-000" / "file-000.parquet"
    out_ep.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(new_rows).to_parquet(out_ep, index=False)

    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = len(new_rows)
    info["total_frames"] = int(global_index)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(new_rows)}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    stats = _stats_from_data_parquets(root)
    with open(root / "meta" / "stats.json", "w") as f:
        json.dump(_jsonify(stats), f, indent=2)

    return {
        "episodes_before": int(len(episodes)),
        "episodes_after": int(len(new_rows)),
        "frames_before": int(len(data)),
        "frames_after": int(global_index),
        "frames_dropped": int(dropped_rows),
        "episodes_clipped": int(clipped_episodes),
        "episodes_dropped": dropped_episodes,
    }


def _validate(root: Path, tolerance_s: float) -> None:
    data = _load_data(root)
    episodes = _load_episodes(root)
    video_keys = _video_keys(episodes)
    ranges = _collect_video_ranges(root, episodes, video_keys)

    if [int(v) for v in episodes["episode_index"].tolist()] != list(range(len(episodes))):
        raise ValueError("episode_index is not consecutive")
    if [int(v) for v in data["index"].tolist()] != list(range(len(data))):
        raise ValueError("global index is not consecutive")

    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        ep_data = data[data["episode_index"].astype(int) == ep_idx]
        if ep_data.empty:
            raise ValueError(f"Episode {ep_idx} is empty")
        if "frame_index" in ep_data.columns:
            frame_indices = [int(v) for v in ep_data["frame_index"].tolist()]
            if frame_indices != list(range(len(ep_data))):
                raise ValueError(f"Episode {ep_idx} frame_index is not consecutive")
        timestamp = ep_data["timestamp"].astype(float).to_numpy()
        for key in video_keys:
            chunk = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            actual_min, actual_max, _ = ranges[(key, chunk, file_idx)]
            query = float(row[f"videos/{key}/from_timestamp"]) + timestamp
            if query.min() < actual_min - tolerance_s or query.max() > actual_max + tolerance_s:
                raise ValueError(
                    f"Episode {ep_idx} still queries outside actual video range for {key}: "
                    f"{query.min()}..{query.max()} vs {actual_min}..{actual_max}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local path or HF dataset repo id")
    parser.add_argument("--output", required=True, help="Local output path or HF dataset repo id")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    args = parser.parse_args()

    input_is_hf = _is_hf_repo_id(args.input)
    output_is_hf = _is_hf_repo_id(args.output)
    if args.push_to_hub and not output_is_hf:
        parser.error("--push-to-hub requires --output to be a Hugging Face repo id")

    tmp_input: tempfile.TemporaryDirectory[str] | None = None
    tmp_output: tempfile.TemporaryDirectory[str] | None = None
    try:
        if input_is_hf:
            tmp_input = tempfile.TemporaryDirectory(prefix="lerobot_actual_clip_src_")
            src = Path(tmp_input.name) / "dataset"
            _download_hf_dataset(args.input, src)
        else:
            src = Path(args.input)
            if not src.is_dir():
                raise SystemExit(f"Input path does not exist: {src}")

        if output_is_hf:
            tmp_output = tempfile.TemporaryDirectory(prefix="lerobot_actual_clip_dst_")
            dst = Path(tmp_output.name) / "dataset"
        else:
            dst = Path(args.output)
            if dst.exists():
                raise SystemExit(f"Output path already exists: {dst}")

        print("Copying dataset ...")
        shutil.copytree(src, dst, symlinks=False)
        stats = _rewrite(dst, tolerance_s=args.tolerance_s)
        _validate(dst, tolerance_s=args.tolerance_s)

        print("Repair stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        if args.push_to_hub:
            _push_to_hub(dst, args.output)
        else:
            print(f"Saved fixed dataset to {dst}")
    finally:
        if tmp_input is not None:
            tmp_input.cleanup()
        if tmp_output is not None:
            tmp_output.cleanup()


if __name__ == "__main__":
    main()
