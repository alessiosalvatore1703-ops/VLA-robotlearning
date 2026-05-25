#!/usr/bin/env python3
"""Repair duplicate LeRobot episode metadata and invalid video timestamps.

This utility is stricter than a plain timestamp clipper:

1. It handles duplicate rows in ``meta/episodes`` with the same
   ``episode_index``.
2. For each episode, it chooses the metadata row that matches the actual MP4
   timestamps best.
3. It clips only frames whose video timestamps are not decodable.
4. It rebuilds episode/data indices and writes fresh stats.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clip_lerobot_to_video_timestamps import _set_feature_stats
from move_episodes_between_datasets import _jsonify, _stats_from_data_parquets


@dataclass
class CandidateScore:
    row_index: int
    valid_mask: np.ndarray
    valid_count: int
    invalid_count: int
    boundary_match: bool
    full_valid: bool
    row: dict[str, Any]


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
    print(f"Uploading repaired dataset to {repo_id} ...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Repair duplicate episode metadata and video timestamps",
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
    return sorted(
        {
            col[len("videos/") : -len("/from_timestamp")]
            for col in episodes.columns
            if col.startswith("videos/") and col.endswith("/from_timestamp")
        }
    )


def _decode_video_times(path: Path) -> np.ndarray:
    import av

    times: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is not None:
                times.append(float(frame.time))
    if not times:
        raise ValueError(f"No decodable video timestamps in {path}")
    return np.asarray(times, dtype=np.float64)


def _collect_video_times(root: Path, episodes: pd.DataFrame, keys: list[str]) -> dict[tuple[str, int, int], np.ndarray]:
    out: dict[tuple[str, int, int], np.ndarray] = {}
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
            out[(key, chunk, file_idx)] = _decode_video_times(path)
    return out


def _nearest_valid_mask(times: np.ndarray, query: np.ndarray, tolerance_s: float) -> np.ndarray:
    if query.size == 0:
        return np.zeros(0, dtype=bool)
    in_range = (query >= times.min() - tolerance_s) & (query <= times.max() + tolerance_s)
    idx = np.searchsorted(times, query)
    idx0 = np.clip(idx - 1, 0, len(times) - 1)
    idx1 = np.clip(idx, 0, len(times) - 1)
    nearest_delta = np.minimum(np.abs(times[idx0] - query), np.abs(times[idx1] - query))
    return in_range & (nearest_delta <= tolerance_s)


def _format_task_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _score_candidate(
    row_index: int,
    row: pd.Series,
    ep_data: pd.DataFrame,
    video_keys: list[str],
    video_times: dict[tuple[str, int, int], np.ndarray],
    tolerance_s: float,
) -> CandidateScore:
    timestamp = ep_data["timestamp"].astype(float).to_numpy()
    valid = np.ones(len(ep_data), dtype=bool)
    raw_row = row.to_dict()

    for key in video_keys:
        chunk = int(raw_row[f"videos/{key}/chunk_index"])
        file_idx = int(raw_row[f"videos/{key}/file_index"])
        times = video_times[(key, chunk, file_idx)]
        query = float(raw_row[f"videos/{key}/from_timestamp"]) + timestamp
        valid &= _nearest_valid_mask(times, query, tolerance_s)

    valid_count = int(valid.sum())
    invalid_count = int((~valid).sum())
    boundary_match = True
    if "dataset_from_index" in raw_row and "index" in ep_data.columns and not ep_data.empty:
        boundary_match &= int(raw_row["dataset_from_index"]) == int(ep_data["index"].min())
    if "dataset_to_index" in raw_row and "index" in ep_data.columns and not ep_data.empty:
        boundary_match &= int(raw_row["dataset_to_index"]) == int(ep_data["index"].max()) + 1

    return CandidateScore(
        row_index=row_index,
        valid_mask=valid,
        valid_count=valid_count,
        invalid_count=invalid_count,
        boundary_match=boundary_match,
        full_valid=valid_count == len(ep_data),
        row=raw_row,
    )


def _choose_candidate(scores: list[CandidateScore]) -> CandidateScore:
    return sorted(
        scores,
        key=lambda score: (
            score.valid_count,
            int(score.full_valid),
            int(score.boundary_match),
            -score.invalid_count,
            -score.row_index,
        ),
        reverse=True,
    )[0]


def _rewrite(root: Path, tolerance_s: float) -> dict[str, Any]:
    data = _load_data(root)
    episodes = _load_episodes(root)
    video_keys = _video_keys(episodes)
    if not video_keys:
        raise ValueError("No video timestamp columns found in episode metadata")
    video_times = _collect_video_times(root, episodes, video_keys)

    old_episode_indices = sorted(int(value) for value in data["episode_index"].unique().tolist())
    data_cols = set(data.columns)
    new_data_frames: list[pd.DataFrame] = []
    new_rows: list[dict[str, Any]] = []
    dropped_episodes: list[int] = []
    clipped_episodes: dict[int, int] = {}
    duplicate_meta_episodes: dict[int, int] = {}
    selected_rows: dict[int, int] = {}
    global_index = 0

    for old_ep_idx in old_episode_indices:
        ep_rows = episodes[episodes["episode_index"].astype(int) == old_ep_idx]
        ep_data = data[data["episode_index"].astype(int) == old_ep_idx].copy()
        if ep_data.empty or ep_rows.empty:
            dropped_episodes.append(old_ep_idx)
            continue

        if len(ep_rows) > 1:
            duplicate_meta_episodes[old_ep_idx] = int(len(ep_rows))

        scores = [
            _score_candidate(int(row_idx), row, ep_data, video_keys, video_times, tolerance_s)
            for row_idx, row in ep_rows.iterrows()
        ]
        best = _choose_candidate(scores)
        selected_rows[old_ep_idx] = best.row_index

        if best.valid_count == 0:
            dropped_episodes.append(old_ep_idx)
            continue

        kept = ep_data.loc[best.valid_mask].copy()
        dropped_frames = int(len(ep_data) - len(kept))
        if dropped_frames:
            clipped_episodes[old_ep_idx] = dropped_frames

        first_ts = float(kept["timestamp"].iloc[0])
        row = dict(best.row)
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
        if "tasks" in row:
            row["tasks"] = _format_task_value(row["tasks"])

        for key in video_keys:
            row[f"videos/{key}/to_timestamp"] = (
                float(row[f"videos/{key}/from_timestamp"]) + float(kept["timestamp"].max())
            )

        for stats_key in [key for key in row if key.startswith("stats/")]:
            del row[stats_key]
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
    if (root / "meta" / "tasks.parquet").exists():
        info["total_tasks"] = int(len(pd.read_parquet(root / "meta" / "tasks.parquet")))
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(new_rows)}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    stats = _stats_from_data_parquets(root)
    with open(root / "meta" / "stats.json", "w") as f:
        json.dump(_jsonify(stats), f, indent=2)

    return {
        "episodes_before_meta_rows": int(len(episodes)),
        "episodes_before_unique_data": int(len(old_episode_indices)),
        "episodes_after": int(len(new_rows)),
        "frames_before": int(len(data)),
        "frames_after": int(global_index),
        "dropped_episodes": dropped_episodes,
        "duplicate_meta_episodes": duplicate_meta_episodes,
        "clipped_episodes": clipped_episodes,
        "selected_source_meta_rows": selected_rows,
    }


def _validate(root: Path, tolerance_s: float) -> None:
    data = _load_data(root)
    episodes = _load_episodes(root)
    video_keys = _video_keys(episodes)
    video_times = _collect_video_times(root, episodes, video_keys)

    if episodes["episode_index"].astype(int).tolist() != list(range(len(episodes))):
        raise ValueError("episode_index is not consecutive")
    if episodes["episode_index"].duplicated().any():
        raise ValueError("episode_index is duplicated in meta/episodes")
    if data["index"].astype(int).tolist() != list(range(len(data))):
        raise ValueError("global index is not consecutive")

    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        ep_data = data[data["episode_index"].astype(int) == ep_idx]
        if ep_data.empty:
            raise ValueError(f"Episode {ep_idx} is empty")
        if int(row["length"]) != len(ep_data):
            raise ValueError(f"Episode {ep_idx} length mismatch")
        if "frame_index" in ep_data.columns:
            frames = ep_data["frame_index"].astype(int).tolist()
            if frames != list(range(len(ep_data))):
                raise ValueError(f"Episode {ep_idx} frame_index is not consecutive")

        timestamp = ep_data["timestamp"].astype(float).to_numpy()
        for key in video_keys:
            chunk = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            times = video_times[(key, chunk, file_idx)]
            query = float(row[f"videos/{key}/from_timestamp"]) + timestamp
            if not _nearest_valid_mask(times, query, tolerance_s).all():
                raise ValueError(f"Episode {ep_idx} still has invalid video timestamps for {key}")

    if not (root / "meta" / "stats.json").exists():
        raise ValueError("Missing meta/stats.json")


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
            tmp_input = tempfile.TemporaryDirectory(prefix="lerobot_meta_repair_src_")
            src = Path(tmp_input.name) / "dataset"
            _download_hf_dataset(args.input, src)
        else:
            src = Path(args.input)
            if not src.is_dir():
                raise SystemExit(f"Input path does not exist: {src}")

        if output_is_hf:
            tmp_output = tempfile.TemporaryDirectory(prefix="lerobot_meta_repair_dst_")
            dst = Path(tmp_output.name) / "dataset"
        else:
            dst = Path(args.output)
            if dst.exists():
                raise SystemExit(f"Output path already exists: {dst}")

        print("Copying dataset ...")
        shutil.copytree(src, dst, symlinks=False)
        stats = _rewrite(dst, tolerance_s=args.tolerance_s)
        _validate(dst, tolerance_s=args.tolerance_s)

        print("Repair summary")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        if args.push_to_hub:
            _push_to_hub(dst, args.output)
        else:
            print(f"Saved repaired dataset to {dst}")
    finally:
        if tmp_input is not None:
            tmp_input.cleanup()
        if tmp_output is not None:
            tmp_output.cleanup()


if __name__ == "__main__":
    main()
