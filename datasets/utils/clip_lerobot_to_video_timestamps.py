#!/usr/bin/env python3
"""Drop LeRobot frames whose timestamps exceed their referenced video segment.

This repairs datasets where the tabular data rows continue past the available
video timestamps. That otherwise crashes training with FrameTimestampError.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAT_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def _is_hf_repo_id(value: str) -> bool:
    path = Path(value)
    if path.exists():
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def _download_hf_dataset(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))


def _push_to_hub(local_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Creating / updating HF dataset repo: {repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading to {repo_id} ...")
    api.upload_folder(folder_path=str(local_path), repo_id=repo_id, repo_type="dataset")
    print(f"Pushed -> https://huggingface.co/datasets/{repo_id}")


def _load_data(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No data parquets found under {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No episode parquets found under {root / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _video_keys(episodes: pd.DataFrame) -> list[str]:
    keys = []
    for col in episodes.columns:
        if col.startswith("videos/") and col.endswith("/from_timestamp"):
            keys.append(col[len("videos/") : -len("/from_timestamp")])
    return keys


def _as_numeric_matrix(values: pd.Series) -> np.ndarray | None:
    if values.empty:
        return None

    first = values.iloc[0]
    if isinstance(first, np.ndarray):
        try:
            return np.stack(values.to_list()).astype(np.float64)
        except Exception:
            return None

    if isinstance(first, (list, tuple)):
        try:
            return np.asarray(values.to_list(), dtype=np.float64)
        except Exception:
            return None

    if np.isscalar(first):
        try:
            return values.to_numpy(dtype=np.float64).reshape(-1, 1)
        except Exception:
            return None

    return None


def _format_stat(value: np.ndarray, original: Any) -> Any:
    value = np.asarray(value)
    if isinstance(original, np.ndarray):
        return value.astype(original.dtype, copy=False)
    if isinstance(original, list):
        return value.tolist()
    return value


def _set_feature_stats(row: dict[str, Any], feature: str, values: pd.Series) -> None:
    prefix = f"stats/{feature}"
    if f"{prefix}/min" not in row:
        return

    matrix = _as_numeric_matrix(values)
    if matrix is None or matrix.size == 0:
        return

    stats: dict[str, np.ndarray] = {
        "min": np.min(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "count": np.full(matrix.shape[1:], matrix.shape[0], dtype=np.int64),
    }
    for name, q in STAT_QUANTILES.items():
        stats[name] = np.quantile(matrix, q, axis=0)

    for name, value in stats.items():
        key = f"{prefix}/{name}"
        if key in row:
            row[key] = _format_stat(value, row[key])


def _repair(root: Path, tolerance_s: float) -> dict[str, int]:
    data = _load_data(root)
    episodes = _load_episodes(root).sort_values("episode_index").reset_index(drop=True)

    required = {"episode_index", "index", "timestamp"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Data is missing required columns: {sorted(missing)}")

    video_keys = _video_keys(episodes)
    if not video_keys:
        raise ValueError("No video timestamp columns found in meta/episodes")

    max_valid_ts_by_episode: dict[int, float] = {}
    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        limits = []
        for key in video_keys:
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            to_ts = float(row[f"videos/{key}/to_timestamp"])
            limits.append(to_ts - from_ts)
        max_valid_ts_by_episode[ep_idx] = min(limits)

    data["_max_valid_timestamp"] = data["episode_index"].astype(int).map(max_valid_ts_by_episode)
    keep_mask = data["timestamp"].astype(float) <= data["_max_valid_timestamp"].astype(float) + tolerance_s
    dropped = data.loc[~keep_mask].copy()
    kept = data.loc[keep_mask].drop(columns=["_max_valid_timestamp"]).copy()

    if kept.empty:
        raise ValueError("Repair would remove every frame")

    kept = kept.sort_values("index").reset_index(drop=True)
    kept["index"] = np.arange(len(kept), dtype=np.int64)
    if "frame_index" in kept.columns:
        kept["frame_index"] = kept.groupby("episode_index").cumcount().astype(np.int64)

    lengths = kept.groupby("episode_index").size().astype(int)
    if (lengths == 0).any():
        raise ValueError("Repair produced an empty episode")

    new_rows = []
    data_cols = set(kept.columns)
    for row in episodes.to_dict("records"):
        ep_idx = int(row["episode_index"])
        ep_data = kept[kept["episode_index"].astype(int) == ep_idx]
        if ep_data.empty:
            raise ValueError(f"Episode {ep_idx} has no frames after repair")

        row["length"] = int(len(ep_data))
        row["dataset_from_index"] = int(ep_data["index"].min())
        row["dataset_to_index"] = int(ep_data["index"].max()) + 1
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0

        # Recompute stats for tabular features that exist in the data parquet.
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
                _set_feature_stats(row, feature, ep_data[feature])

        new_rows.append(row)

    data_dir = root / "data"
    shutil.rmtree(data_dir)
    out_data = data_dir / "chunk-000" / "file-000.parquet"
    out_data.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_data, index=False)

    episodes_dir = root / "meta" / "episodes"
    shutil.rmtree(episodes_dir)
    out_episodes = episodes_dir / "chunk-000" / "file-000.parquet"
    out_episodes.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(new_rows).sort_values("episode_index").to_parquet(out_episodes, index=False)

    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_frames"] = int(len(kept))
    info["total_episodes"] = int(len(episodes))
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(episodes)}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    bad_episode_count = int(dropped["episode_index"].nunique()) if not dropped.empty else 0
    return {
        "episodes": int(len(episodes)),
        "frames_before": int(len(data)),
        "frames_after": int(len(kept)),
        "frames_dropped": int(len(dropped)),
        "episodes_clipped": bad_episode_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local path or HF dataset repo id")
    parser.add_argument("--output", required=True, help="Local output path or HF dataset repo id")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-6,
        help="Small tolerance for keeping timestamps at the segment boundary.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    input_is_hf = _is_hf_repo_id(args.input)
    output_is_hf = _is_hf_repo_id(args.output)
    if args.push_to_hub and not output_is_hf:
        parser.error("--push-to-hub requires --output to be a Hugging Face repo id")

    tmp_input: tempfile.TemporaryDirectory[str] | None = None
    tmp_output: tempfile.TemporaryDirectory[str] | None = None

    try:
        if input_is_hf:
            tmp_input = tempfile.TemporaryDirectory(prefix="lerobot_clip_src_")
            src = Path(tmp_input.name) / "dataset"
            _download_hf_dataset(args.input, src)
        else:
            src = Path(args.input)
            if not src.is_dir():
                raise SystemExit(f"Input path does not exist: {src}")

        if output_is_hf:
            tmp_output = tempfile.TemporaryDirectory(prefix="lerobot_clip_dst_")
            dst = Path(tmp_output.name) / "dataset"
        else:
            dst = Path(args.output)
            if dst.exists():
                raise SystemExit(f"Output path already exists: {dst}")

        print("Copying dataset ...")
        shutil.copytree(src, dst, symlinks=False)
        stats = _repair(dst, tolerance_s=args.tolerance_s)
        print("Repair stats:")
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
