#!/usr/bin/env python3
"""Move selected episodes from one LeRobot v3 dataset repo to another.

The script rewrites both datasets with clean, consecutive episode/global frame
indices while preserving observations, actions, timestamps, prompts, and video
segments. It uploads the destination first and the source second, so a failed
upload does not remove the moved episodes before they exist in the destination.
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


STAT_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


@dataclass(frozen=True)
class EpisodeRef:
    root: Path
    source_name: str
    old_episode_index: int
    row: dict[str, Any]
    data: pd.DataFrame


def _is_string_like(values: Any) -> bool:
    if pd.api.types.is_string_dtype(values):
        return True
    try:
        return all(isinstance(value, str) for value in list(values))
    except TypeError:
        return False


def _download_hf_dataset(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Force-downloading latest {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        force_download=True,
    )


def _upload_folder(local_root: Path, repo_id: str, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading clean dataset to {repo_id} ...")
    api.upload_folder(
        folder_path=str(local_root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
        delete_patterns="*",
    )
    print(f"Pushed -> https://huggingface.co/datasets/{repo_id}")


def _load_info(root: Path) -> dict[str, Any]:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def _load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No episode metadata parquet files found under {root / 'meta' / 'episodes'}")
    return (
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


def _load_data(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No data parquet files found under {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _load_tasks(root: Path) -> list[str]:
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    if _is_string_like(tasks_df.index):
        return [str(value) for value in tasks_df.index]
    if "task" in tasks_df.columns and _is_string_like(tasks_df["task"]):
        return [str(value) for value in tasks_df["task"]]
    reset_df = tasks_df.reset_index()
    for col in reset_df.columns:
        if _is_string_like(reset_df[col]):
            return [str(value) for value in reset_df[col]]
    raise ValueError(f"Could not find task strings in {root / 'meta' / 'tasks.parquet'}")


def _video_keys(info: dict[str, Any], episodes: pd.DataFrame) -> list[str]:
    keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    if keys:
        return keys
    out = []
    for col in episodes.columns:
        if col.startswith("videos/") and col.endswith("/from_timestamp"):
            out.append(col[len("videos/") : -len("/from_timestamp")])
    return sorted(set(out))


def _as_task_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    return [str(value)]


def _replace_tasks_value(value: Any, prompt: str) -> Any:
    if isinstance(value, np.ndarray):
        return np.array([prompt], dtype=object)
    if isinstance(value, list):
        return [prompt]
    if isinstance(value, tuple):
        return (prompt,)
    return np.array([prompt], dtype=object)


def _prompt_for_episode(row: dict[str, Any], data: pd.DataFrame, tasks: list[str]) -> str:
    prompts = _as_task_list(row.get("tasks"))
    if prompts:
        return prompts[0]
    if "task_index" in data.columns:
        unique = sorted(int(value) for value in data["task_index"].dropna().unique())
        if len(unique) == 1 and 0 <= unique[0] < len(tasks):
            return tasks[unique[0]]
    raise ValueError(f"Could not determine prompt for episode {row.get('episode_index')}")


def _to_matrix(values: pd.Series) -> np.ndarray | None:
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


def _format_like(value: np.ndarray, original: Any) -> Any:
    value = np.asarray(value)
    if isinstance(original, np.ndarray):
        return value.astype(original.dtype, copy=False)
    if isinstance(original, list):
        return value.tolist()
    return value


def _set_feature_stats(row: dict[str, Any], feature: str, values: pd.Series) -> None:
    prefix = f"stats/{feature}"

    matrix = _to_matrix(values)
    if matrix is None or matrix.size == 0:
        return

    stats = {
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
        row[key] = _format_like(value, row[key]) if key in row else np.asarray(value).tolist()


def _episode_stats_from_row(row: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        _, feature, stat_name = key.split("/", 2)
        stats.setdefault(feature, {})[stat_name] = np.asarray(value, dtype=np.float64)
    return stats


def _aggregate_feature_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([item["mean"] for item in items])
    variances = np.stack([item["std"] ** 2 for item in items])
    counts = np.stack([item["count"] for item in items])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    total_mean = (means * counts).sum(axis=0) / total_count
    weighted_variances = (variances + (means - total_mean) ** 2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    out = {
        "min": np.min(np.stack([item["min"] for item in items]), axis=0),
        "max": np.max(np.stack([item["max"] for item in items]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    quantile_keys = [k for k in items[0] if k.startswith("q") and k[1:].isdigit()]
    for key in quantile_keys:
        if all(key in item for item in items):
            values = np.stack([item[key] for item in items])
            out[key] = (values * counts).sum(axis=0) / total_count
    return out


def _stats_from_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    stats = {
        "min": np.min(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "count": np.array([matrix.shape[0]], dtype=np.int64),
    }
    for name, q in STAT_QUANTILES.items():
        stats[name] = np.quantile(matrix, q, axis=0)
    return stats


def _stats_from_data_parquets(root: Path) -> dict[str, dict[str, np.ndarray]]:
    data = _load_data(root)
    stats: dict[str, dict[str, np.ndarray]] = {}

    for column in data.columns:
        matrix = _to_matrix(data[column])
        if matrix is None or matrix.size == 0:
            continue
        stats[column] = _stats_from_matrix(matrix)

    return stats


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


def _write_stats(rows: list[dict[str, Any]], root: Path) -> None:
    per_episode = [_episode_stats_from_row(row) for row in rows]
    features = sorted({feature for stats in per_episode for feature in stats})
    aggregate: dict[str, dict[str, np.ndarray]] = {}
    for feature in features:
        items = [stats[feature] for stats in per_episode if feature in stats]
        if items and all("mean" in item and "std" in item and "count" in item for item in items):
            aggregate[feature] = _aggregate_feature_stats(items)

    if not aggregate:
        aggregate = _stats_from_data_parquets(root)

    stats_path = root / "meta" / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(_jsonify(aggregate), f, indent=2)


def _copy_auxiliary_files(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.name in {"data", "meta", "videos", ".git"}:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=False)
        else:
            shutil.copy2(item, target)


def _select_episodes(root: Path, name: str, episode_indices: list[int]) -> list[EpisodeRef]:
    info = _load_info(root)
    episodes = _load_episodes(root)
    data = _load_data(root)
    tasks = _load_tasks(root)

    available = set(int(value) for value in episodes["episode_index"].tolist())
    missing = sorted(set(episode_indices) - available)
    if missing:
        raise ValueError(f"{name}: requested episode(s) not present: {missing}")

    refs = []
    for ep_idx in episode_indices:
        row = episodes[episodes["episode_index"].astype(int) == int(ep_idx)].iloc[0].to_dict()
        ep_data = data[data["episode_index"].astype(int) == int(ep_idx)].copy()
        if ep_data.empty:
            raise ValueError(f"{name}: episode {ep_idx} has no frame rows")
        prompt = _prompt_for_episode(row, ep_data, tasks)
        row["tasks"] = _replace_tasks_value(row.get("tasks"), prompt)
        refs.append(EpisodeRef(root=root, source_name=name, old_episode_index=int(ep_idx), row=row, data=ep_data))

    return refs


def _build_dataset(ref_root: Path, episode_refs: list[EpisodeRef], dst: Path) -> dict[str, int]:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    _copy_auxiliary_files(ref_root, dst)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    ref_info = _load_info(ref_root)
    ref_episodes = _load_episodes(ref_root)
    ref_tasks = _load_tasks(ref_root)
    video_keys = _video_keys(ref_info, ref_episodes)

    prompts: list[str] = []
    prompt_to_index: dict[str, int] = {}
    video_map: dict[tuple[str, str, int, int], int] = {}
    out_video_counts = {key: 0 for key in video_keys}
    rows: list[dict[str, Any]] = []
    total_frames = 0

    for new_ep_idx, ref in enumerate(episode_refs):
        prompt = _prompt_for_episode(ref.row, ref.data, ref_tasks)
        if prompt not in prompt_to_index:
            prompt_to_index[prompt] = len(prompts)
            prompts.append(prompt)
        new_task_index = prompt_to_index[prompt]

        ep_data = ref.data.sort_values("index" if "index" in ref.data.columns else "frame_index").reset_index(drop=True)
        ep_len = len(ep_data)
        ep_data["episode_index"] = np.int64(new_ep_idx)
        ep_data["index"] = np.arange(total_frames, total_frames + ep_len, dtype=np.int64)
        if "frame_index" in ep_data.columns:
            ep_data["frame_index"] = np.arange(ep_len, dtype=np.int64)
        if "task_index" in ep_data.columns:
            ep_data["task_index"] = np.int64(new_task_index)

        data_path = dst / "data" / "chunk-000" / f"file-{new_ep_idx:03d}.parquet"
        ep_data.to_parquet(data_path, index=False)

        row = dict(ref.row)
        row["episode_index"] = int(new_ep_idx)
        row["length"] = int(ep_len)
        row["dataset_from_index"] = int(total_frames)
        row["dataset_to_index"] = int(total_frames + ep_len)
        row["data/chunk_index"] = 0
        row["data/file_index"] = int(new_ep_idx)
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        if "tasks" in row:
            row["tasks"] = _replace_tasks_value(row["tasks"], prompt)

        for stats_key in [key for key in row if key.startswith("stats/")]:
            del row[stats_key]

        for key in video_keys:
            src_chunk = int(row[f"videos/{key}/chunk_index"])
            src_file = int(row[f"videos/{key}/file_index"])
            map_key = (str(ref.root), key, src_chunk, src_file)
            if map_key not in video_map:
                new_file = out_video_counts[key]
                src_video = ref.root / "videos" / key / f"chunk-{src_chunk:03d}" / f"file-{src_file:03d}.mp4"
                dst_video = dst / "videos" / key / "chunk-000" / f"file-{new_file:03d}.mp4"
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)
                video_map[map_key] = new_file
                out_video_counts[key] += 1
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = video_map[map_key]

        for feature in [
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]:
            if feature in ep_data.columns:
                _set_feature_stats(row, feature, ep_data[feature])

        rows.append(row)
        total_frames += ep_len

    episodes_path = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pd.DataFrame(rows).to_parquet(episodes_path, index=False)

    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(prompts)))},
        index=pd.Index(prompts),
    )
    (dst / "meta").mkdir(exist_ok=True)
    tasks_df.to_parquet(dst / "meta" / "tasks.parquet")

    info = dict(ref_info)
    info["total_episodes"] = len(episode_refs)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = len(prompts)
    info["splits"] = {"train": f"0:{len(episode_refs)}"}
    if "total_chunks" in info:
        info["total_chunks"] = 1
    if "total_videos" in info:
        info["total_videos"] = int(sum(out_video_counts.values()))
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    _write_stats(rows, dst)

    return {
        "episodes": len(episode_refs),
        "frames": int(total_frames),
        "tasks": len(prompts),
        "videos": int(sum(out_video_counts.values())),
    }


def _validate(root: Path) -> None:
    info = _load_info(root)
    episodes = _load_episodes(root)
    data = _load_data(root)

    expected_eps = list(range(len(episodes)))
    actual_eps = [int(value) for value in episodes["episode_index"].tolist()]
    if actual_eps != expected_eps:
        raise ValueError(f"{root}: episode_index is not consecutive")
    if int(info["total_episodes"]) != len(episodes):
        raise ValueError(f"{root}: info total_episodes mismatch")
    if int(info["total_frames"]) != len(data):
        raise ValueError(f"{root}: info total_frames mismatch")
    if int(episodes["length"].sum()) != len(data):
        raise ValueError(f"{root}: episode lengths do not sum to data rows")
    if len(episodes) and int(data["index"].min()) != 0:
        raise ValueError(f"{root}: global index does not start at 0")
    if len(data) and [int(v) for v in data["index"].tolist()] != list(range(len(data))):
        raise ValueError(f"{root}: global index is not consecutive")

    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        ep_data = data[data["episode_index"].astype(int) == ep_idx]
        if len(ep_data) != int(row["length"]):
            raise ValueError(f"{root}: episode {ep_idx} length mismatch")
        if int(row["dataset_from_index"]) != int(ep_data["index"].min()):
            raise ValueError(f"{root}: episode {ep_idx} dataset_from_index mismatch")
        if int(row["dataset_to_index"]) != int(ep_data["index"].max()) + 1:
            raise ValueError(f"{root}: episode {ep_idx} dataset_to_index mismatch")
        if "frame_index" in ep_data.columns:
            frames = [int(v) for v in ep_data["frame_index"].tolist()]
            if frames != list(range(len(ep_data))):
                raise ValueError(f"{root}: episode {ep_idx} frame_index is not consecutive")
        if not _as_task_list(row.get("tasks")):
            raise ValueError(f"{root}: episode {ep_idx} has no task prompt")
    if not (root / "meta" / "stats.json").exists():
        raise ValueError(f"{root}: missing meta/stats.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--dest-repo-id", required=True)
    parser.add_argument("--episodes", required=True, nargs="+", type=int)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    base_tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        base_tmp = tempfile.TemporaryDirectory(prefix="lerobot_move_episodes_")
        work = Path(base_tmp.name)

    source_root = work / "source_latest"
    dest_root = work / "dest_latest"
    source_out = work / "source_updated"
    dest_out = work / "dest_updated"

    try:
        _download_hf_dataset(args.source_repo_id, source_root)
        _download_hf_dataset(args.dest_repo_id, dest_root)

        source_episodes = _load_episodes(source_root)
        source_indices = [int(value) for value in source_episodes["episode_index"].tolist()]
        move_set = set(args.episodes)
        keep_source_indices = [idx for idx in source_indices if idx not in move_set]

        moved_refs = _select_episodes(source_root, args.source_repo_id, args.episodes)
        source_keep_refs = _select_episodes(source_root, args.source_repo_id, keep_source_indices)

        dest_episodes = _load_episodes(dest_root)
        dest_indices = [int(value) for value in dest_episodes["episode_index"].tolist()]
        dest_refs = _select_episodes(dest_root, args.dest_repo_id, dest_indices)

        dest_stats = _build_dataset(dest_root, dest_refs + moved_refs, dest_out)
        source_stats = _build_dataset(source_root, source_keep_refs, source_out)

        _validate(dest_out)
        _validate(source_out)

        print("\nMove summary")
        print(f"  moved episodes from source: {args.episodes}")
        print(f"  source after move: {source_stats}")
        print(f"  destination after move: {dest_stats}")
        for ref in moved_refs:
            prompt = _prompt_for_episode(ref.row, ref.data, _load_tasks(ref.root))
            print(
                f"  moved source episode {ref.old_episode_index}: "
                f"{len(ref.data)} frames | {prompt}"
            )

        if args.push_to_hub:
            _upload_folder(
                dest_out,
                args.dest_repo_id,
                f"Append episodes {args.episodes} moved from {args.source_repo_id}",
            )
            _upload_folder(
                source_out,
                args.source_repo_id,
                f"Remove episodes {args.episodes} moved to {args.dest_repo_id}",
            )
    finally:
        if base_tmp is not None:
            base_tmp.cleanup()


if __name__ == "__main__":
    main()
