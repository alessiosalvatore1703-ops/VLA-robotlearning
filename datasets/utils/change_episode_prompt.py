#!/usr/bin/env python3
"""Change the task prompt for one episode in a LeRobot v3 dataset.

This updates both places that matter for LeRobot training/visualization:

- data parquet rows: the selected episode's ``task_index``
- meta/episodes parquet rows: the selected episode's ``tasks`` field and
  per-episode ``stats/task_index/*`` values when present

The global ``meta/tasks.parquet`` is preserved unless the new prompt is not
already present, in which case a new task row is appended.
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


TASK_INDEX_STATS = (
    "min",
    "max",
    "mean",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
)


def _is_string_like(values: Any) -> bool:
    if pd.api.types.is_string_dtype(values):
        return True
    try:
        return all(isinstance(value, str) for value in list(values))
    except TypeError:
        return False


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


def _upload_files(local_root: Path, repo_id: str, relpaths: list[Path]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    for relpath in relpaths:
        print(f"Uploading {relpath.as_posix()} ...")
        api.upload_file(
            path_or_fileobj=str(local_root / relpath),
            path_in_repo=relpath.as_posix(),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Fix prompt for episode metadata in {repo_id}",
        )


def _extract_tasks(tasks_df: pd.DataFrame) -> tuple[str, list[int], list[str]]:
    """Return (storage, task_indices, task_strings).

    storage is one of:
    - "index": task strings are stored as the dataframe index
    - "task_column": task strings are stored in a column named "task"
    - "string_column:<name>": task strings are stored in another string column
    """

    if _is_string_like(tasks_df.index):
        tasks = [str(value) for value in tasks_df.index]
        storage = "index"
    elif "task" in tasks_df.columns and _is_string_like(tasks_df["task"]):
        tasks = [str(value) for value in tasks_df["task"]]
        storage = "task_column"
    else:
        reset_df = tasks_df.reset_index()
        task_col = None
        for col in reset_df.columns:
            if _is_string_like(reset_df[col]):
                task_col = str(col)
                break
        if task_col is None:
            raise ValueError("Could not find task strings in meta/tasks.parquet")
        tasks = [str(value) for value in reset_df[task_col]]
        storage = f"string_column:{task_col}"

    if "task_index" in tasks_df.columns:
        task_indices = [int(value) for value in tasks_df["task_index"]]
    else:
        task_indices = list(range(len(tasks)))

    if len(task_indices) != len(tasks):
        raise ValueError("Task index count does not match task string count")
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("Duplicate task_index values in meta/tasks.parquet")

    return storage, task_indices, tasks


def _append_task(
    tasks_df: pd.DataFrame,
    storage: str,
    new_task_index: int,
    new_prompt: str,
) -> pd.DataFrame:
    out = tasks_df.copy()
    if storage == "index":
        row = pd.DataFrame(
            {"task_index": [new_task_index]},
            index=pd.Index([new_prompt], name=tasks_df.index.name),
        )
        return pd.concat([out, row])

    if storage == "task_column":
        row_data = {col: [pd.NA] for col in out.columns}
        row_data["task_index"] = [new_task_index]
        row_data["task"] = [new_prompt]
        return pd.concat([out, pd.DataFrame(row_data)], ignore_index=True)

    raise ValueError(
        f"Unsupported tasks.parquet layout for appending a new task ({storage}). "
        "Add the prompt manually or use a dataset with index/task-column tasks."
    )


def _as_task_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    return [str(value)]


def _replace_tasks_value(value: Any, new_prompt: str) -> Any:
    if isinstance(value, np.ndarray):
        return np.array([new_prompt], dtype=object)
    if isinstance(value, list):
        return [new_prompt]
    if isinstance(value, tuple):
        return (new_prompt,)
    return new_prompt


def _set_episode_task_index_stats(row: dict[str, Any], task_index: int) -> dict[str, Any]:
    for suffix in TASK_INDEX_STATS:
        key = f"stats/task_index/{suffix}"
        if key in row:
            row[key] = np.array([float(task_index)])
    std_key = "stats/task_index/std"
    if std_key in row:
        row[std_key] = np.array([0.0])
    return row


def _task_to_index_first(task_strings: list[str], task_indices: list[int]) -> dict[str, int]:
    task_to_index: dict[str, int] = {}
    for task, task_index in zip(task_strings, task_indices):
        task_to_index.setdefault(task, task_index)
    return task_to_index


def _unique_data_task_indices(root: Path) -> set[int]:
    used: set[int] = set()
    for data_file in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(data_file)
        if "task_index" in df.columns:
            used.update(int(value) for value in df["task_index"].dropna().unique())
    return used


def _compact_tasks(root: Path) -> list[Path]:
    """Drop unused/duplicate task rows and remap task_index to 0..N-1."""

    meta_dir = root / "meta"
    tasks_path = meta_dir / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_path)
    _, task_indices, task_strings = _extract_tasks(tasks_df)
    used_indices = _unique_data_task_indices(root)

    if not used_indices:
        return []

    index_to_task = {idx: task for idx, task in zip(task_indices, task_strings)}
    unknown = sorted(used_indices - set(index_to_task))
    if unknown:
        raise ValueError(f"Data uses task_index values that are missing from tasks.parquet: {unknown}")

    old_to_new: dict[int, int] = {}
    new_tasks: list[str] = []
    new_task_to_index: dict[str, int] = {}

    for old_index, task in zip(task_indices, task_strings):
        if old_index not in used_indices:
            continue
        if task not in new_task_to_index:
            new_task_to_index[task] = len(new_tasks)
            new_tasks.append(task)
        old_to_new[old_index] = new_task_to_index[task]

    current_used_rows = [
        (old_index, task)
        for old_index, task in zip(task_indices, task_strings)
        if old_index in used_indices
    ]
    already_compact = (
        len(current_used_rows) == len(task_indices)
        and len(new_tasks) == len(task_indices)
        and all(old_index == new_index for old_index, new_index in old_to_new.items())
    )
    if already_compact:
        return []

    changed: set[Path] = set()

    for data_file in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(data_file)
        if "task_index" not in df.columns:
            continue
        values = set(int(value) for value in df["task_index"].dropna().unique())
        if not values.intersection(old_to_new):
            continue
        df["task_index"] = df["task_index"].map(lambda value: old_to_new[int(value)]).astype(np.int64)
        df.to_parquet(data_file, index=False)
        changed.add(data_file.relative_to(root))

    for episode_file in sorted((meta_dir / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(episode_file)
        rows = []
        touched = False
        for row in df.to_dict("records"):
            task_values = _as_task_list(row.get("tasks")) if "tasks" in row else []
            unique_tasks = set(task_values)
            if len(unique_tasks) == 1:
                task = next(iter(unique_tasks))
                if task in new_task_to_index:
                    row = _set_episode_task_index_stats(row, new_task_to_index[task])
                    touched = True
            rows.append(row)
        if touched:
            pd.DataFrame(rows).to_parquet(episode_file, index=False)
            changed.add(episode_file.relative_to(root))

    compact_df = pd.DataFrame(
        {"task_index": list(range(len(new_tasks)))},
        index=pd.Index(new_tasks, name=tasks_df.index.name),
    )
    compact_df.to_parquet(tasks_path)
    changed.add(tasks_path.relative_to(root))

    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info["total_tasks"] = len(new_tasks)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        changed.add(info_path.relative_to(root))

    print(f"Compacted tasks to {len(new_tasks)} used prompt(s):")
    for task_index, task in enumerate(new_tasks):
        print(f"  {task_index}: {task!r}")

    return sorted(changed)


def change_episode_prompt(
    root: Path,
    episode_index: int,
    old_prompt: str | None,
    new_prompt: str,
) -> list[Path]:
    meta_dir = root / "meta"
    tasks_path = meta_dir / "tasks.parquet"
    if not tasks_path.exists():
        raise ValueError(f"Missing LeRobot v3 task metadata: {tasks_path}")

    changed: set[Path] = set()

    tasks_df = pd.read_parquet(tasks_path)
    storage, task_indices, task_strings = _extract_tasks(tasks_df)
    task_to_index = _task_to_index_first(task_strings, task_indices)

    if old_prompt is not None and old_prompt not in task_to_index:
        print(
            "Warning: old prompt is not present in tasks.parquet; "
            "will validate it from meta/episodes instead."
        )

    if new_prompt not in task_to_index:
        new_task_index = max(task_indices, default=-1) + 1
        tasks_df = _append_task(tasks_df, storage, new_task_index, new_prompt)
        tasks_df.to_parquet(tasks_path)
        changed.add(tasks_path.relative_to(root))
        task_to_index[new_prompt] = new_task_index
        print(f"Added new task prompt with task_index={new_task_index}: {new_prompt!r}")
    else:
        new_task_index = task_to_index[new_prompt]

    old_task_index = task_to_index.get(old_prompt) if old_prompt is not None else None

    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise ValueError(f"No data parquet files found under {root / 'data'}")

    episode_frame_count = 0
    old_task_indices_seen: set[int] = set()
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        if "episode_index" not in df.columns:
            continue
        mask = df["episode_index"].astype(int) == int(episode_index)
        if not mask.any():
            continue
        if "task_index" not in df.columns:
            raise ValueError(f"{data_file} contains episode_index but no task_index column")

        episode_frame_count += int(mask.sum())
        old_task_indices_seen.update(int(value) for value in df.loc[mask, "task_index"].dropna().unique())
        df.loc[mask, "task_index"] = np.int64(new_task_index)
        df.to_parquet(data_file, index=False)
        changed.add(data_file.relative_to(root))

    if episode_frame_count == 0:
        raise ValueError(f"Could not find episode_index={episode_index} in data parquets")

    if old_task_index is not None and old_task_indices_seen and old_task_indices_seen != {old_task_index}:
        raise ValueError(
            "The selected episode data did not exclusively use the expected old prompt "
            f"task_index={old_task_index}; saw {sorted(old_task_indices_seen)}"
        )

    episode_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise ValueError(f"No episode metadata parquet files found under {meta_dir / 'episodes'}")

    episode_rows = 0
    previous_episode_prompts: list[str] = []
    for episode_file in episode_files:
        df = pd.read_parquet(episode_file)
        if "episode_index" not in df.columns:
            continue
        mask = df["episode_index"].astype(int) == int(episode_index)
        if not mask.any():
            continue

        rows = []
        for row in df.to_dict("records"):
            if int(row.get("episode_index", -1)) == int(episode_index):
                episode_rows += 1
                previous_episode_prompts.extend(_as_task_list(row.get("tasks")))
                if "tasks" in row:
                    row["tasks"] = _replace_tasks_value(row["tasks"], new_prompt)
                row = _set_episode_task_index_stats(row, new_task_index)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(episode_file, index=False)
        changed.add(episode_file.relative_to(root))

    if episode_rows == 0:
        raise ValueError(f"Could not find episode_index={episode_index} in episode metadata")

    if old_prompt is not None and previous_episode_prompts and old_prompt not in previous_episode_prompts:
        raise ValueError(
            "The episode metadata did not contain the expected old prompt. "
            f"Saw: {previous_episode_prompts}"
        )

    info_path = meta_dir / "info.json"
    if info_path.exists() and tasks_path.relative_to(root) in changed:
        with open(info_path) as f:
            info = json.load(f)
        info["total_tasks"] = len(task_to_index)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        changed.add(info_path.relative_to(root))

    print(f"Episode {episode_index}: updated {episode_frame_count} frame rows")
    print(f"Previous episode prompt(s): {previous_episode_prompts}")
    print(f"New prompt: {new_prompt!r} (task_index={new_task_index})")

    changed.update(_compact_tasks(root))

    return sorted(changed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id or local dataset path.")
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--old-prompt", default=None, help="Expected current prompt for validation.")
    parser.add_argument("--new-prompt", required=True)
    parser.add_argument("--push-to-hub", action="store_true", help="Upload changed files back to --repo-id.")
    parser.add_argument("--output-dir", default=None, help="Optional local copy destination.")
    args = parser.parse_args()

    input_is_hf = _is_hf_repo_id(args.repo_id)
    if args.push_to_hub and not input_is_hf:
        parser.error("--push-to-hub requires --repo-id to be a Hugging Face repo id")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if input_is_hf:
        temp_dir = tempfile.TemporaryDirectory(prefix="lerobot_episode_prompt_")
        root = Path(temp_dir.name) / "dataset"
        _download_hf_dataset(args.repo_id, root)
    else:
        root = Path(args.repo_id)
        if not root.is_dir():
            raise SystemExit(f"Dataset path does not exist: {root}")

    try:
        if args.output_dir:
            output_root = Path(args.output_dir)
            if output_root.exists():
                raise SystemExit(f"Output directory already exists: {output_root}")
            shutil.copytree(root, output_root)
            root = output_root

        changed = change_episode_prompt(
            root=root,
            episode_index=args.episode_index,
            old_prompt=args.old_prompt,
            new_prompt=args.new_prompt,
        )
        print("Changed files:")
        for relpath in changed:
            print(f"  {relpath.as_posix()}")

        if args.push_to_hub:
            _upload_files(root, args.repo_id, changed)
            print(f"Pushed update to https://huggingface.co/datasets/{args.repo_id}")
        elif not args.output_dir and input_is_hf:
            print("No --push-to-hub or --output-dir was passed, so changes were only made in a temporary copy.")
    finally:
        if temp_dir is not None and not args.output_dir:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
