#!/usr/bin/env python3
"""Create relative bowl prompts from ordinal bowl datasets, then merge.

The dataset config name defines the bowl colors from left to right:

    ETHrobotlearning/config1-red-blue-green

means:
    1st bowl -> red
    2nd bowl -> blue
    3rd bowl -> green

For an ordinal prompt targeting the middle bowl, the episode is duplicated with
both valid adjacent references:

    Put the banana into the bowl on the right of the red bowl from the robot perspective
    Put the banana into the bowl on the left of the green bowl from the robot perspective

For edge bowls, only the one valid adjacent reference is generated.

Usage:
    python datasets/utils/relabel_relative_bowls_and_merge.py

    python datasets/utils/relabel_relative_bowls_and_merge.py \\
        --output ETHrobotlearning/task2-relative
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from merge_datasets import (
    _download_hf_dataset,
    _hf_whoami,
    _is_bare_name,
    _is_hf_repo_id,
    _load_episodes,
    _push_to_hub,
    _require_hf,
    merge,
)
from relabel_bowls_and_merge import (
    DEFAULT_INPUTS,
    ORDINAL_RE,
    ORDINAL_TO_POSITION,
    _dedupe_preserve_order,
    _parse_color_order,
    _read_tasks,
    _replace_task_values,
    _set_task_index_stats,
    _source_name,
)


DEFAULT_OUTPUT = "ETHrobotlearning/task2-relative"

PROMPT_RE = re.compile(
    r"^\s*Put\s+the\s+(?P<object>.+?)\s+"
    r"(?:into|in|to|inside)\s+(?:the\s+)?"
    rf"(?P<ordinal>{ORDINAL_RE})\s+bowl"
    r"(?:\s+from\s+(?:the\s+)?left)?"
    r"(?:\s+from\s+the\s+robot\s+perspective)?\.?\s*$",
    re.IGNORECASE,
)

STAT_SUFFIXES = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")


def _parse_ordinal_prompt(task: str) -> Tuple[str, int]:
    match = PROMPT_RE.match(task)
    if not match:
        raise ValueError(f"Could not parse ordinal bowl prompt: {task!r}")
    obj = match.group("object").strip()
    ordinal = match.group("ordinal").lower()
    return obj, ORDINAL_TO_POSITION[ordinal]


def _relative_prompts(task: str, colors: Tuple[str, str, str]) -> List[str]:
    obj, target_pos = _parse_ordinal_prompt(task)

    if target_pos == 0:
        return [
            f"Put the {obj} into the bowl on the left of the {colors[1]} bowl "
            "from the robot perspective"
        ]
    if target_pos == 2:
        return [
            f"Put the {obj} into the bowl on the right of the {colors[1]} bowl "
            "from the robot perspective"
        ]

    return [
        f"Put the {obj} into the bowl on the right of the {colors[0]} bowl "
        "from the robot perspective",
        f"Put the {obj} into the bowl on the left of the {colors[2]} bowl "
        "from the robot perspective",
    ]


def _stats_for_sequence(values: np.ndarray) -> Dict[str, np.ndarray]:
    values = values.astype(float)
    return {
        "min": np.array([float(values.min())]),
        "max": np.array([float(values.max())]),
        "mean": np.array([float(values.mean())]),
        "std": np.array([float(values.std())]),
        "q01": np.array([float(np.quantile(values, 0.01))]),
        "q10": np.array([float(np.quantile(values, 0.10))]),
        "q50": np.array([float(np.quantile(values, 0.50))]),
        "q90": np.array([float(np.quantile(values, 0.90))]),
        "q99": np.array([float(np.quantile(values, 0.99))]),
    }


def _set_scalar_stats(row: Dict, feature: str, values: Iterable[int]) -> None:
    arr = np.array(list(values), dtype=float)
    stats = _stats_for_sequence(arr)
    for suffix in STAT_SUFFIXES:
        key = f"stats/{feature}/{suffix}"
        if key in row:
            row[key] = stats[suffix]
    count_key = f"stats/{feature}/count"
    if count_key in row:
        row[count_key] = np.array([len(arr)])
    std_key = f"stats/{feature}/std"
    if std_key in row:
        row[std_key] = stats["std"]


def _episode_task_index(ep_df: pd.DataFrame, task_indices: List[int]) -> int:
    if "task_index" not in ep_df.columns:
        if len(task_indices) == 1:
            return task_indices[0]
        raise ValueError("Episode data has no task_index column and dataset has multiple tasks")

    indices = sorted(int(value) for value in ep_df["task_index"].dropna().unique())
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one task_index per episode, found {indices}")
    return indices[0]


def _episode_data(src: Path, episode_row: pd.Series) -> pd.DataFrame:
    chunk_idx = int(episode_row["data/chunk_index"])
    file_idx = int(episode_row["data/file_index"])
    data_path = src / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
    df = pd.read_parquet(data_path)
    ep_idx = int(episode_row["episode_index"])
    ep_df = df[df["episode_index"].astype(int) == ep_idx].copy().reset_index(drop=True)
    if ep_df.empty:
        raise ValueError(f"No rows for episode_index={ep_idx} in {data_path}")
    return ep_df


def expand_dataset(
    src: Path,
    dst: Path,
    source_label: str,
    allow_unchanged: bool = False,
) -> None:
    """Copy a source dataset and duplicate/relabel episodes with relative prompts."""
    meta_dir = src / "meta"
    tasks_path = meta_dir / "tasks.parquet"
    if not tasks_path.exists():
        raise ValueError(f"{src} is not a LeRobot v3 dataset: missing meta/tasks.parquet")
    if dst.exists():
        raise ValueError(f"Destination already exists: {dst}")

    colors = _parse_color_order(source_label)
    task_df, old_task_indices, old_tasks = _read_tasks(tasks_path)
    old_task_by_index = {
        task_index: task for task_index, task in zip(old_task_indices, old_tasks)
    }

    old_to_new_tasks: Dict[str, List[str]] = {}
    unchanged: List[str] = []
    for old_task in old_tasks:
        try:
            new_tasks = _relative_prompts(old_task, colors)
        except ValueError:
            if not allow_unchanged:
                raise
            new_tasks = [old_task]
        old_to_new_tasks[old_task] = new_tasks
        if new_tasks == [old_task]:
            unchanged.append(old_task)

    if unchanged and not allow_unchanged:
        examples = "\n".join(f"  {task!r}" for task in unchanged[:5])
        raise ValueError(
            f"{source_label}: {len(unchanged)} task prompt(s) did not match the ordinal "
            f"bowl pattern. Pass --allow-unchanged to keep them.\n{examples}"
        )

    new_tasks = _dedupe_preserve_order(
        new_task for old_task in old_tasks for new_task in old_to_new_tasks[old_task]
    )
    new_task_to_index = {task: idx for idx, task in enumerate(new_tasks)}

    print(f"Relabeling {_source_name(source_label)}:")
    print(f"  color order: 1st={colors[0]}, 2nd={colors[1]}, 3rd={colors[2]}")
    for old_task in old_tasks:
        for new_task in old_to_new_tasks[old_task]:
            print(f"  {old_task!r} -> {new_task!r}")

    shutil.copytree(src, dst)
    shutil.rmtree(dst / "data")
    shutil.rmtree(dst / "meta" / "episodes")
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    index_name = task_df.index.name if pd.api.types.is_string_dtype(task_df.index) else None
    out_tasks = pd.DataFrame(
        {"task_index": list(range(len(new_tasks)))},
        index=pd.Index(new_tasks, name=index_name),
    )
    out_tasks.to_parquet(dst / "meta" / "tasks.parquet")

    episodes = _load_episodes(src)
    new_episode_rows: List[Dict] = []
    new_episode_index = 0
    global_frame_index = 0
    data_file_index = 0

    for _, episode_row in tqdm(
        episodes.iterrows(),
        total=len(episodes),
        desc="  expanding episodes",
        unit="episode",
    ):
        ep_df = _episode_data(src, episode_row)
        old_task_index = _episode_task_index(ep_df, old_task_indices)
        if old_task_index not in old_task_by_index:
            raise ValueError(
                f"Episode {int(episode_row['episode_index'])} references unknown "
                f"task_index={old_task_index}"
            )
        old_task = old_task_by_index[old_task_index]

        for new_task in old_to_new_tasks[old_task]:
            new_task_index = new_task_to_index[new_task]
            new_df = ep_df.copy()
            n_frames = len(new_df)

            new_df["episode_index"] = np.int64(new_episode_index)
            if "frame_index" in new_df.columns:
                new_df["frame_index"] = np.arange(n_frames, dtype=np.int64)
            new_df["index"] = np.arange(
                global_frame_index,
                global_frame_index + n_frames,
                dtype=np.int64,
            )
            if "task_index" in new_df.columns:
                new_df["task_index"] = np.int64(new_task_index)

            out_data = dst / "data" / "chunk-000" / f"file-{data_file_index:03d}.parquet"
            new_df.to_parquet(out_data, index=False)

            row = episode_row.to_dict()
            row["episode_index"] = new_episode_index
            row["dataset_from_index"] = global_frame_index
            row["dataset_to_index"] = global_frame_index + n_frames
            row["data/chunk_index"] = 0
            row["data/file_index"] = data_file_index
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            if "tasks" in row:
                row["tasks"] = _replace_task_values(row["tasks"], {old_task: new_task})

            _set_scalar_stats(row, "episode_index", [new_episode_index] * n_frames)
            _set_scalar_stats(row, "index", range(global_frame_index, global_frame_index + n_frames))
            if "frame_index" in new_df.columns:
                _set_scalar_stats(row, "frame_index", range(n_frames))
            _set_task_index_stats(row, new_task_index)

            new_episode_rows.append(row)
            new_episode_index += 1
            global_frame_index += n_frames
            data_file_index += 1

    pd.DataFrame(new_episode_rows).to_parquet(
        dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        index=False,
    )

    info_path = dst / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = new_episode_index
    info["total_frames"] = global_frame_index
    info["total_tasks"] = len(new_tasks)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{new_episode_index}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(
        f"  expanded to {new_episode_index} episodes and "
        f"{global_frame_index} frames"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create relative bowl prompts from ordinal bowl datasets, then merge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        metavar="SRC",
        help="Source datasets: local paths or HF repo IDs. Defaults to the six ETHrobotlearning configs.",
    )
    ap.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="DST",
        help=f"Output: local path, bare dataset name, or HF repo ID. Default: {DEFAULT_OUTPUT}",
    )
    ap.add_argument(
        "--allow-unchanged",
        action="store_true",
        help="Keep task prompts that do not match the ordinal bowl pattern instead of failing.",
    )
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("Provide at least two --inputs datasets.")

    any_hf_input = any(_is_hf_repo_id(value) for value in args.inputs)
    output_is_hf = _is_bare_name(args.output) or _is_hf_repo_id(args.output)

    if any_hf_input or output_is_hf:
        _require_hf()

    hf_output_repo: Optional[str] = None
    if output_is_hf:
        hf_output_repo = (
            args.output if _is_hf_repo_id(args.output) else f"{_hf_whoami()}/{args.output}"
        )

    tmp_dirs: List[tempfile.TemporaryDirectory] = []
    expanded_dirs: List[Path] = []

    try:
        print("Loading and expanding source datasets ...")
        for raw_source in args.inputs:
            if _is_hf_repo_id(raw_source):
                src_tmp = tempfile.TemporaryDirectory(prefix="lerobot_relative_src_")
                tmp_dirs.append(src_tmp)
                src = Path(src_tmp.name)
                _download_hf_dataset(raw_source, src)
            else:
                src = Path(raw_source)
                if not src.is_dir():
                    sys.exit(f"Error: source path does not exist: {src}")

            expanded_tmp = tempfile.TemporaryDirectory(prefix="lerobot_relative_expand_")
            tmp_dirs.append(expanded_tmp)
            expanded = Path(expanded_tmp.name) / _source_name(raw_source)
            expand_dataset(
                src,
                expanded,
                source_label=raw_source,
                allow_unchanged=args.allow_unchanged,
            )
            expanded_dirs.append(expanded)

        if output_is_hf:
            dst_tmp = tempfile.TemporaryDirectory(prefix="lerobot_relative_dst_")
            tmp_dirs.append(dst_tmp)
            dst = Path(dst_tmp.name) / "dataset"
        else:
            dst = Path(args.output)
            if dst.exists():
                sys.exit(f"Error: destination already exists: {dst}")

        merge(expanded_dirs, dst)

        if output_is_hf:
            _push_to_hub(dst, hf_output_repo)
        else:
            print(f"Done  ->  {dst}")
    finally:
        for tmp_dir in tmp_dirs:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main()
