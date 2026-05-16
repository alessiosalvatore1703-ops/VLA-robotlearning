#!/usr/bin/env python3
"""Relabel ordinal bowl prompts to color prompts, then merge LeRobot v3 datasets.

The dataset config name defines the bowl colors from left to right:

    ETHrobotlearning/config1-red-blue-green

means:
    1st bowl -> red
    2nd bowl -> blue
    3rd bowl -> green

Prompts such as:

    Put the banana into the 2nd bowl from the left from the robot perspective

are rewritten to:

    Put the banana in the blue colored bowl

Usage:
    python datasets/utils/relabel_bowls_and_merge.py \\
        --output ETHrobotlearning/banana-bowls-color-prompts

    # Local output instead of pushing to Hub. Use ./ so it is treated as a path.
    python datasets/utils/relabel_bowls_and_merge.py \\
        --output ./banana-bowls-color-prompts
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
    _push_to_hub,
    _require_hf,
    merge,
)


DEFAULT_INPUTS = [
    "ETHrobotlearning/config1-red-blue-green",
    "ETHrobotlearning/config2-green-red-blue",
    "ETHrobotlearning/config3-green-blue-red",
    "ETHrobotlearning/config4-red-green-blue",
    "ETHrobotlearning/config5-blue-green-red",
    "ETHrobotlearning/config6-blue-red-green",
]

ORDINAL_TO_POSITION = {
    "1st": 0,
    "1nd": 0,
    "1rd": 0,
    "1th": 0,
    "first": 0,
    "2nd": 1,
    "2st": 1,
    "2rd": 1,
    "2th": 1,
    "second": 1,
    "3rd": 2,
    "3st": 2,
    "3nd": 2,
    "3th": 2,
    "third": 2,
}

ORDINAL_RE = "|".join(
    re.escape(value) for value in sorted(ORDINAL_TO_POSITION, key=len, reverse=True)
)

TARGET_RE = re.compile(
    r"\b(?:into|in|to|inside)\s+(?:the\s+)?"
    rf"(?P<ordinal>{ORDINAL_RE})\s+bowl"
    r"(?:\s+from\s+(?:the\s+)?left)?"
    r"(?:\s+from\s+the\s+robot\s+perspective)?",
    re.IGNORECASE,
)

CONFIG_RE = re.compile(
    r"^config\d+-(?P<first>[a-z]+)-(?P<second>[a-z]+)-(?P<third>[a-z]+)$",
    re.IGNORECASE,
)

STAT_SUFFIXES = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")


def _source_name(source: str) -> str:
    return Path(source.rstrip("/")).name


def _parse_color_order(source: str) -> Tuple[str, str, str]:
    name = _source_name(source)
    match = CONFIG_RE.match(name)
    if not match:
        raise ValueError(
            f"Could not infer bowl colors from {source!r}. "
            "Expected a name like config1-red-blue-green."
        )
    return (
        match.group("first").lower(),
        match.group("second").lower(),
        match.group("third").lower(),
    )


def _rewrite_prompt(task: str, colors: Tuple[str, str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        ordinal = match.group("ordinal").lower()
        color = colors[ORDINAL_TO_POSITION[ordinal]]
        return f"in the {color} colored bowl"

    return TARGET_RE.sub(replace, task)


def _read_tasks(tasks_path: Path) -> Tuple[pd.DataFrame, List[int], List[str]]:
    df = pd.read_parquet(tasks_path)

    if pd.api.types.is_string_dtype(df.index):
        tasks = [str(value) for value in df.index]
    elif "task" in df.columns and pd.api.types.is_string_dtype(df["task"]):
        tasks = [str(value) for value in df["task"]]
    else:
        df2 = df.reset_index()
        task_col = None
        for col in df2.columns:
            if pd.api.types.is_string_dtype(df2[col]):
                task_col = col
                break
        if task_col is None:
            raise ValueError(f"Could not find task strings in {tasks_path}")
        tasks = [str(value) for value in df2[task_col]]

    if "task_index" in df.columns:
        task_indices = [int(value) for value in df["task_index"]]
    else:
        task_indices = list(range(len(tasks)))

    if len(task_indices) != len(tasks):
        raise ValueError(f"Task index count does not match task count in {tasks_path}")
    if len(set(task_indices)) != len(task_indices):
        raise ValueError(f"Duplicate task_index values in {tasks_path}")

    return df, task_indices, tasks


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _set_task_index_stats(row: Dict, task_index: int) -> None:
    for suffix in STAT_SUFFIXES:
        key = f"stats/task_index/{suffix}"
        if key in row:
            row[key] = np.array([float(task_index)])
    key = "stats/task_index/std"
    if key in row:
        row[key] = np.array([0.0])


def _replace_task_values(value: object, task_map: Dict[str, str]) -> object:
    if isinstance(value, np.ndarray):
        return np.array([task_map.get(str(item), str(item)) for item in value], dtype=object)
    if isinstance(value, list):
        return [task_map.get(str(item), str(item)) for item in value]
    if isinstance(value, tuple):
        return tuple(task_map.get(str(item), str(item)) for item in value)
    if isinstance(value, str):
        return task_map.get(value, value)
    return value


def relabel_dataset(
    src: Path,
    dst: Path,
    source_label: str,
    allow_unchanged: bool = False,
) -> None:
    """Copy one dataset and rewrite ordinal bowl tasks to color bowl tasks."""
    meta_dir = src / "meta"
    tasks_path = meta_dir / "tasks.parquet"
    if not tasks_path.exists():
        raise ValueError(f"{src} is not a LeRobot v3 dataset: missing meta/tasks.parquet")
    if dst.exists():
        raise ValueError(f"Destination already exists: {dst}")

    colors = _parse_color_order(source_label)

    task_df, old_task_indices, old_tasks = _read_tasks(tasks_path)
    if not old_tasks:
        raise ValueError(f"No task prompts found in {tasks_path}")

    old_to_new_task: Dict[str, str] = {}
    unchanged = []
    for task in old_tasks:
        new_task = _rewrite_prompt(task, colors)
        old_to_new_task[task] = new_task
        if new_task == task:
            unchanged.append(task)

    if unchanged and not allow_unchanged:
        examples = "\n".join(f"  {task!r}" for task in unchanged[:5])
        raise ValueError(
            f"{source_label}: {len(unchanged)} task prompt(s) did not match the ordinal "
            f"bowl pattern. Pass --allow-unchanged to keep them.\n{examples}"
        )

    new_tasks = _dedupe_preserve_order(old_to_new_task[task] for task in old_tasks)
    new_task_to_index = {task: idx for idx, task in enumerate(new_tasks)}
    old_task_index_to_new = {
        old_idx: new_task_to_index[old_to_new_task[old_task]]
        for old_idx, old_task in zip(old_task_indices, old_tasks)
    }

    print(f"Relabeling {_source_name(source_label)}:")
    print(f"  color order: 1st={colors[0]}, 2nd={colors[1]}, 3rd={colors[2]}")
    for old_task in old_tasks:
        print(f"  {old_task!r} -> {old_to_new_task[old_task]!r}")

    shutil.copytree(src, dst)
    dst_meta = dst / "meta"

    index_name = task_df.index.name if pd.api.types.is_string_dtype(task_df.index) else None
    out_tasks = pd.DataFrame(
        {"task_index": list(range(len(new_tasks)))},
        index=pd.Index(new_tasks, name=index_name),
    )
    out_tasks.to_parquet(dst_meta / "tasks.parquet")

    data_files = sorted((dst / "data").rglob("*.parquet"))
    if not data_files:
        raise ValueError(f"No data parquet files found in {dst / 'data'}")

    for data_file in tqdm(data_files, desc="  updating data", unit="file"):
        df = pd.read_parquet(data_file)
        if "task_index" in df.columns:
            unknown = sorted(
                set(int(value) for value in df["task_index"].dropna().unique())
                - set(old_task_index_to_new)
            )
            if unknown:
                raise ValueError(f"{data_file} contains unknown task_index values: {unknown}")
            df["task_index"] = df["task_index"].map(old_task_index_to_new).astype(np.int64)
            df.to_parquet(data_file, index=False)

    episode_files = sorted((dst_meta / "episodes").rglob("*.parquet"))
    for episode_file in tqdm(episode_files, desc="  updating episodes", unit="file"):
        df = pd.read_parquet(episode_file)
        if "tasks" in df.columns:
            rows = []
            for row in df.to_dict("records"):
                row["tasks"] = _replace_task_values(row["tasks"], old_to_new_task)
                task_values = row["tasks"]
                if isinstance(task_values, np.ndarray):
                    unique_tasks = set(str(item) for item in task_values)
                elif isinstance(task_values, (list, tuple)):
                    unique_tasks = set(str(item) for item in task_values)
                elif isinstance(task_values, str):
                    unique_tasks = {task_values}
                else:
                    unique_tasks = set()
                if len(unique_tasks) == 1:
                    task = next(iter(unique_tasks))
                    if task in new_task_to_index:
                        _set_task_index_stats(row, new_task_to_index[task])
                rows.append(row)
            pd.DataFrame(rows).to_parquet(episode_file, index=False)

    info_path = dst_meta / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_tasks"] = len(new_tasks)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Relabel ordinal bowl prompts to color prompts, then merge datasets.",
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
        required=True,
        metavar="DST",
        help="Output: local path, bare dataset name, or HF repo ID. Bare names are pushed to your HF account.",
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
    relabeled_dirs: List[Path] = []

    print("Loading and relabeling source datasets ...")
    try:
        for raw_source in args.inputs:
            if _is_hf_repo_id(raw_source):
                src_tmp = tempfile.TemporaryDirectory(prefix="lerobot_color_src_")
                tmp_dirs.append(src_tmp)
                src = Path(src_tmp.name)
                _download_hf_dataset(raw_source, src)
            else:
                src = Path(raw_source)
                if not src.is_dir():
                    sys.exit(f"Error: source path does not exist: {src}")

            relabeled_tmp = tempfile.TemporaryDirectory(prefix="lerobot_color_relabel_")
            tmp_dirs.append(relabeled_tmp)
            relabeled = Path(relabeled_tmp.name) / _source_name(raw_source)
            relabel_dataset(
                src,
                relabeled,
                source_label=raw_source,
                allow_unchanged=args.allow_unchanged,
            )
            relabeled_dirs.append(relabeled)

        tmp_output_dir: Optional[tempfile.TemporaryDirectory] = None
        if output_is_hf:
            tmp_output_dir = tempfile.TemporaryDirectory(prefix="lerobot_color_merge_dst_")
            tmp_dirs.append(tmp_output_dir)
            dst = Path(tmp_output_dir.name) / "dataset"
        else:
            dst = Path(args.output)
            if dst.exists():
                sys.exit(f"Error: destination already exists: {dst}")

        merge(relabeled_dirs, dst)

        if output_is_hf:
            _push_to_hub(dst, hf_output_repo)
        else:
            print(f"Done  ->  {dst}")
    finally:
        for tmp_dir in tmp_dirs:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main()
