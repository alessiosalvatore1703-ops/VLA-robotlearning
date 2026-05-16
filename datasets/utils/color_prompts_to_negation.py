#!/usr/bin/env python3
"""Convert color bowl prompts to negation-style prompts in a LeRobot v3 dataset.

Input prompts such as:

    Put the banana in the red colored bowl

are rewritten to:

    Put the banana into the bowl that is not green and not blue.

By default this reads ETHrobotlearning/task2-colors and pushes
ETHrobotlearning/task2-negation.

Usage:
    python datasets/utils/color_prompts_to_negation.py

    python datasets/utils/color_prompts_to_negation.py \\
        --input ETHrobotlearning/task2-colors \\
        --output ETHrobotlearning/task2-negation
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

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
)
from relabel_bowls_and_merge import (
    _dedupe_preserve_order,
    _read_tasks,
    _replace_task_values,
    _set_task_index_stats,
)


DEFAULT_INPUT = "ETHrobotlearning/task2-colors"
DEFAULT_OUTPUT = "ETHrobotlearning/task2-negation"

NEGATED_COLORS = {
    "red": ("green", "blue"),
    "green": ("red", "blue"),
    "blue": ("red", "green"),
}

COLOR_PROMPT_RE = re.compile(
    r"^Put\s+the\s+(?P<object>.+?)\s+"
    r"(?:into|in|to|inside)\s+(?:the\s+)?"
    r"(?P<color>red|green|blue)(?:\s+colored)?\s+bowl\.?$",
    re.IGNORECASE,
)


def _rewrite_prompt(task: str) -> str:
    match = COLOR_PROMPT_RE.match(task.strip())
    if not match:
        return task

    obj = match.group("object").strip()
    color = match.group("color").lower()
    first_negated, second_negated = NEGATED_COLORS[color]
    return (
        f"Put the {obj} into the bowl that is not "
        f"{first_negated} and not {second_negated}."
    )


def convert_dataset(src: Path, dst: Path, allow_unchanged: bool = False) -> None:
    """Copy one LeRobot v3 dataset and rewrite color prompts to negation prompts."""
    meta_dir = src / "meta"
    tasks_path = meta_dir / "tasks.parquet"
    if not tasks_path.exists():
        raise ValueError(f"{src} is not a LeRobot v3 dataset: missing meta/tasks.parquet")
    if dst.exists():
        raise ValueError(f"Destination already exists: {dst}")

    task_df, old_task_indices, old_tasks = _read_tasks(tasks_path)
    if not old_tasks:
        raise ValueError(f"No task prompts found in {tasks_path}")

    old_to_new_task: Dict[str, str] = {}
    unchanged: List[str] = []
    for task in old_tasks:
        new_task = _rewrite_prompt(task)
        old_to_new_task[task] = new_task
        if new_task == task:
            unchanged.append(task)

    if unchanged and not allow_unchanged:
        examples = "\n".join(f"  {task!r}" for task in unchanged[:5])
        raise ValueError(
            f"{len(unchanged)} task prompt(s) did not match the color bowl pattern. "
            f"Pass --allow-unchanged to keep them.\n{examples}"
        )

    new_tasks = _dedupe_preserve_order(old_to_new_task[task] for task in old_tasks)
    new_task_to_index = {task: idx for idx, task in enumerate(new_tasks)}
    old_task_index_to_new = {
        old_idx: new_task_to_index[old_to_new_task[old_task]]
        for old_idx, old_task in zip(old_task_indices, old_tasks)
    }

    print("Task prompt rewrites:")
    for old_task in old_tasks:
        print(f"  {old_task!r} -> {old_to_new_task[old_task]!r}")

    print("Copying dataset ...")
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

    for data_file in tqdm(data_files, desc="updating data", unit="file"):
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
    for episode_file in tqdm(episode_files, desc="updating episodes", unit="file"):
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
        description="Convert color bowl prompts to negation-style prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        metavar="SRC",
        help=f"Source dataset: local path or HF repo ID. Default: {DEFAULT_INPUT}",
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
        help="Keep task prompts that do not match the color bowl pattern instead of failing.",
    )
    args = ap.parse_args()

    input_is_hf = _is_hf_repo_id(args.input)
    output_is_hf = _is_bare_name(args.output) or _is_hf_repo_id(args.output)

    if input_is_hf or output_is_hf:
        _require_hf()

    hf_output_repo: Optional[str] = None
    if output_is_hf:
        hf_output_repo = (
            args.output if _is_hf_repo_id(args.output) else f"{_hf_whoami()}/{args.output}"
        )

    tmp_dirs: List[tempfile.TemporaryDirectory] = []
    try:
        if input_is_hf:
            src_tmp = tempfile.TemporaryDirectory(prefix="lerobot_negation_src_")
            tmp_dirs.append(src_tmp)
            src = Path(src_tmp.name)
            _download_hf_dataset(args.input, src)
        else:
            src = Path(args.input)
            if not src.is_dir():
                sys.exit(f"Error: source path does not exist: {src}")

        if output_is_hf:
            dst_tmp = tempfile.TemporaryDirectory(prefix="lerobot_negation_dst_")
            tmp_dirs.append(dst_tmp)
            dst = Path(dst_tmp.name) / "dataset"
        else:
            dst = Path(args.output)
            if dst.exists():
                sys.exit(f"Error: destination already exists: {dst}")

        convert_dataset(src, dst, allow_unchanged=args.allow_unchanged)

        if output_is_hf:
            _push_to_hub(dst, hf_output_repo)
        else:
            print(f"Done  ->  {dst}")
    finally:
        for tmp_dir in tmp_dirs:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main()
