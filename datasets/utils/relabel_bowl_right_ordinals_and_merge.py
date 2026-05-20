#!/usr/bin/env python3
"""Relabel left-counted ordinal bowl prompts as right-counted prompts, then merge.

The default inputs are the six ETHrobotlearning color-configuration datasets.
Each source dataset contains prompts such as:

    Put the banana into the 1st bowl from the left from the robot perspective

This script preserves the physical target bowl, but rewrites the prompt to
count from the right:

    Put the banana into the 3rd bowl from the right from the robot perspective

Then it merges the relabeled LeRobot v3 datasets and optionally pushes the
result to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from merge_datasets import merge
from relabel_bowl_positions_and_merge import (
    DEFAULT_INPUTS,
    TASK_RE,
    _download_hf_dataset,
    _hf_whoami,
    _is_bare_name,
    _is_hf_repo_id,
    _load_tasks,
    _push_to_hub,
    _replace_task_value,
    _require_hf,
    _write_tasks,
)


RIGHT_ORDINAL_BY_LEFT_ORDINAL = {
    1: "3rd",
    2: "2nd",
    3: "1st",
}


def relabel_task(task: str) -> str:
    match = TASK_RE.match(task)
    if not match:
        raise ValueError(f"could not parse ordinal bowl prompt: {task!r}")

    obj = " ".join(match.group("object").split())
    left_ordinal = int(match.group("ordinal"))
    right_ordinal = RIGHT_ORDINAL_BY_LEFT_ORDINAL[left_ordinal]
    return f"Put the {obj} into the {right_ordinal} bowl from the right from the robot perspective"


def relabel_dataset(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    meta_dir = dst / "meta"
    tasks_df, old_tasks, storage = _load_tasks(meta_dir)
    new_tasks = [relabel_task(task) for task in old_tasks]
    task_map: Dict[str, str] = dict(zip(old_tasks, new_tasks))

    _write_tasks(meta_dir, tasks_df, new_tasks, storage)

    episode_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
    for ep_file in episode_files:
        ep_df = pd.read_parquet(ep_file)
        if "tasks" in ep_df.columns:
            ep_df["tasks"] = ep_df["tasks"].map(lambda v: _replace_task_value(v, task_map))
            ep_df.to_parquet(ep_file, index=False)

    changed = sum(1 for old, new in zip(old_tasks, new_tasks) if old != new)
    print(f"  Relabeled {changed}/{len(old_tasks)} task prompt(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel left-counted bowl prompts as right-counted ordinal prompts and merge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        metavar="SRC",
        help="Source LeRobot v3 datasets, local paths or HF repo IDs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DST",
        help="Output local path, bare dataset name, or HF repo ID.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary downloaded and relabeled datasets for inspection.",
    )
    args = parser.parse_args()

    any_hf_input = any(_is_hf_repo_id(s) for s in args.inputs)
    output_is_hf = _is_bare_name(args.output) or _is_hf_repo_id(args.output)

    if any_hf_input or output_is_hf:
        _require_hf()

    hf_output_repo: Optional[str] = None
    if output_is_hf:
        hf_output_repo = args.output if _is_hf_repo_id(args.output) else f"{_hf_whoami()}/{args.output}"

    if args.keep_work_dir:
        tmp_root: Optional[tempfile.TemporaryDirectory] = None
        work_root = Path(tempfile.mkdtemp(prefix="lerobot_right_ordinal_relabel_"))
    else:
        tmp_root = tempfile.TemporaryDirectory(prefix="lerobot_right_ordinal_relabel_")
        work_root = Path(tmp_root.name)
    local_sources: List[Path] = []
    relabeled_sources: List[Path] = []

    try:
        print("Loading source datasets ...")
        for i, raw in enumerate(args.inputs):
            if _is_hf_repo_id(raw):
                src = work_root / f"source-{i:02d}"
                _download_hf_dataset(raw, src)
            else:
                src = Path(raw)
                if not src.is_dir():
                    sys.exit(f"Error: source path does not exist: {src}")
            local_sources.append(src)

        print("\nRelabeling task prompts ...")
        for i, src in enumerate(tqdm(local_sources, desc="relabeling", unit="dataset")):
            dst = work_root / f"relabeled-{i:02d}"
            relabel_dataset(src, dst)
            relabeled_sources.append(dst)

        if output_is_hf:
            merged_dst = work_root / "merged"
        else:
            merged_dst = Path(args.output)
            if merged_dst.exists():
                sys.exit(f"Error: destination already exists: {merged_dst}")

        print("\nMerging relabeled datasets ...")
        merge(relabeled_sources, merged_dst)

        if output_is_hf:
            _push_to_hub(merged_dst, hf_output_repo)
        else:
            print(f"Done -> {merged_dst}")

        if args.keep_work_dir:
            print(f"Work dir kept at: {work_root}")
    finally:
        if tmp_root is not None:
            tmp_root.cleanup()


if __name__ == "__main__":
    main()
