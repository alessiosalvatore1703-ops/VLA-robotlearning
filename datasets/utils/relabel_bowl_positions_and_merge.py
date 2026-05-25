#!/usr/bin/env python3
"""Relabel ordinal bowl prompts as position prompts, then merge datasets.

The default inputs are the six ETHrobotlearning color-configuration datasets.
Each source dataset contains prompts such as:

    Put the banana into the 2nd bowl from the left from the robot perspective

This script rewrites those tasks as:

    Put the banana into the bowl on the center from the robot perspective

Then it merges the relabeled LeRobot v3 datasets and optionally pushes the
result to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from merge_datasets import merge


DEFAULT_INPUTS = [
    "ETHrobotlearning/config1-red-blue-green",
    "ETHrobotlearning/config2-green-red-blue",
    "ETHrobotlearning/config3-green-blue-red",
    "ETHrobotlearning/config4-red-green-blue",
    "ETHrobotlearning/config5-blue-green-red",
    "ETHrobotlearning/config6-blue-red-green",
]

POSITION_BY_ORDINAL = {
    1: "left",
    2: "center",
    3: "right",
}

TASK_RE = re.compile(
    r"^\s*put\s+the\s+(?P<object>.+?)\s+into\s+the\s+"
    r"(?P<ordinal>[1-3])\s*(?:st|nd|rd|th|[a-z]{2})?\s+"
    r"bowl\s+from\s+the\s+left\s+from\s+the\s+robot\s+perspective\s*\.?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Hugging Face helpers
# ---------------------------------------------------------------------------

def _require_hf() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")


def _hf_whoami() -> str:
    from huggingface_hub import whoami
    try:
        return whoami()["name"]
    except Exception as exc:
        sys.exit(f"Error: could not read HF identity ({exc}). Run: huggingface-cli login")


def _is_hf_repo_id(s: str) -> bool:
    p = Path(s)
    if p.exists():
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts)


def _is_bare_name(s: str) -> bool:
    return "/" not in s and not Path(s).is_absolute() and not s.startswith(".")


def _download_hf_dataset(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))


def _push_to_hub(local_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Creating / updating HF dataset repo: {repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading to {repo_id} ...")
    api.upload_folder(folder_path=str(local_path), repo_id=repo_id, repo_type="dataset")
    print(f"Pushed -> https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Relabeling helpers
# ---------------------------------------------------------------------------

def relabel_task(task: str) -> str:
    match = TASK_RE.match(task)
    if not match:
        raise ValueError(f"could not parse ordinal bowl prompt: {task!r}")

    obj = " ".join(match.group("object").split())
    ordinal = int(match.group("ordinal"))
    position = POSITION_BY_ORDINAL[ordinal]
    return f"Put the {obj} into the bowl on the {position} from the robot perspective"


def _load_tasks(meta_dir: Path) -> tuple[pd.DataFrame, List[str], str]:
    tasks_path = meta_dir / "tasks.parquet"
    if not tasks_path.exists():
        sys.exit(f"Error: {tasks_path} not found. Only LeRobot v3 datasets are supported.")

    df = pd.read_parquet(tasks_path)
    if pd.api.types.is_string_dtype(df.index):
        return df, [str(t) for t in df.index], "index"
    if "task" in df.columns and pd.api.types.is_string_dtype(df["task"]):
        return df, [str(t) for t in df["task"]], "task"

    df2 = df.reset_index()
    for col in df2.columns:
        if pd.api.types.is_string_dtype(df2[col]):
            return df, [str(t) for t in df2[col]], col

    sys.exit(f"Error: could not find task strings in {tasks_path}")


def _write_tasks(meta_dir: Path, df: pd.DataFrame, tasks: List[str], storage: str) -> None:
    out = df.copy()
    if storage == "index":
        out.index = pd.Index(tasks, name=df.index.name)
    elif storage in out.columns:
        out[storage] = tasks
    else:
        # Fall back to the canonical LeRobot v3 shape used by merge_datasets.py.
        out = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks))
    out.to_parquet(meta_dir / "tasks.parquet")


def _replace_task_value(value: object, task_map: Dict[str, str]) -> object:
    if isinstance(value, np.ndarray):
        return np.array([task_map.get(str(v), str(v)) for v in value], dtype=object)
    if isinstance(value, list):
        return [task_map.get(str(v), str(v)) for v in value]
    if isinstance(value, tuple):
        return tuple(task_map.get(str(v), str(v)) for v in value)
    if isinstance(value, str):
        return task_map.get(value, value)
    return value


def relabel_dataset(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    meta_dir = dst / "meta"
    tasks_df, old_tasks, storage = _load_tasks(meta_dir)
    new_tasks = [relabel_task(task) for task in old_tasks]
    task_map = dict(zip(old_tasks, new_tasks))

    _write_tasks(meta_dir, tasks_df, new_tasks, storage)

    episode_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
    for ep_file in episode_files:
        ep_df = pd.read_parquet(ep_file)
        if "tasks" in ep_df.columns:
            ep_df["tasks"] = ep_df["tasks"].map(lambda v: _replace_task_value(v, task_map))
            ep_df.to_parquet(ep_file, index=False)

    changed = sum(1 for old, new in zip(old_tasks, new_tasks) if old != new)
    print(f"  Relabeled {changed}/{len(old_tasks)} task prompt(s).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel ordinal bowl prompts as left/center/right prompts and merge.",
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
        work_root = Path(tempfile.mkdtemp(prefix="lerobot_position_relabel_"))
    else:
        tmp_root = tempfile.TemporaryDirectory(prefix="lerobot_position_relabel_")
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
