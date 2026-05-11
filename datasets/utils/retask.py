#!/usr/bin/env python3
"""Replace the task prompt in a single-prompt LeRobot dataset.

Rewrites meta/tasks.parquet (or tasks.jsonl for v2.x) and the tasks column
in every meta/episodes parquet.  All other files are copied unchanged.

Input  — local path OR Hugging Face repo ID  (e.g. username/my_dataset)
Output — local path OR bare dataset name     (e.g. my_dataset_retask → pushed to your HF account)

Usage:
    # HF → HF
    python datasets/utils/retask.py \\
        --input  alice/my_dataset \\
        --output my_dataset_retask \\
        --task   "Put the banana in the green bowl."

    # local → local
    python datasets/utils/retask.py \\
        --input  /data/dataset \\
        --output /data/dataset_retask \\
        --task   "Put the banana in the green bowl."
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------

def _require_hf() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("Error: huggingface_hub is not installed.  Run: pip install huggingface_hub")


def _hf_whoami() -> str:
    from huggingface_hub import whoami
    try:
        return whoami()["name"]
    except Exception as exc:
        sys.exit(f"Error: could not read HF identity ({exc}).  Run: huggingface-cli login")


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
    print(f"Downloading {repo_id} from Hugging Face Hub …")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    print("Download complete.")


def _push_to_hub(local_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Creating / updating HF dataset repo: {repo_id} …")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading to {repo_id} …")
    api.upload_folder(folder_path=str(local_path), repo_id=repo_id, repo_type="dataset")
    print(f"Pushed → https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Task detection helpers
# ---------------------------------------------------------------------------

def _load_tasks_v3(meta_dir: Path) -> list[str]:
    """Return list of task strings from tasks.parquet (v3.x format)."""
    df = pd.read_parquet(meta_dir / "tasks.parquet")
    if pd.api.types.is_string_dtype(df.index):
        return list(df.index)
    if "task" in df.columns and pd.api.types.is_string_dtype(df["task"]):
        return list(df["task"])
    df2 = df.reset_index()
    for col in df2.columns:
        if pd.api.types.is_string_dtype(df2[col]):
            return list(df2[col])
    return []


def _load_tasks_v2(meta_dir: Path) -> list[str]:
    """Return list of task strings from tasks.jsonl (v2.x format)."""
    tasks = []
    with open(meta_dir / "tasks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line)["task"])
    return tasks


def _is_v3(src: Path) -> bool:
    return (src / "meta" / "tasks.parquet").exists()


# ---------------------------------------------------------------------------
# Core retask
# ---------------------------------------------------------------------------

def retask(src: Path, dst: Path, new_task: str) -> None:
    meta_dir = src / "meta"

    # --- detect format and load current tasks ---
    if _is_v3(src):
        current_tasks = _load_tasks_v3(meta_dir)
    elif (meta_dir / "tasks.jsonl").exists():
        current_tasks = _load_tasks_v2(meta_dir)
    else:
        sys.exit("Error: could not find tasks.parquet or tasks.jsonl in meta/.")

    if len(current_tasks) != 1:
        sys.exit(
            f"Error: expected exactly 1 task prompt, found {len(current_tasks)}:\n"
            + "\n".join(f"  {t!r}" for t in current_tasks)
        )

    old_task = current_tasks[0]
    print(f"Old task: {old_task!r}")
    print(f"New task: {new_task!r}")

    # --- copy everything to dst ---
    print("Copying dataset …")
    if dst.exists():
        sys.exit(f"Error: destination already exists: {dst}")
    shutil.copytree(src, dst)

    dst_meta = dst / "meta"

    # --- update tasks.parquet (v3) or tasks.jsonl (v2) ---
    if _is_v3(src):
        src_df = pd.read_parquet(meta_dir / "tasks.parquet")
        new_index = [new_task if t == old_task else t for t in src_df.index]
        new_df = src_df.copy()
        new_df.index = pd.Index(new_index, name=src_df.index.name)
        new_df.to_parquet(dst_meta / "tasks.parquet")
    else:
        with open(dst_meta / "tasks.jsonl", "w") as f:
            for line in open(meta_dir / "tasks.jsonl"):
                entry = json.loads(line)
                if entry.get("task") == old_task:
                    entry["task"] = new_task
                f.write(json.dumps(entry) + "\n")

    # --- update episodes parquets (v3 only: tasks column holds arrays) ---
    if _is_v3(src):
        ep_files = sorted((dst_meta / "episodes").rglob("*.parquet"))
        for ep_file in tqdm(ep_files, desc="updating episodes", unit="file"):
            df = pd.read_parquet(ep_file)
            if "tasks" in df.columns:
                def _replace(val: object) -> object:
                    if isinstance(val, np.ndarray):
                        return np.array(
                            [new_task if t == old_task else t for t in val],
                            dtype=val.dtype,
                        )
                    if val == old_task:
                        return new_task
                    return val
                df["tasks"] = df["tasks"].map(_replace)
                df.to_parquet(ep_file, index=False)

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replace the task prompt in a single-prompt LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",  required=True, metavar="SRC",
                    help="Source dataset: local path or HF repo ID (user/dataset).")
    ap.add_argument("--output", required=True, metavar="DST",
                    help="Output: local path, or bare name to push to your HF account.")
    ap.add_argument("--task",   required=True, metavar="TASK",
                    help="New task prompt string.")
    args = ap.parse_args()

    input_is_hf  = _is_hf_repo_id(args.input)
    output_is_hf = _is_bare_name(args.output) or _is_hf_repo_id(args.output)

    if input_is_hf or output_is_hf:
        _require_hf()

    hf_output_repo: Optional[str] = None
    if output_is_hf:
        if _is_hf_repo_id(args.output):
            hf_output_repo = args.output
        else:
            hf_output_repo = f"{_hf_whoami()}/{args.output}"

    tmp_input_dir: Optional[tempfile.TemporaryDirectory] = None
    if input_is_hf:
        tmp_input_dir = tempfile.TemporaryDirectory(prefix="lerobot_retask_src_")
        src = Path(tmp_input_dir.name)
        _download_hf_dataset(args.input, src)
    else:
        src = Path(args.input)
        if not src.is_dir():
            sys.exit(f"Error: source path does not exist: {src}")

    tmp_output_dir: Optional[tempfile.TemporaryDirectory] = None
    if output_is_hf:
        tmp_output_dir = tempfile.TemporaryDirectory(prefix="lerobot_retask_dst_")
        dst = Path(tmp_output_dir.name) / "dataset"
    else:
        dst = Path(args.output)

    try:
        retask(src, dst, args.task)
        if output_is_hf:
            _push_to_hub(dst, hf_output_repo)
        else:
            print(f"\nDone  →  {dst}")
    finally:
        if tmp_input_dir is not None:
            tmp_input_dir.cleanup()
        if tmp_output_dir is not None:
            tmp_output_dir.cleanup()


if __name__ == "__main__":
    main()
