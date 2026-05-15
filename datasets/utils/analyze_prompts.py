#!/usr/bin/env python3
"""Analyze task prompts in a LeRobot v2/v3 dataset.

Prints:
  - All unique prompts with episode counts and percentages
  - Basic stats: total episodes, unique prompts, avg/min/max prompt length
  - Flag for potential duplicates (same text after lowercasing / stripping)

Input — local path OR HF repo ID (e.g. username/my_dataset)

Usage:
    python datasets/utils/analyze_prompts.py --input /data/my_dataset
    python datasets/utils/analyze_prompts.py --input alice/my_dataset
    python datasets/utils/analyze_prompts.py --input /data/my_dataset --episodes
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# HF helpers  (same pattern as other utils)
# ---------------------------------------------------------------------------

def _require_hf() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("huggingface_hub not installed.  Run: pip install huggingface_hub")


def _is_hf_repo_id(s: str) -> bool:
    p = Path(s)
    if p.exists():
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts)


def _download_hf_dataset(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo_id} from Hugging Face Hub …")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    print("Download complete.")


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def _is_v3(root: Path) -> bool:
    return (root / "meta" / "tasks.parquet").exists()


def _load_tasks_v3(meta_dir: Path) -> list[str]:
    import pandas as pd
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
    tasks = []
    with open(meta_dir / "tasks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line)["task"])
    return tasks


# ---------------------------------------------------------------------------
# Episode → task mapping
# ---------------------------------------------------------------------------

def _episode_task_counts_v3(meta_dir: Path, all_tasks: list[str]) -> Counter:
    """Count how many episodes are assigned each task (v3: episodes/ parquets)."""
    import numpy as np
    import pandas as pd

    counter: Counter = Counter()
    ep_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
    for ep_file in ep_files:
        df = pd.read_parquet(ep_file)
        if "tasks" not in df.columns:
            continue
        for val in df["tasks"]:
            if isinstance(val, np.ndarray):
                for t in val:
                    counter[str(t)] += 1
            elif val is not None:
                counter[str(val)] += 1
    return counter


def _episode_task_counts_v2(meta_dir: Path) -> Counter:
    """Count per-episode task assignments from episodes.jsonl (v2)."""
    counter: Counter = Counter()
    ep_file = meta_dir / "episodes.jsonl"
    if not ep_file.exists():
        return counter
    with open(ep_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            tasks = entry.get("tasks", [])
            if isinstance(tasks, list):
                for t in tasks:
                    counter[t] += 1
            elif isinstance(tasks, str):
                counter[tasks] += 1
    return counter


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(root: Path, show_episodes: bool) -> None:
    meta_dir = root / "meta"

    if _is_v3(root):
        all_tasks = _load_tasks_v3(meta_dir)
        task_counts = _episode_task_counts_v3(meta_dir, all_tasks)
        version = "v3"
    elif (meta_dir / "tasks.jsonl").exists():
        all_tasks = _load_tasks_v2(meta_dir)
        task_counts = _episode_task_counts_v2(meta_dir)
        version = "v2"
    else:
        sys.exit("Could not find tasks.parquet or tasks.jsonl in meta/.")

    # Fall back: if no episode mapping found, treat each task as count=unknown
    if not task_counts:
        task_counts = Counter({t: 0 for t in all_tasks})

    total_episodes = sum(task_counts.values()) or len(all_tasks)
    unique_prompts = list(dict.fromkeys(all_tasks))  # preserve order, deduplicated

    # ---- per-task rows ----
    lengths = [len(t) for t in unique_prompts]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n{'='*70}")
    print(f"Dataset : {root}")
    print(f"Format  : LeRobot {version}")
    print(f"Unique prompts : {len(unique_prompts)}")
    print(f"Total episodes : {total_episodes}")
    print(f"Prompt length  : avg={avg_len:.0f}  min={min(lengths) if lengths else 0}  max={max(lengths) if lengths else 0}")
    print(f"{'='*70}\n")

    # Sort by episode count descending
    sorted_tasks = sorted(unique_prompts, key=lambda t: task_counts.get(t, 0), reverse=True)

    col_w = 55
    print(f"{'#':<4} {'Count':>6}  {'%':>5}  Prompt")
    print(f"{'-'*4} {'-'*6}  {'-'*5}  {'-'*col_w}")
    for i, task in enumerate(sorted_tasks, 1):
        count = task_counts.get(task, 0)
        pct   = count / total_episodes * 100 if total_episodes else 0
        # Truncate long prompts for display
        display = task if len(task) <= col_w else task[:col_w - 3] + "…"
        print(f"{i:<4} {count:>6}  {pct:>4.1f}%  {display!r}")

    # ---- near-duplicate detection ----
    normalized: dict[str, list[str]] = {}
    for task in unique_prompts:
        key = " ".join(task.lower().split()).rstrip(".")
        normalized.setdefault(key, []).append(task)

    near_dupes = {k: v for k, v in normalized.items() if len(v) > 1}
    if near_dupes:
        print(f"\n  WARNING: {len(near_dupes)} near-duplicate group(s) detected (differ only in case/punctuation/whitespace):")
        for key, variants in near_dupes.items():
            print(f"    Normalized: {key!r}")
            for v in variants:
                print(f"      - {v!r}")

    # ---- per-episode detail ----
    if show_episodes and version == "v3":
        import pandas as pd, numpy as np
        print(f"\n{'='*70}")
        print("Episode → task mapping")
        print(f"{'='*70}")
        ep_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
        for ep_file in ep_files:
            df = pd.read_parquet(ep_file)
            if "tasks" not in df.columns:
                continue
            ep_ids = df.get("episode_index", df.index).tolist()
            for ep_id, val in zip(ep_ids, df["tasks"]):
                tasks_list = list(val) if isinstance(val, np.ndarray) else [val]
                for t in tasks_list:
                    print(f"  ep {ep_id:>5}: {t!r}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze task prompts in a LeRobot v2/v3 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input", required=True, metavar="SRC",
                    help="Local path or HF repo ID (user/dataset).")
    ap.add_argument("--episodes", action="store_true",
                    help="Also print episode-level task assignments (v3 only).")
    args = ap.parse_args()

    is_hf = _is_hf_repo_id(args.input)
    if is_hf:
        _require_hf()

    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if is_hf:
        tmp_dir = tempfile.TemporaryDirectory(prefix="lerobot_analyze_")
        root = Path(tmp_dir.name)
        _download_hf_dataset(args.input, root)
    else:
        root = Path(args.input)
        if not root.is_dir():
            sys.exit(f"Path does not exist: {root}")

    try:
        analyze(root, show_episodes=args.episodes)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main()
