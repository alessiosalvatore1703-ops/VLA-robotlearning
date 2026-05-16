#!/usr/bin/env python3
"""
Debug script for the merge → augment pipeline.

Checks (without modifying anything):
  1. Python environment & required packages
  2. HF token validity and org write access
  3. All 6 source datasets exist and are readable
  4. Per-dataset stats (episodes, frames, fps, video keys, tasks)
  5. Prompt relabelling dry-run (shows old → new for every task)
  6. Intermediate / final repos — whether they already exist
  7. Brightness partition preview (how episodes will be split across levels)
"""

import os
import re
import sys
import json
import importlib
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

INPUTS = [
    "ETHrobotlearning/config1-red-blue-green",
    "ETHrobotlearning/config2-green-red-blue",
    "ETHrobotlearning/config3-green-blue-red",
    "ETHrobotlearning/config4-red-green-blue",
    "ETHrobotlearning/config5-blue-green-red",
    "ETHrobotlearning/config6-blue-red-green",
]

TMP_REPO   = "ETHrobotlearning/colours-task2-premerge"
FINAL_REPO = "ETHrobotlearning/colours-task2"

BRIGHTNESS_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.3, 1.1, 1.2]

UTILS_DIR = Path(__file__).parent / "datasets" / "utils"

# ── helpers ───────────────────────────────────────────────────────────────────

OK   = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"
INFO = "\033[36mℹ\033[0m"

def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)

def ok(msg):   print(f"  {OK}  {msg}")
def fail(msg): print(f"  {FAIL}  {msg}")
def warn(msg): print(f"  {WARN}  {msg}")
def info(msg): print(f"  {INFO}  {msg}")

errors = []

def record_fail(msg: str) -> None:
    fail(msg)
    errors.append(msg)

# ── 1. Packages ───────────────────────────────────────────────────────────────

section("1 · Python environment & packages")
info(f"Python: {sys.version.split()[0]}  ({sys.executable})")

REQUIRED = ["pandas", "numpy", "pyarrow", "av", "huggingface_hub", "tqdm"]
for pkg in REQUIRED:
    try:
        mod = importlib.import_module(pkg.replace("-", "_"))
        ver = getattr(mod, "__version__", "?")
        ok(f"{pkg} {ver}")
    except ImportError:
        record_fail(f"{pkg} is NOT installed")

# ── 2. HF token ───────────────────────────────────────────────────────────────

section("2 · HuggingFace token & org access")

token = os.environ.get("HF_TOKEN")
if not token:
    record_fail("HF_TOKEN env var is not set")
else:
    ok("HF_TOKEN is set")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info.get("name", "?")
        ok(f"Authenticated as: {username}")

        orgs = [o["name"] for o in user_info.get("orgs", [])]
        org = "ETHrobotlearning"
        if org in orgs:
            ok(f"Member of org: {org}")
        else:
            record_fail(f"Not a member of {org}. Orgs: {orgs}")
    except Exception as exc:
        record_fail(f"HF auth failed: {exc}")
        api = None

# ── 3. Source datasets ────────────────────────────────────────────────────────

section("3 · Source datasets (metadata only, no videos)")

dataset_info: dict = {}

if token:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)

    for repo_id in INPUTS:
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            ok(f"Exists: {repo_id}")
        except Exception as exc:
            record_fail(f"Cannot access {repo_id}: {exc}")
            dataset_info[repo_id] = None
            continue

        try:
            local = Path(snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns=["videos/*"],
                token=token,
            ))

            import pandas as pd
            import pyarrow.parquet as pq

            with open(local / "meta" / "info.json") as f:
                meta = json.load(f)

            parquets = sorted((local / "data" / "chunk-000").glob("*.parquet"))
            df = pd.concat([pq.read_table(p).to_pandas() for p in parquets], ignore_index=True)
            n_eps = int(df["episode_index"].max()) + 1
            n_frames = len(df)

            tasks_path = local / "meta" / "tasks.parquet"
            task_df = pq.read_table(tasks_path).to_pandas()

            video_keys = [k for k, v in meta.get("features", {}).items() if v.get("dtype") == "video"]

            dataset_info[repo_id] = {
                "local": local,
                "meta": meta,
                "n_eps": n_eps,
                "n_frames": n_frames,
                "fps": meta.get("fps"),
                "video_keys": video_keys,
                "task_df": task_df,
            }

            info(f"  episodes={n_eps}  frames={n_frames}  fps={meta.get('fps')}  video_keys={video_keys}")

        except Exception as exc:
            record_fail(f"  Failed to read {repo_id}: {exc}")
            dataset_info[repo_id] = None

# ── 4. Consistency checks ─────────────────────────────────────────────────────

section("4 · Cross-dataset consistency")

valid_ds = {k: v for k, v in dataset_info.items() if v is not None}

if len(valid_ds) >= 2:
    fps_vals   = {ds: v["fps"] for ds, v in valid_ds.items()}
    vkey_vals  = {ds: tuple(sorted(v["video_keys"])) for ds, v in valid_ds.items()}

    fps_set  = set(fps_vals.values())
    vkey_set = set(vkey_vals.values())

    if len(fps_set) == 1:
        ok(f"FPS consistent across all datasets: {next(iter(fps_set))}")
    else:
        record_fail(f"FPS mismatch: {fps_vals}")

    if len(vkey_set) == 1:
        ok(f"Video keys consistent: {next(iter(vkey_set))}")
    else:
        record_fail(f"Video key mismatch: {vkey_vals}")

    total_eps    = sum(v["n_eps"]    for v in valid_ds.values())
    total_frames = sum(v["n_frames"] for v in valid_ds.values())
    info(f"Total episodes across 6 datasets: {total_eps}")
    info(f"Total frames   across 6 datasets: {total_frames}")
    info(f"After merge+augment (×2):         {total_eps*2} episodes / {total_frames*2} frames")
else:
    warn("Skipping consistency checks (fewer than 2 datasets loaded)")

# ── 5. Prompt relabelling dry-run ─────────────────────────────────────────────

section("5 · Prompt relabelling dry-run")

sys.path.insert(0, str(UTILS_DIR))
try:
    from relabel_bowls_and_merge import _parse_color_order, _rewrite_prompt, TARGET_RE

    for repo_id in INPUTS:
        ds = dataset_info.get(repo_id)
        if ds is None:
            warn(f"Skipping {repo_id} (not loaded)")
            continue

        try:
            colors = _parse_color_order(repo_id)
            info(f"{Path(repo_id).name}: 1st={colors[0]} 2nd={colors[1]} 3rd={colors[2]}")
        except ValueError as exc:
            record_fail(f"Color parse failed for {repo_id}: {exc}")
            continue

        task_df = ds["task_df"]
        task_col = None
        for col in ["task"] + list(task_df.columns):
            if col in task_df.columns and pd.api.types.is_string_dtype(task_df[col]):
                task_col = col
                break
        if task_col is None:
            if pd.api.types.is_string_dtype(task_df.index):
                tasks = list(task_df.index)
            else:
                warn(f"  Cannot find task column in {repo_id}")
                continue
        else:
            tasks = list(task_df[task_col])

        all_matched = True
        for task in tasks:
            new_task = _rewrite_prompt(task, colors)
            if new_task == task:
                warn(f"  UNCHANGED: {task!r}")
                all_matched = False
            else:
                ok(f"  {task!r}")
                info(f"      → {new_task!r}")
        if all_matched:
            ok(f"  All prompts matched and will be relabelled")

except ImportError as exc:
    record_fail(f"Could not import relabel_bowls_and_merge: {exc}")


# ── 6. Intermediate / final repo collision ────────────────────────────────────

section("6 · Hub repo collision check")

if token:
    for repo_id, label in [(TMP_REPO, "intermediate"), (FINAL_REPO, "final")]:
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            warn(f"{label} repo already exists: {repo_id}  ← will cause the script to fail or overwrite")
        except Exception:
            ok(f"{label} repo does not exist yet: {repo_id}")

# ── 7. Brightness partition preview ───────────────────────────────────────────

section("7 · Brightness partition preview")

if valid_ds:
    total_eps = sum(v["n_eps"] for v in valid_ds.values())
    n_levels = len(BRIGHTNESS_LEVELS)
    info(f"Brightness levels ({n_levels}): {BRIGHTNESS_LEVELS}")
    info(f"Merged episode count (before augment): {total_eps}")

    partition_counts = [0] * n_levels
    for ep_id in range(total_eps):
        p = (ep_id * n_levels) // total_eps
        partition_counts[p] += 1

    for i, (level, count) in enumerate(zip(BRIGHTNESS_LEVELS, partition_counts)):
        info(f"  brightness={level:.2f}  →  {count} episodes")

    info(f"After augmentation: {total_eps * 2} total episodes  ({total_eps} original + {total_eps} augmented)")
else:
    warn("Skipping brightness preview (no datasets loaded)")

# ── Summary ───────────────────────────────────────────────────────────────────

section("Summary")

if errors:
    print(f"\n  {FAIL}  {len(errors)} problem(s) found — fix before running the pipeline:\n")
    for e in errors:
        print(f"     • {e}")
    sys.exit(1)
else:
    print(f"\n  {OK}  All checks passed. Ready to run run_merge_augment_colours.sh")
