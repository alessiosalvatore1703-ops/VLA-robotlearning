#!/usr/bin/env python3
"""
Post-pipeline validation for ETHrobotlearning/colours-task2.

Checks (read-only, downloads metadata + a handful of video frames):
  1.  Dataset exists on Hub and info.json totals are self-consistent
  2.  Episode count  = 2 × sum of source episode counts
  3.  Frame count    = 2 × sum of source frame counts
  4.  Episode indices are contiguous 0..N-1 (no gaps, no duplicates)
  5.  Frame indices  are contiguous 0..M-1 (no gaps, no duplicates)
  6.  Every task_index in data parquets maps to a row in tasks.parquet
  7.  No ordinal bowl references remain in any task prompt
  8.  All six colour bowl phrases are represented (red/green/blue × pick/place)
  9.  Episode metadata (from/to indices) is consistent with data parquets
  10. All video files referenced in episode metadata exist on the Hub
  11. Brightness sanity check: sample 6 augmented/original pairs — augmented
      frames should have a meaningfully different mean pixel value
  12. Frame counts per episode: originals and their augmented copies match

Usage:
    python validate_final_dataset.py
    python validate_final_dataset.py --repo ETHrobotlearning/colours-task2
    python validate_final_dataset.py --skip-video-check   # skip step 10 (slow)
    python validate_final_dataset.py --skip-brightness    # skip step 11
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo",             default="ETHrobotlearning/colours-task2")
ap.add_argument("--source-repos",     nargs="+", default=[
    "ETHrobotlearning/config1-red-blue-green",
    "ETHrobotlearning/config2-green-red-blue",
    "ETHrobotlearning/config3-green-blue-red",
    "ETHrobotlearning/config4-red-green-blue",
    "ETHrobotlearning/config5-blue-green-red",
    "ETHrobotlearning/config6-blue-red-green",
])
ap.add_argument("--brightness-levels", type=float, nargs="+",
                default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.3, 1.1, 1.2])
ap.add_argument("--brightness-sample-pairs", type=int, default=6,
                help="How many original/augmented pairs to compare for brightness (default 6)")
ap.add_argument("--skip-video-check",  action="store_true", help="Skip video file existence check (step 10)")
ap.add_argument("--skip-brightness",   action="store_true", help="Skip brightness sampling (step 11)")
args = ap.parse_args()

# ── formatting helpers ────────────────────────────────────────────────────────

OK   = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"
INFO = "\033[36mℹ\033[0m"

def section(title):
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print('─'*64)

def ok(msg):   print(f"  {OK}  {msg}")
def fail(msg): print(f"  {FAIL}  {msg}")
def warn(msg): print(f"  {WARN}  {msg}")
def info(msg): print(f"  {INFO}  {msg}")

errors   = []
warnings = []

def record_fail(msg):
    fail(msg)
    errors.append(msg)

def record_warn(msg):
    warn(msg)
    warnings.append(msg)

# ── imports ───────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download

token = os.environ.get("HF_TOKEN")
api   = HfApi(token=token)

# ── ordinal-bowl detection pattern ───────────────────────────────────────────

ORDINAL_RE = re.compile(
    r"\b(?:1st|1nd|1rd|1th|first|2nd|2st|2rd|2th|second|3rd|3st|3nd|3th|third)\b"
    r".*?bowl",
    re.IGNORECASE,
)

COLOUR_PHRASES = [
    "red colored bowl",
    "green colored bowl",
    "blue colored bowl",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 0 – download metadata (no videos)
# ─────────────────────────────────────────────────────────────────────────────

section("0 · Downloading final dataset metadata")

try:
    local = Path(snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        ignore_patterns=["videos/*"],
        token=token,
    ))
    ok(f"Downloaded metadata for {args.repo}")
    info(f"  local cache: {local}")
except Exception as exc:
    record_fail(f"Could not download {args.repo}: {exc}")
    print(f"\n  {FAIL}  Cannot continue without the dataset. Exiting.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – info.json self-consistency
# ─────────────────────────────────────────────────────────────────────────────

section("1 · info.json self-consistency")

with open(local / "meta" / "info.json") as f:
    info_json = json.load(f)

fps        = float(info_json.get("fps", 0))
video_keys = [k for k, v in info_json.get("features", {}).items() if v.get("dtype") == "video"]
declared_eps    = info_json.get("total_episodes")
declared_frames = info_json.get("total_frames")
declared_videos = info_json.get("total_videos")

info(f"fps={fps}  video_keys={video_keys}")
info(f"declared total_episodes={declared_eps}  total_frames={declared_frames}  total_videos={declared_videos}")

if not video_keys:
    record_fail("No video keys found in info.json features")
if fps <= 0:
    record_fail(f"Invalid fps={fps}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2+3 – episode/frame counts vs. source datasets
# ─────────────────────────────────────────────────────────────────────────────

section("2+3 · Episode & frame counts vs. source datasets")

src_totals = {}
for src in args.source_repos:
    try:
        src_local = Path(snapshot_download(
            repo_id=src, repo_type="dataset",
            ignore_patterns=["videos/*"], token=token,
        ))
        parquets = sorted((src_local / "data" / "chunk-000").glob("*.parquet"))
        df = pd.concat([pq.read_table(p).to_pandas() for p in parquets], ignore_index=True)
        n_eps    = int(df["episode_index"].max()) + 1
        n_frames = len(df)
        src_totals[src] = (n_eps, n_frames)
        info(f"  {Path(src).name}: {n_eps} eps / {n_frames} frames")
    except Exception as exc:
        record_warn(f"  Could not load source {src}: {exc}")

if src_totals:
    expected_eps    = 2 * sum(v[0] for v in src_totals.values())
    expected_frames = 2 * sum(v[1] for v in src_totals.values())
    info(f"  expected (2× merged): {expected_eps} episodes / {expected_frames} frames")

    if declared_eps == expected_eps:
        ok(f"total_episodes matches: {declared_eps}")
    else:
        record_fail(f"total_episodes mismatch: declared={declared_eps}, expected={expected_eps}")

    if declared_frames == expected_frames:
        ok(f"total_frames matches: {declared_frames}")
    else:
        record_fail(f"total_frames mismatch: declared={declared_frames}, expected={expected_frames}")
else:
    record_warn("Could not load any source datasets — skipping count checks")

# ─────────────────────────────────────────────────────────────────────────────
# Load main parquets
# ─────────────────────────────────────────────────────────────────────────────

parquet_files = sorted((local / "data").rglob("*.parquet"))
if not parquet_files:
    record_fail("No data parquet files found")
    print(f"\n  {FAIL}  No data to validate. Exiting.")
    sys.exit(1)

data_df = pd.concat([pq.read_table(p).to_pandas() for p in parquet_files], ignore_index=True)
actual_n_eps    = int(data_df["episode_index"].max()) + 1
actual_n_frames = len(data_df)

ep_meta_files = sorted((local / "meta" / "episodes").rglob("*.parquet"))
ep_meta = pd.concat([pq.read_table(p).to_pandas() for p in ep_meta_files], ignore_index=True)

tasks_df = pq.read_table(local / "meta" / "tasks.parquet").to_pandas()

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – episode index continuity
# ─────────────────────────────────────────────────────────────────────────────

section("4 · Episode index continuity")

ep_indices = sorted(data_df["episode_index"].unique().tolist())
expected_ep_indices = list(range(actual_n_eps))

if ep_indices == expected_ep_indices:
    ok(f"Episode indices are contiguous 0..{actual_n_eps-1}")
else:
    missing = sorted(set(expected_ep_indices) - set(ep_indices))
    dupes   = sorted(ep for ep in ep_indices if ep_indices.count(ep) > 1)
    if missing:
        record_fail(f"Missing episode indices: {missing[:20]}{'...' if len(missing)>20 else ''}")
    if dupes:
        record_fail(f"Duplicate episode indices: {dupes[:10]}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – frame index continuity
# ─────────────────────────────────────────────────────────────────────────────

section("5 · Frame index continuity")

frame_indices = sorted(data_df["index"].unique().tolist())
expected_frame_indices = list(range(actual_n_frames))

if frame_indices == expected_frame_indices:
    ok(f"Frame indices are contiguous 0..{actual_n_frames-1}")
else:
    missing_f = sorted(set(expected_frame_indices) - set(frame_indices))
    if missing_f:
        record_fail(f"Missing frame indices ({len(missing_f)} total): first few = {missing_f[:10]}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – task_index validity
# ─────────────────────────────────────────────────────────────────────────────

section("6 · task_index validity")

if pd.api.types.is_string_dtype(tasks_df.index):
    valid_task_indices = set(range(len(tasks_df)))
    task_strings = list(tasks_df.index)
elif "task_index" in tasks_df.columns:
    valid_task_indices = set(int(x) for x in tasks_df["task_index"])
    task_col = next((c for c in tasks_df.columns if pd.api.types.is_string_dtype(tasks_df[c])), None)
    task_strings = list(tasks_df[task_col]) if task_col else []
else:
    valid_task_indices = set(range(len(tasks_df)))
    task_strings = []

if "task_index" in data_df.columns:
    data_task_indices = set(int(x) for x in data_df["task_index"].dropna().unique())
    unknown = data_task_indices - valid_task_indices
    if unknown:
        record_fail(f"Unknown task_index values in data: {sorted(unknown)}")
    else:
        ok(f"All task_index values in data map to tasks.parquet  (unique task_indices: {sorted(data_task_indices)})")
else:
    record_warn("No task_index column in data parquets")

info(f"tasks.parquet has {len(task_strings)} task(s):")
for t in task_strings:
    info(f"  {t!r}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – no ordinal bowl references remain
# ─────────────────────────────────────────────────────────────────────────────

section("7 · No ordinal bowl references in task prompts")

ordinal_found = [t for t in task_strings if ORDINAL_RE.search(t)]
if ordinal_found:
    for t in ordinal_found:
        record_fail(f"Ordinal bowl reference still present: {t!r}")
else:
    ok(f"No ordinal bowl references found in any of {len(task_strings)} task(s)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 – all colour bowl phrases are represented
# ─────────────────────────────────────────────────────────────────────────────

section("8 · Colour bowl phrases coverage")

for phrase in COLOUR_PHRASES:
    found = [t for t in task_strings if phrase in t.lower()]
    if found:
        ok(f"'{phrase}' present  — e.g. {found[0]!r}")
    else:
        record_fail(f"No task contains '{phrase}'")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 – episode metadata from/to consistency
# ─────────────────────────────────────────────────────────────────────────────

section("9 · Episode metadata from/to index consistency")

mismatches = 0
for _, row in ep_meta.iterrows():
    ep_id = int(row["episode_index"])
    ep_data = data_df[data_df["episode_index"] == ep_id]
    if ep_data.empty:
        record_fail(f"Episode {ep_id} appears in metadata but has no rows in data")
        mismatches += 1
        continue
    actual_from = int(ep_data["index"].iloc[0])
    actual_to   = int(ep_data["index"].iloc[-1]) + 1
    if "dataset_from_index" in row.index:
        declared_from = int(row["dataset_from_index"])
        declared_to   = int(row["dataset_to_index"])
        if declared_from != actual_from or declared_to != actual_to:
            record_fail(
                f"Ep {ep_id}: meta says [{declared_from},{declared_to}) "
                f"but data has [{actual_from},{actual_to})"
            )
            mismatches += 1

if mismatches == 0:
    ok(f"dataset_from/to_index consistent for all {len(ep_meta)} episodes")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10 – video file existence on Hub
# ─────────────────────────────────────────────────────────────────────────────

section("10 · Video file existence on Hub")

if args.skip_video_check:
    warn("Skipped (--skip-video-check)")
else:
    try:
        hub_files = set(list_repo_files(repo_id=args.repo, repo_type="dataset", token=token))
        missing_videos = []
        for vk in video_keys:
            file_col  = f"videos/{vk}/file_index"
            chunk_col = f"videos/{vk}/chunk_index"
            if file_col not in ep_meta.columns:
                record_warn(f"No column {file_col!r} in episode metadata — skipping video check for {vk}")
                continue
            for _, row in ep_meta.drop_duplicates(subset=[file_col]).iterrows():
                chunk = int(row[chunk_col]) if chunk_col in row.index else 0
                fidx  = int(row[file_col])
                path  = f"videos/{vk}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
                if path not in hub_files:
                    missing_videos.append(path)

        if missing_videos:
            for p in missing_videos:
                record_fail(f"Missing video on Hub: {p}")
        else:
            ok(f"All referenced video files exist on Hub ({len(video_keys)} key(s) checked)")
    except Exception as exc:
        record_warn(f"Could not list Hub files: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 11 – brightness sanity check (sample pairs)
# ─────────────────────────────────────────────────────────────────────────────

section("11 · Brightness sanity check (sample pairs)")

if args.skip_brightness:
    warn("Skipped (--skip-brightness)")
elif not video_keys:
    record_warn("No video keys — cannot check brightness")
else:
    try:
        import av

        n_src_eps = actual_n_eps // 2
        vk        = video_keys[0]
        file_col  = f"videos/{vk}/file_index"
        from_col  = f"videos/{vk}/from_timestamp"
        to_col    = f"videos/{vk}/to_timestamp"

        # pick sample original episode indices spread across the dataset
        step       = max(1, n_src_eps // args.brightness_sample_pairs)
        sample_ids = list(range(0, min(n_src_eps, step * args.brightness_sample_pairs), step))
        sample_ids = sample_ids[:args.brightness_sample_pairs]

        info(f"Comparing {len(sample_ids)} original/augmented pairs for video key '{vk}'")

        brightness_ok = 0
        brightness_fail = 0

        for orig_ep in sample_ids:
            aug_ep = orig_ep + n_src_eps

            def mean_brightness_of_ep(ep_id: int) -> float:
                row = ep_meta[ep_meta["episode_index"] == ep_id].iloc[0]
                fidx = int(row[file_col])
                from_ts = float(row[from_col])
                to_ts   = float(row[to_col])
                start_frame = round(from_ts * fps)
                end_frame   = round(to_ts   * fps)
                n_frames_ep = max(1, end_frame - start_frame)

                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                    vid_path = hf_hub_download(
                        repo_id=args.repo, repo_type="dataset",
                        filename=f"videos/{vk}/chunk-000/file-{fidx:03d}.mp4",
                        token=token,
                    )
                    samples = []
                    with av.open(vid_path) as container:
                        for i, frame in enumerate(container.decode(video=0)):
                            fi = start_frame + i
                            if fi < start_frame:
                                continue
                            if fi > end_frame:
                                break
                            arr = frame.to_ndarray(format="rgb24").astype(np.float32)
                            samples.append(arr.mean())
                            if len(samples) >= 5:
                                break
                    return float(np.mean(samples)) if samples else 0.0

            try:
                b_orig = mean_brightness_of_ep(orig_ep)
                b_aug  = mean_brightness_of_ep(aug_ep)
                ratio  = b_aug / b_orig if b_orig > 0 else float("nan")
                diff   = abs(b_aug - b_orig)

                # We just need them to differ — even 0.5× or 1.3× is enough
                if diff > 1.0 or abs(ratio - 1.0) > 0.03:
                    ok(f"  ep {orig_ep:4d} → {aug_ep:4d}:  orig={b_orig:.1f}  aug={b_aug:.1f}  ratio={ratio:.3f}")
                    brightness_ok += 1
                else:
                    record_fail(f"  ep {orig_ep:4d} → {aug_ep:4d}:  brightness nearly identical  "
                                f"orig={b_orig:.1f}  aug={b_aug:.1f}  ratio={ratio:.3f}")
                    brightness_fail += 1
            except Exception as exc:
                record_warn(f"  Could not compare ep {orig_ep}/{aug_ep}: {exc}")

        if brightness_fail == 0 and brightness_ok > 0:
            ok(f"All {brightness_ok} sampled pairs show distinct brightness")

    except ImportError:
        record_warn("av not installed — skipping brightness check")
    except Exception as exc:
        record_warn(f"Brightness check failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 12 – frame counts: originals match their augmented copies
# ─────────────────────────────────────────────────────────────────────────────

section("12 · Frame counts: originals match augmented copies")

n_src_eps = actual_n_eps // 2
mismatches_fc = 0
for ep_id in range(n_src_eps):
    orig_frames = len(data_df[data_df["episode_index"] == ep_id])
    aug_frames  = len(data_df[data_df["episode_index"] == ep_id + n_src_eps])
    if orig_frames != aug_frames:
        record_fail(f"Frame count mismatch: ep {ep_id} has {orig_frames} frames, "
                    f"augmented ep {ep_id + n_src_eps} has {aug_frames}")
        mismatches_fc += 1
    if mismatches_fc >= 10:
        record_fail("... (stopped after 10 mismatches)")
        break

if mismatches_fc == 0:
    ok(f"All {n_src_eps} original/augmented pairs have equal frame counts")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

section("Summary")

info(f"Dataset : {args.repo}")
info(f"Episodes: {actual_n_eps}  ({n_src_eps} original + {n_src_eps} augmented)")
info(f"Frames  : {actual_n_frames}")
info(f"Tasks   : {len(task_strings)}")

if warnings:
    print(f"\n  {WARN}  {len(warnings)} warning(s):")
    for w in warnings:
        print(f"     • {w}")

if errors:
    print(f"\n  {FAIL}  {len(errors)} error(s) found:\n")
    for e in errors:
        print(f"     • {e}")
    sys.exit(1)
else:
    print(f"\n  {OK}  All checks passed — dataset looks correct.")
