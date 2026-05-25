#!/usr/bin/env python3
"""Recompute meta/stats.json and reformat a LeRobot v3 dataset so it is
training-ready for SmolVLA, then optionally push the corrected copy to the Hub.

A "non-training-ready" dataset here is one that is structurally a LeRobot v3
dataset (data parquets, videos, meta/info.json, meta/episodes/, meta/tasks.parquet)
but is missing meta/stats.json and/or has an inconsistent meta/info.json
(wrong `splits`, stale `total_*`). SmolVLA training loads normalization stats
from meta/stats.json via `LeRobotDatasetMetadata`; without that file
`meta.stats` is None and training crashes or trains unnormalized. Likewise, if
`splits["train"]` does not cover every episode, half the data is silently
invisible to the dataloader.

What this script does:
  1. Copies --source into --output (videos included); --source may be a local
     directory or a HuggingFace dataset repo id (downloaded on the fly).
  2. Recomputes dataset-level statistics for every non-string feature:
       - vector/scalar features (action, observation.state, timestamp, ...)
         exactly from the data parquets,
       - video features by sampling decoded frames from the encoded videos.
  3. Writes meta/stats.json in the canonical LeRobot v3 layout (using lerobot's
     own helpers, so the format is guaranteed compatible).
  4. Rewrites meta/info.json `total_*` counts and `splits` so the whole dataset
     sits in the train split.
  5. Optionally uploads --output to a HuggingFace dataset repo.

Usage:
    # local -> local
    python datasets/utils/make_training_ready.py \
        --source dataset --output dataset_trainready

    # local -> local, then push the corrected dataset to the Hub
    python datasets/utils/make_training_ready.py \
        --source dataset --output dataset_trainready \
        --repo-id ETHrobotlearning/my-dataset

    # HF repo -> local -> push back to the same repo name
    python datasets/utils/make_training_ready.py \
        --source ETHrobotlearning/my-dataset --output /tmp/fixed --push

Must run in an environment with `lerobot` installed (it imports lerobot's stats
helpers and SmolVLA's normalization mapping).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import av
except ImportError:  # pragma: no cover
    av = None

try:
    from lerobot.datasets.compute_stats import (
        DEFAULT_QUANTILES,
        auto_downsample_height_width,
        get_feature_stats,
    )
    from lerobot.datasets.io_utils import load_stats, write_stats
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"Could not import lerobot ({exc}). Run this script inside the environment "
        "where lerobot is installed (e.g. `conda activate lerobot`)."
    )


# --------------------------------------------------------------------------- #
# Source loading
# --------------------------------------------------------------------------- #
def resolve_source(source: str, token: str | None) -> tuple[Path, str | None]:
    """Return (local_dir, repo_id). If `source` is not a local directory it is
    treated as a HuggingFace dataset repo id and downloaded."""
    p = Path(source)
    if p.is_dir():
        return p, None
    from huggingface_hub import snapshot_download

    print(f"  '{source}' is not a local directory -> downloading from the Hub...")
    local = Path(snapshot_download(repo_id=source, repo_type="dataset", token=token))
    return local, source


# --------------------------------------------------------------------------- #
# Stats computation
# --------------------------------------------------------------------------- #
def load_data_frame(root: Path) -> pd.DataFrame:
    """Concatenate every data parquet into a single DataFrame."""
    parquets = sorted((root / "data").rglob("*.parquet"))
    if not parquets:
        sys.exit(f"No data parquets found under {root / 'data'}")
    return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)


def numeric_feature_stats(df: pd.DataFrame, key: str) -> dict:
    """Exact dataset-level stats for a numeric (vector or scalar) feature."""
    col = df[key]
    sample = col.iloc[0]
    if isinstance(sample, (list, np.ndarray)):  # vector feature, e.g. (N, 6)
        arr = np.stack(col.to_numpy()).astype(np.float32)
        keepdims = False
    else:  # scalar feature, e.g. (N,)
        arr = col.to_numpy().astype(np.float32)
        keepdims = True
    return get_feature_stats(arr, axis=0, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES)


def sample_video_frames(video_files: list[Path], total_frames: int, target: int) -> np.ndarray:
    """Decode `video_files` and return a uniformly strided, downsampled sample of
    frames as a (N, 3, H, W) uint8 array — mirroring lerobot's image sampling."""
    if av is None:
        sys.exit("PyAV ('av') is required to compute video stats. `pip install av`.")
    stride = max(1, total_frames // max(1, target))
    frames: list[np.ndarray] = []
    for vf in video_files:
        with av.open(str(vf)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i % stride:
                    continue
                rgb = frame.to_ndarray(format="rgb24")          # (H, W, 3)
                chw = np.transpose(rgb, (2, 0, 1))               # (3, H, W)
                frames.append(auto_downsample_height_width(chw).astype(np.uint8))
    if not frames:
        raise RuntimeError("Decoded zero frames from the supplied videos.")
    return np.stack(frames)


def image_feature_stats(frames: np.ndarray) -> dict:
    """Per-channel stats for sampled video frames, normalized to [0, 1] with
    shape (3, 1, 1) — matching lerobot.datasets.compute_stats.compute_episode_stats.

    Frames are cast to float32 first: lerobot's RunningQuantileStats squares the
    batch (`batch**2`), which silently overflows for uint8 input and yields a
    bogus (typically zero) standard deviation."""
    frames = frames.astype(np.float32)
    stats = get_feature_stats(frames, axis=(0, 2, 3), keepdims=True, quantile_list=DEFAULT_QUANTILES)
    return {
        k: (v if k == "count" else np.squeeze(v / 255.0, axis=0))
        for k, v in stats.items()
    }


def compute_stats(root: Path, info: dict, image_samples: int, skip_image_stats: bool) -> dict:
    """Compute the full dataset-level stats dict for every non-string feature."""
    features = info["features"]
    df = load_data_frame(root)
    total_frames = len(df)

    stats: dict[str, dict] = {}
    for key, spec in features.items():
        dtype = spec.get("dtype")
        if dtype == "string":
            continue
        if dtype in ("image", "video"):
            if skip_image_stats:
                print(f"  {key}: skipped (--skip-image-stats)")
                continue
            video_files = sorted((root / "videos" / key).rglob("*.mp4"))
            if not video_files:
                print(f"  {key}: WARNING no video files found, skipping")
                continue
            frames = sample_video_frames(video_files, total_frames, image_samples)
            stats[key] = image_feature_stats(frames)
            print(f"  {key}: {len(frames)} frames sampled from {len(video_files)} video file(s)")
        else:
            if key not in df.columns:
                print(f"  {key}: WARNING absent from data parquets, skipping")
                continue
            stats[key] = numeric_feature_stats(df, key)
            print(f"  {key}: stats over {total_frames} frames")
    return stats


# --------------------------------------------------------------------------- #
# info.json reformatting
# --------------------------------------------------------------------------- #
def reformat_info(root: Path, info: dict) -> dict:
    """Recompute `total_*` counts and fix `splits` so the whole dataset is
    training-visible. Returns the corrected info dict (also written to disk)."""
    new = dict(info)

    df = load_data_frame(root)
    total_frames = len(df)
    total_episodes = int(df["episode_index"].nunique())

    ep_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if ep_meta_files:
        ep_meta = pd.concat([pd.read_parquet(p) for p in ep_meta_files], ignore_index=True)
        total_episodes = max(total_episodes, len(ep_meta))

    tasks_path = root / "meta" / "tasks.parquet"
    total_tasks = len(pd.read_parquet(tasks_path)) if tasks_path.exists() else new.get("total_tasks", 0)

    total_videos = len(list((root / "videos").rglob("*.mp4"))) if (root / "videos").is_dir() else 0
    data_chunks = {p.name for p in (root / "data").glob("chunk-*") if p.is_dir()}

    changes = []
    for field, value in (
        ("total_frames", total_frames),
        ("total_episodes", total_episodes),
        ("total_tasks", total_tasks),
        ("total_videos", total_videos),
        ("total_chunks", len(data_chunks)),
    ):
        if new.get(field) != value:
            changes.append(f"{field}: {new.get(field)} -> {value}")
        new[field] = value

    new_splits = {"train": f"0:{total_episodes}"}
    if new.get("splits") != new_splits:
        changes.append(f"splits: {new.get('splits')} -> {new_splits}")
    new["splits"] = new_splits

    with open(root / "meta" / "info.json", "w") as f:
        json.dump(new, f, indent=2)

    if changes:
        print("  info.json updated:")
        for c in changes:
            print(f"    {c}")
    else:
        print("  info.json already consistent")
    return new


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(root: Path, info: dict) -> None:
    """Sanity-check that the corrected dataset is loadable as SmolVLA expects."""
    from lerobot.configs import NormalizationMode
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    stats = load_stats(root)
    if stats is None:
        sys.exit("Verification failed: meta/stats.json could not be loaded.")

    # SmolVLA normalizes STATE and ACTION with MEAN_STD; those keys must exist
    # with finite, non-degenerate std so normalization does not divide by zero.
    norm = SmolVLAConfig().normalization_mapping
    required = {"observation.state": norm.get("STATE"), "action": norm.get("ACTION")}
    for key, mode in required.items():
        if mode != NormalizationMode.MEAN_STD:
            continue
        if key not in stats:
            sys.exit(f"Verification failed: '{key}' missing from meta/stats.json.")
        std = np.asarray(stats[key]["std"], dtype=float)
        if not np.all(np.isfinite(std)):
            sys.exit(f"Verification failed: '{key}' std contains non-finite values.")
        if np.any(std == 0):
            print(f"  WARNING: '{key}' has zero std in some dimension(s) — "
                  "a constant channel; SmolVLA will not normalize it.")

    n_eps = info["total_episodes"]
    if info["splits"].get("train") != f"0:{n_eps}":
        sys.exit("Verification failed: train split does not cover every episode.")
    print(f"  OK — meta/stats.json has {len(stats)} feature(s); "
          f"train split covers all {n_eps} episodes.")


# --------------------------------------------------------------------------- #
# Push
# --------------------------------------------------------------------------- #
def push(root: Path, repo_id: str, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"  repo '{repo_id}' exists — uploading corrected files...")
    except Exception:
        print(f"  creating dataset repo '{repo_id}'...")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)

    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Recompute meta/stats.json and fix info.json for training",
    )
    print(f"  pushed -> https://huggingface.co/datasets/{repo_id}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute stats and reformat a LeRobot v3 dataset for SmolVLA training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Source dataset: a local directory or a HuggingFace dataset repo id.")
    parser.add_argument("--output", required=True,
                        help="Output directory for the corrected, training-ready dataset.")
    parser.add_argument("--repo-id", default=None,
                        help="HuggingFace dataset repo to push the corrected dataset to. "
                             "If omitted and --push is set, the source repo id is reused.")
    parser.add_argument("--push", action="store_true",
                        help="Push the corrected dataset to the Hub (implied when --repo-id is given).")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (defaults to the HF_TOKEN env var).")
    parser.add_argument("--image-samples", type=int, default=1500,
                        help="Approximate number of video frames to sample for image stats.")
    parser.add_argument("--skip-image-stats", action="store_true",
                        help="Do not compute video stats. SmolVLA uses IDENTITY normalization "
                             "for images, so they are not strictly required for SmolVLA training.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite --output if it already exists.")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    output = Path(args.output)

    # ---- 1. resolve + copy source ---------------------------------------- #
    print("[1/5] Resolving and copying source dataset...")
    src_dir, src_repo_id = resolve_source(args.source, token)
    if not (src_dir / "meta" / "info.json").exists():
        sys.exit(f"{src_dir} is not a LeRobot dataset (no meta/info.json).")

    if output.exists():
        if not args.overwrite:
            sys.exit(f"--output '{output}' already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    if src_dir.resolve() == output.resolve():
        sys.exit("--source and --output must differ.")
    shutil.copytree(src_dir, output)
    print(f"  copied -> {output}")

    with open(output / "meta" / "info.json") as f:
        info = json.load(f)
    print(f"  codebase_version={info.get('codebase_version')} fps={info.get('fps')}")

    # ---- 2. recompute stats.json ----------------------------------------- #
    print("\n[2/5] Recomputing dataset statistics...")
    stats = compute_stats(output, info, args.image_samples, args.skip_image_stats)
    write_stats(stats, output)
    print(f"  wrote {output / 'meta' / 'stats.json'}")

    # ---- 3. reformat info.json ------------------------------------------- #
    print("\n[3/5] Reformatting meta/info.json...")
    info = reformat_info(output, info)

    # ---- 4. verify ------------------------------------------------------- #
    print("\n[4/5] Verifying training-readiness...")
    verify(output, info)

    # ---- 5. push --------------------------------------------------------- #
    print("\n[5/5] Pushing to the Hub...")
    repo_id = args.repo_id or (src_repo_id if args.push else None)
    if repo_id:
        push(output, repo_id, token)
    else:
        print("  skipped (no --repo-id and --push not set).")

    print(f"\nDone. Training-ready dataset at: {output}")


if __name__ == "__main__":
    main()
