"""LeRobot v2 format I/O — parquet + MP4 video."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.io as tvio


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_meta(dataset_path: Path) -> Dict:
    meta_dir = dataset_path / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)

    episodes: List[Dict] = []
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    tasks: List[Dict] = []
    with open(meta_dir / "tasks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    return {"info": info, "episodes": episodes, "tasks": tasks}


def _episode_chunk(episode_idx: int, chunks_size: int) -> int:
    return episode_idx // chunks_size


def _parquet_path(root: Path, episode_idx: int, chunks_size: int) -> Path:
    chunk = _episode_chunk(episode_idx, chunks_size)
    return root / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"


def _video_path(root: Path, episode_idx: int, video_key: str, chunks_size: int) -> Path:
    chunk = _episode_chunk(episode_idx, chunks_size)
    return root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_idx:06d}.mp4"


def video_keys(meta: Dict) -> List[str]:
    return [k for k, v in meta["info"]["features"].items() if v.get("dtype") == "video"]


def load_episode(dataset_path: Path, episode_idx: int, meta: Dict) -> Dict:
    """Return dict with 'df', 'videos' (THWC uint8 tensors), 'instruction'."""
    chunks_size = meta["info"]["chunks_size"]

    parquet = _parquet_path(dataset_path, episode_idx, chunks_size)
    df = pd.read_parquet(parquet)

    task_idx = int(df["task_index"].iloc[0])
    instruction = meta["tasks"][task_idx]["task"]

    videos: Dict[str, torch.Tensor] = {}
    for vk in video_keys(meta):
        vpath = _video_path(dataset_path, episode_idx, vk, chunks_size)
        frames, _, _ = tvio.read_video(str(vpath), pts_unit="sec", output_format="THWC")
        videos[vk] = frames  # uint8, shape (T, H, W, C)

    return {"df": df, "videos": videos, "instruction": instruction, "task_idx": task_idx}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_episode(
    output_path: Path,
    episode_idx: int,
    global_frame_start: int,
    episode_data: Dict,
    new_task_idx: int,
    fps: float,
    chunks_size: int,
) -> int:
    """Write parquet + videos for one episode. Returns frame count."""
    df = episode_data["df"].copy()
    n_frames = len(df)

    df["episode_index"] = episode_idx
    df["task_index"] = new_task_idx
    df["index"] = np.arange(global_frame_start, global_frame_start + n_frames)

    parquet = _parquet_path(output_path, episode_idx, chunks_size)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)

    for vk, frames in episode_data["videos"].items():
        vpath = _video_path(output_path, episode_idx, vk, chunks_size)
        vpath.parent.mkdir(parents=True, exist_ok=True)
        tvio.write_video(str(vpath), frames, fps=fps)

    return n_frames


def save_meta(
    output_path: Path,
    episodes_meta: List[Dict],
    tasks: List[Dict],
    original_info: Dict,
    total_frames: int,
    aug_description: str = "",
) -> None:
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = len(episodes_meta)
    n_video_streams = len([k for k, v in original_info["features"].items() if v.get("dtype") == "video"])

    info = {
        **original_info,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": n_video_streams * n_episodes,
        "total_chunks": max(1, (n_episodes - 1) // original_info["chunks_size"] + 1),
        "splits": {"train": f"0:{n_episodes}"},
    }
    if aug_description:
        info["augmentation"] = aug_description

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


def copy_stats(src: Path, dst: Path) -> None:
    """Copy normalization stats file if it exists."""
    for name in ("stats.json", "stats.safetensors"):
        src_file = src / "meta" / name
        if src_file.exists():
            shutil.copy2(src_file, dst / "meta" / name)
