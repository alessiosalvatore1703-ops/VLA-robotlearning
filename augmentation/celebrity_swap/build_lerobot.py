"""
Assemble the augmented episodes into a LeRobot v3 format dataset.

LeRobot v3 layout (matching task3-TOY-clean):
  data/chunk-{chunk:03d}/file-{file:03d}.parquet   ← per-episode action/state
  videos/observation.images.front/chunk-{chunk:03d}/file-{file:03d}.mp4
  meta/info.json
  meta/tasks.parquet
  meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet

The output dataset contains:
  • All *original* source episodes (unchanged)
  • All augmented episodes (same actions/states, new video + new task string)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CHUNKS_SIZE = 1000
VIDEO_KEY   = "observation.images.front"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    src_path: Path,
    dst_path: Path,
    aug_results: list[dict],
    original_info: dict,
    include_originals: bool = True,
) -> None:
    """
    Write a complete LeRobot v3 dataset to *dst_path*.

    *aug_results* is a list of dicts, one per augmented episode:
        {
          "src_episode_idx": int,
          "new_task_str": str,
          "new_video_path": Path,   ← tmp video already written by episode_worker
        }
    """
    dst_path.mkdir(parents=True, exist_ok=True)

    # ── Task registry ──────────────────────────────────────────────────────────
    task_index: dict[str, int] = {}

    def get_task_idx(task: str) -> int:
        if task not in task_index:
            task_index[task] = len(task_index)
        return task_index[task]

    # ── Load source meta ───────────────────────────────────────────────────────
    src_tasks    = _load_tasks(src_path)
    src_episodes = _load_episodes(src_path)

    # Pre-populate task registry with source tasks (preserves original indices)
    for task_str in src_tasks.values():
        get_task_idx(task_str)

    # ── Process all episodes in order: originals first, then augmentations ─────
    new_episode_idx  = 0
    global_frame_idx = 0
    episodes_meta: list[dict] = []

    # originals
    if include_originals:
        for src_ep_idx, ep_info in enumerate(src_episodes):
            src_task_str = ep_info["task_str"]
            task_idx     = get_task_idx(src_task_str)
            n_frames     = ep_info["length"]

            _copy_parquet(src_path, src_ep_idx, dst_path, new_episode_idx,
                          task_idx, global_frame_idx)
            _copy_video(src_path, src_ep_idx, dst_path, new_episode_idx)

            episodes_meta.append(_ep_meta(new_episode_idx, n_frames, src_task_str))
            global_frame_idx += n_frames
            new_episode_idx  += 1

    # augmentations
    for rec in aug_results:
        src_ep_idx   = rec["src_episode_idx"]
        new_task_str = rec["new_task_str"]
        new_vid_path = Path(rec["new_video_path"])
        task_idx     = get_task_idx(new_task_str)
        n_frames     = src_episodes[src_ep_idx]["length"]

        _copy_parquet(src_path, src_ep_idx, dst_path, new_episode_idx,
                      task_idx, global_frame_idx)
        _place_video(new_vid_path, dst_path, new_episode_idx)

        episodes_meta.append(_ep_meta(new_episode_idx, n_frames, new_task_str))
        global_frame_idx += n_frames
        new_episode_idx  += 1

    # ── Write metadata ─────────────────────────────────────────────────────────
    _write_tasks_parquet(dst_path, task_index)
    _write_episodes_parquet(dst_path, episodes_meta)
    _write_info_json(dst_path, original_info, new_episode_idx,
                     global_frame_idx, len(task_index))

    # Copy stats if present (action normalization etc.)
    _copy_stats(src_path, dst_path)

    print(f"Dataset ready: {new_episode_idx} episodes, "
          f"{global_frame_idx} frames, {len(task_index)} tasks  →  {dst_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ep_chunk(ep_idx: int) -> int:
    return ep_idx // CHUNKS_SIZE


def _parquet_path(root: Path, ep_idx: int) -> Path:
    return root / f"data/chunk-{_ep_chunk(ep_idx):03d}/file-{ep_idx:03d}.parquet"


def _video_path(root: Path, ep_idx: int) -> Path:
    return root / f"videos/{VIDEO_KEY}/chunk-{_ep_chunk(ep_idx):03d}/file-{ep_idx:03d}.mp4"


def _load_tasks(src: Path) -> dict[int, str]:
    """Return {task_index_int: task_str} from meta/tasks.parquet.

    Actual v3 format: the task string is the row INDEX; 'task_index' is the
    integer column.  Example:
        index (str)                           task_index (int)
        "Place the coke on Taylor Swift."     0
        "Place the coke on Barack Obama."     1
    """
    p = src / "meta" / "tasks.parquet"
    if not p.exists():
        raise FileNotFoundError(f"tasks.parquet not found at {p}")
    df = pd.read_parquet(p)
    if "task_index" in df.columns:
        # Standard v3: task string is the DataFrame index
        return {int(row["task_index"]): str(task_str)
                for task_str, row in df.iterrows()}
    # Fallback for datasets that store task string in a 'task' column
    if "task" in df.columns:
        return {i: str(row["task"]) for i, row in df.iterrows()}
    raise ValueError(f"Cannot parse tasks.parquet columns: {df.columns.tolist()}")


def _load_episodes(src: Path) -> list[dict]:
    """Return list of {episode_index, task_str, length}, sorted by episode_index.

    The 'tasks' column in the v3 episodes parquet is a numpy array of task
    strings (not integers).  We extract the first element.
    """
    ep_dir = src / "meta" / "episodes" / "chunk-000"
    if not ep_dir.exists():
        raise FileNotFoundError(f"episodes dir not found: {ep_dir}")
    dfs = [pd.read_parquet(p) for p in sorted(ep_dir.glob("*.parquet"))]
    df  = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df  = df.sort_values("episode_index").reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        raw_tasks = row.get("tasks", [])
        if isinstance(raw_tasks, (list, np.ndarray)) and len(raw_tasks) > 0:
            task_str = str(raw_tasks[0])
        else:
            task_str = str(raw_tasks)
        records.append({
            "episode_index": int(row["episode_index"]),
            "task_str":      task_str,
            "length":        int(row["length"]),
        })
    return records


def _copy_parquet(
    src: Path,
    src_ep_idx: int,
    dst: Path,
    new_ep_idx: int,
    task_idx: int,
    global_frame_start: int,
) -> None:
    src_pq = _parquet_path(src, src_ep_idx)
    if not src_pq.exists():
        raise FileNotFoundError(f"Source parquet not found: {src_pq}")

    df = pd.read_parquet(src_pq).copy()
    n  = len(df)
    df["episode_index"] = new_ep_idx
    df["task_index"]    = task_idx
    df["index"]         = np.arange(global_frame_start, global_frame_start + n)

    dst_pq = _parquet_path(dst, new_ep_idx)
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_pq, index=False)


def _copy_video(src: Path, src_ep_idx: int, dst: Path, new_ep_idx: int) -> None:
    src_v = _video_path(src, src_ep_idx)
    if not src_v.exists():
        raise FileNotFoundError(f"Source video not found: {src_v}")
    dst_v = _video_path(dst, new_ep_idx)
    dst_v.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_v, dst_v)


def _place_video(tmp_vid: Path, dst: Path, new_ep_idx: int) -> None:
    dst_v = _video_path(dst, new_ep_idx)
    dst_v.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_vid), str(dst_v))


def _ep_meta(ep_idx: int, length: int, task_str: str) -> dict:
    # Store task as a list of strings (matches v3 episodes parquet 'tasks' column)
    return {"episode_index": ep_idx, "tasks": [task_str], "length": length}


def _write_tasks_parquet(dst: Path, task_index: dict[str, int]) -> None:
    # Match v3 format: task string is the DataFrame index, 'task_index' is the column
    df = pd.DataFrame(
        {"task_index": list(task_index.values())},
        index=pd.Index(list(task_index.keys()), name=None),
    ).sort_values("task_index")
    out = dst / "meta" / "tasks.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)


def _write_episodes_parquet(dst: Path, metas: list[dict]) -> None:
    df  = pd.DataFrame(metas)
    out = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def _write_info_json(
    dst: Path,
    original_info: dict,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> None:
    n_chunks = max(1, (total_episodes - 1) // CHUNKS_SIZE + 1)
    info: dict[str, Any] = {
        **original_info,
        "total_episodes": total_episodes,
        "total_frames":   total_frames,
        "total_tasks":    total_tasks,
        "total_videos":   total_episodes,
        "total_chunks":   n_chunks,
        "splits":         {"train": f"0:{total_episodes}"},
        "augmentation": (
            "celebrity_swap: portraits replaced with synthetic A5 textures from "
            "ielminawi/celeb30.  Actions/states unchanged from source."
        ),
    }
    out = dst / "meta" / "info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=2)


def _copy_stats(src: Path, dst: Path) -> None:
    for name in ("stats.json", "stats.safetensors"):
        p = src / "meta" / name
        if p.exists():
            shutil.copy2(p, dst / "meta" / name)
