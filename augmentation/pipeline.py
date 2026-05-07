"""Augmentation pipeline orchestration."""

from __future__ import annotations

import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from augmentation.config import AugmentationConfig
from augmentation.dataset_io import (
    copy_stats,
    load_episode,
    load_meta,
    save_episode,
    save_meta,
    video_keys,
)
from augmentation.text_augment import generate_variants
from augmentation.validate import check_episode
from augmentation.visual_augment import augment_video, sample_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task registry — maps instruction strings to task indices
# ---------------------------------------------------------------------------

class _TaskRegistry:
    def __init__(self) -> None:
        self._index: Dict[str, int] = {}
        self._tasks: List[Dict] = []

    def get_or_add(self, instruction: str) -> int:
        if instruction not in self._index:
            idx = len(self._tasks)
            self._index[instruction] = idx
            self._tasks.append({"task_index": idx, "task": instruction})
        return self._index[instruction]

    @property
    def tasks(self) -> List[Dict]:
        return self._tasks


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_path: Path,
    cfg: AugmentationConfig,
) -> None:
    rng = random.Random(cfg.seed)
    meta = load_meta(input_path)
    fps = float(meta["info"]["fps"])
    n_source_episodes = len(meta["episodes"])

    logger.info(f"Source dataset: {n_source_episodes} episodes, fps={fps}")

    task_registry = _TaskRegistry()
    output_episodes_meta: List[Dict] = []
    new_episode_idx = 0
    total_frames = 0
    skipped = 0

    for src_idx in range(n_source_episodes):
        logger.info(f"Processing episode {src_idx + 1}/{n_source_episodes}")
        src_data = load_episode(input_path, src_idx, meta)
        src_instruction = src_data["instruction"]
        n_frames = len(src_data["df"])

        # --- Optionally keep original episode unchanged ---
        if cfg.include_original:
            task_idx = task_registry.get_or_add(src_instruction)
            frames_written = save_episode(
                output_path=output_path,
                episode_idx=new_episode_idx,
                global_frame_start=total_frames,
                episode_data=src_data,
                new_task_idx=task_idx,
                fps=fps,
                chunks_size=cfg.chunks_size,
            )
            output_episodes_meta.append(
                _episode_meta(new_episode_idx, n_frames, [task_idx])
            )
            total_frames += frames_written
            new_episode_idx += 1

        # --- Text variants × visual variants ---
        text_variants = generate_variants(
            src_instruction, cfg.n_text_variants, rng, include_original=False
        )

        for text_var in text_variants:
            task_idx = task_registry.get_or_add(text_var)

            for _ in range(cfg.n_visual_variants):
                vis_params = sample_params(cfg, rng)

                aug_videos = {
                    vk: augment_video(frames, vis_params)
                    for vk, frames in src_data["videos"].items()
                }
                aug_data = {**src_data, "videos": aug_videos, "instruction": text_var}

                ok, warnings = check_episode(src_data, aug_data)
                if not ok:
                    logger.warning(
                        f"  Episode {src_idx} variant skipped due to validation errors: "
                        + "; ".join(warnings)
                    )
                    skipped += 1
                    continue
                if warnings:
                    for w in warnings:
                        logger.warning(f"  Validation warning: {w}")

                frames_written = save_episode(
                    output_path=output_path,
                    episode_idx=new_episode_idx,
                    global_frame_start=total_frames,
                    episode_data=aug_data,
                    new_task_idx=task_idx,
                    fps=fps,
                    chunks_size=cfg.chunks_size,
                )
                output_episodes_meta.append(
                    _episode_meta(new_episode_idx, n_frames, [task_idx])
                )
                total_frames += frames_written
                new_episode_idx += 1

    logger.info(
        f"Done. {new_episode_idx} episodes written "
        f"({skipped} variants skipped), {total_frames} total frames."
    )

    aug_description = (
        f"Augmented from {n_source_episodes} source episodes. "
        f"Text variants: {cfg.n_text_variants}, visual variants: {cfg.n_visual_variants}. "
        f"Seed: {cfg.seed}."
    )
    save_meta(
        output_path,
        output_episodes_meta,
        task_registry.tasks,
        meta["info"],
        total_frames,
        aug_description,
    )
    copy_stats(input_path, output_path)

    if cfg.push_to_hub:
        _push_to_hub(output_path, cfg)


def _episode_meta(episode_idx: int, length: int, task_indices: List[int]) -> Dict:
    return {"episode_index": episode_idx, "tasks": task_indices, "length": length}


def _push_to_hub(output_path: Path, cfg: AugmentationConfig) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    if not cfg.hub_repo_id:
        logger.error("hub_repo_id is not set in config.")
        return

    api = HfApi()
    api.create_repo(
        repo_id=cfg.hub_repo_id,
        repo_type="dataset",
        private=cfg.hub_private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=cfg.hub_repo_id,
        repo_type="dataset",
    )
    logger.info(f"Dataset pushed to https://huggingface.co/datasets/{cfg.hub_repo_id}")
