"""
Per-episode celebrity-swap worker.

For one source episode:
  1. Load video frames (torchvision).
  2. Detect 3 portrait quads on a clean frame.
  3. Identify which quad = which of {Taylor Swift, Barack Obama, Yann LeCun}.
  4. Sample N_AUG replacement celeb triples.
  5. For each replacement triple:
       - Load new textures.
       - Composite every frame with occlusion-aware blending.
       - Write output video.
  6. Return list of (new_video_path, new_task_str) pairs.

Designed to be called from a multiprocessing pool — no shared GPU state.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.io as tvio

from augmentation.celebrity_swap.compose_frame import (
    composite_portraits,
    load_texture,
)
from augmentation.celebrity_swap.detect_portraits import (
    detect_portrait_quads,
    find_clean_frame,
)
from augmentation.celebrity_swap.identify_portraits import identify_from_task
from augmentation.celebrity_swap.prepare_celeb_bank import CELEB_NAMES, _name_to_slug

logger = logging.getLogger(__name__)

IN_DIST_CELEBS = ["Taylor Swift", "Barack Obama", "Yann LeCun"]
# Remove the 3 in-dist celebs from the OOD pool so swaps are always novel
OOD_CELEBS = [n for n in CELEB_NAMES if n not in IN_DIST_CELEBS]


@dataclass
class AugResult:
    new_task_str: str
    new_video_path: Path
    # Mapping original celeb → replacement celeb (for all 3 positions)
    swap_map: dict[str, str]


def _sample_replacement_triple(
    quad_to_celeb: dict[int, str],
    target_celeb: str,
    target_new_celeb: str,
    all_celebs: list[str],
    rng: random.Random,
) -> dict[str, str]:
    """
    Build {orig_celeb → new_celeb} for all 3 positions.
    The target position uses *target_new_celeb*.
    The other two get randomly sampled distinct celebs.
    """
    used = {target_new_celeb}
    mapping: dict[str, str] = {}

    for quad_idx, orig in quad_to_celeb.items():
        if orig == target_celeb:
            mapping[orig] = target_new_celeb
        else:
            pool = [c for c in all_celebs if c not in used]
            if not pool:
                pool = [c for c in all_celebs if c != target_new_celeb]
            chosen = rng.choice(pool)
            mapping[orig] = chosen
            used.add(chosen)

    return mapping


def process_episode(
    src_video_path: Path,
    dst_dir: Path,
    src_task_str: str,
    bank_dir: Path,
    n_aug: int,
    rng: random.Random,
    target_celeb_queue: Optional[list[str]] = None,
    fps: float = 10.0,
) -> list[AugResult]:
    """
    Process one source episode and write *n_aug* augmented versions.

    *target_celeb_queue*: if provided, pop the first *n_aug* entries as the
    target replacement celebs (for balanced sampling).  Otherwise sample randomly.

    Returns a list of AugResult, one per augmented episode.
    """
    # ── Load frames ────────────────────────────────────────────────────────────
    try:
        frames_thwc, _, _ = tvio.read_video(str(src_video_path), pts_unit="sec",
                                            output_format="THWC")
    except Exception as exc:
        logger.error("Could not read %s: %s", src_video_path, exc)
        return []

    frames_np = frames_thwc.numpy()  # (T, H, W, 3) RGB uint8
    T = frames_np.shape[0]
    if T == 0:
        logger.warning("Empty video: %s", src_video_path)
        return []

    # Convert to BGR for OpenCV processing
    frames_bgr = frames_np[..., ::-1].copy()   # (T, H, W, 3)

    # ── Detect quads on frame 0, then find the cleanest reference frame ────────
    quads = detect_portrait_quads(frames_bgr[0])
    if len(quads) < 3:
        logger.warning("%s: only %d quads detected (expected 3); skipping.",
                       src_video_path.name, len(quads))
        return []

    clean_bgr = find_clean_frame(frames_bgr, quads)

    # Re-detect quads on the clean frame (may be slightly better)
    quads_clean = detect_portrait_quads(clean_bgr)
    if len(quads_clean) == 3:
        quads = quads_clean

    # ── Identify which quad is which celebrity ─────────────────────────────────
    quad_to_celeb = identify_from_task(clean_bgr, quads, src_task_str, IN_DIST_CELEBS)
    logger.debug("%s quad mapping: %s", src_video_path.name, quad_to_celeb)

    target_celeb = _extract_target_celeb(src_task_str)
    if target_celeb is None:
        logger.warning("Could not extract target celeb from: %s", src_task_str)
        target_celeb = IN_DIST_CELEBS[0]

    # ── Warm up KNN background subtractor on the first few frames ─────────────
    bg_sub = cv2.createBackgroundSubtractorKNN(
        history=20, dist2Threshold=400, detectShadows=False
    )
    warmup = min(8, T)
    for t in range(warmup):
        bg_sub.apply(frames_bgr[t], learningRate=0.05)

    # ── Generate n_aug augmented versions ─────────────────────────────────────
    results: list[AugResult] = []
    dst_dir.mkdir(parents=True, exist_ok=True)

    for aug_i in range(n_aug):
        # Pick target replacement
        if target_celeb_queue and len(target_celeb_queue) > 0:
            new_target = target_celeb_queue.pop(0)
        else:
            new_target = rng.choice(OOD_CELEBS)

        swap_map = _sample_replacement_triple(
            quad_to_celeb, target_celeb, new_target, OOD_CELEBS, rng
        )
        new_task_str = f"Place the coke on {new_target}."

        # Load textures in the same left-to-right order as quads
        textures: list[np.ndarray] = []
        for q_idx in range(len(quads)):
            orig_celeb = quad_to_celeb.get(q_idx, IN_DIST_CELEBS[q_idx % 3])
            new_celeb  = swap_map.get(orig_celeb, rng.choice(OOD_CELEBS))
            slug       = _name_to_slug(new_celeb)
            try:
                tex = load_texture(bank_dir, slug)
            except FileNotFoundError:
                logger.warning("Texture missing for %s; using first available.", new_celeb)
                tex = _fallback_texture(bank_dir)
            textures.append(tex)

        # ── Composite every frame ──────────────────────────────────────────────
        aug_frames: list[np.ndarray] = []
        for t in range(T):
            fgmask = bg_sub.apply(frames_bgr[t], learningRate=0.001)
            composited = composite_portraits(
                frames_bgr[t], clean_bgr, quads, textures, bg_fgmask=fgmask
            )
            aug_frames.append(composited)

        # Convert back to RGB for torchvision writer
        aug_rgb = np.stack([f[..., ::-1] for f in aug_frames], axis=0)  # (T, H, W, 3)
        aug_tensor = torch.from_numpy(aug_rgb.copy())

        # ── Write video ────────────────────────────────────────────────────────
        out_name = f"{src_video_path.stem}_swap{aug_i:02d}.mp4"
        out_path = dst_dir / out_name
        tvio.write_video(str(out_path), aug_tensor, fps=fps, video_codec="libx264",
                         options={"crf": "23", "preset": "fast"})

        results.append(AugResult(
            new_task_str=new_task_str,
            new_video_path=out_path,
            swap_map=swap_map,
        ))
        logger.debug("  [%s aug%02d] %s → %s", src_video_path.stem, aug_i,
                     target_celeb, new_target)

    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_target_celeb(task_str: str) -> Optional[str]:
    for name in IN_DIST_CELEBS:
        if name.lower() in task_str.lower():
            return name
    return None


def _fallback_texture(bank_dir: Path) -> np.ndarray:
    """Return the first available texture in the bank."""
    tex_dir = bank_dir / "textures"
    for p in sorted(tex_dir.glob("*.png")):
        tex = cv2.imread(str(p))
        if tex is not None:
            return tex
    # Last resort: solid grey
    return np.full((420, 296, 3), 128, dtype=np.uint8)
