"""
Visual QA tool: detect portrait quads on random source frames and render a
debug grid showing the original frame, detected quads overlay, and (if a bank
is available) the first composite result.

Output: PNG grid saved to --out-dir.

Usage:
    python -m augmentation.celebrity_swap.run_pipeline viz \\
        --src ./local_task3 --n-episodes 20 --out-dir ./viz_output

    # Or directly:
    python augmentation/celebrity_swap/viz_qa.py \\
        --src ./local_task3 --n-episodes 20
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torchvision.io as tvio

from augmentation.celebrity_swap.detect_portraits import (
    detect_portrait_quads, find_clean_frame
)

logger = logging.getLogger(__name__)

# Colour palette for quad outlines (BGR) — one per quad position
QUAD_COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255)]


def _draw_quads(frame_bgr: np.ndarray, quads: list[np.ndarray]) -> np.ndarray:
    out = frame_bgr.copy()
    for i, quad in enumerate(quads):
        color = QUAD_COLORS[i % len(QUAD_COLORS)]
        pts   = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        cx, cy = int(quad[:, 0].mean()), int(quad[:, 1].mean())
        cv2.putText(out, str(i), (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def _draw_composite_preview(
    clean_bgr: np.ndarray,
    quads: list[np.ndarray],
    bank_dir: Path | None,
) -> np.ndarray:
    """Try to composite first available textures for a quick preview."""
    if bank_dir is None or not (bank_dir / "textures").exists():
        return clean_bgr.copy()

    tex_paths = sorted((bank_dir / "textures").glob("*.png"))
    if not tex_paths:
        return clean_bgr.copy()

    try:
        from augmentation.celebrity_swap.compose_frame import composite_portraits, load_texture
        textures = []
        for i in range(len(quads)):
            tex = cv2.imread(str(tex_paths[i % len(tex_paths)]))
            textures.append(tex)
        return composite_portraits(clean_bgr, clean_bgr, quads, textures)
    except Exception as exc:
        logger.debug("Composite preview failed: %s", exc)
        return clean_bgr.copy()


def _make_grid(panels: list[np.ndarray], labels: list[str],
               target_h: int = 240) -> np.ndarray:
    """Resize all panels to *target_h* and concatenate horizontally."""
    resized = []
    for panel, label in zip(panels, labels):
        h, w = panel.shape[:2]
        new_w = int(w * target_h / h)
        img = cv2.resize(panel, (new_w, target_h))
        # Add label bar at top
        bar = np.full((24, new_w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        resized.append(np.vstack([bar, img]))
    return np.hstack(resized)


def visualise_detection(
    src_path: Path,
    n_episodes: int = 20,
    out_dir: Path = Path("./viz_output"),
    bank_dir: Path | None = None,
    seed: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Discover available video files
    vid_dir = src_path / "videos" / "observation.images.front" / "chunk-000"
    if not vid_dir.exists():
        logger.error("Video directory not found: %s", vid_dir)
        return

    all_vids = sorted(vid_dir.glob("*.mp4"))
    if not all_vids:
        logger.error("No MP4 files found in %s", vid_dir)
        return

    sample = rng.sample(all_vids, min(n_episodes, len(all_vids)))

    n_ok    = 0
    n_fail  = 0
    summary = []

    for vid_path in sample:
        ep_id = vid_path.stem  # e.g. "file-042"
        try:
            frames_thwc, _, _ = tvio.read_video(str(vid_path), pts_unit="sec",
                                                output_format="THWC")
        except Exception as exc:
            logger.warning("Could not read %s: %s", vid_path.name, exc)
            n_fail += 1
            continue

        frames_bgr = frames_thwc.numpy()[..., ::-1].copy()  # RGB→BGR
        if len(frames_bgr) == 0:
            n_fail += 1
            continue

        quads      = detect_portrait_quads(frames_bgr[0])
        clean_bgr  = find_clean_frame(frames_bgr, quads)
        quads_clean = detect_portrait_quads(clean_bgr)
        if len(quads_clean) == 3:
            quads = quads_clean

        status = f"found={len(quads)}/3"
        summary.append(f"{ep_id}: {status}")

        # Sample a few frames: clean frame, mid frame, last frame
        T = len(frames_bgr)
        sample_frames = {
            "clean (f0)":   clean_bgr,
            f"mid (f{T//2})": frames_bgr[T // 2],
            f"last (f{T-1})": frames_bgr[-1],
        }

        panels = []
        labels = []
        for label, frame in sample_frames.items():
            panels.append(_draw_quads(frame, quads))
            labels.append(label)

        # Composite preview on clean frame
        panels.append(_draw_composite_preview(clean_bgr, quads, bank_dir))
        labels.append("composite preview")

        grid = _make_grid(panels, labels)
        out_path = out_dir / f"{ep_id}_quads.jpg"
        cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])

        if len(quads) == 3:
            n_ok += 1
        else:
            n_fail += 1
            logger.warning("  %s: only %d/3 quads — check image", ep_id, len(quads))

    print(f"\nQA summary: {n_ok}/{len(sample)} episodes had all 3 quads detected.")
    print(f"Grids saved to: {out_dir}")
    for line in summary:
        print(" ", line)

    # Write a combined summary sheet (first 4 rows of 4 grids per row)
    all_grids = sorted(out_dir.glob("*_quads.jpg"))
    if len(all_grids) >= 4:
        rows = []
        for i in range(0, min(16, len(all_grids)), 4):
            batch  = [cv2.imread(str(p)) for p in all_grids[i:i + 4]]
            # Uniform height for row
            max_h  = max(b.shape[0] for b in batch)
            resized = [cv2.resize(b, (int(b.shape[1] * max_h / b.shape[0]), max_h))
                       for b in batch]
            rows.append(np.hstack(resized))
        max_w = max(r.shape[1] for r in rows)
        padded = [cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1],
                                     cv2.BORDER_CONSTANT, value=(30, 30, 30))
                  for r in rows]
        sheet = np.vstack(padded)
        cv2.imwrite(str(out_dir / "_summary_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"Summary sheet: {out_dir / '_summary_sheet.jpg'}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",        required=True, help="Local dataset path")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--out-dir",    default="./viz_output")
    ap.add_argument("--bank-dir",   default=None)
    ap.add_argument("--seed",       type=int, default=0)
    a = ap.parse_args()
    visualise_detection(
        Path(a.src),
        n_episodes=a.n_episodes,
        out_dir=Path(a.out_dir),
        bank_dir=Path(a.bank_dir) if a.bank_dir else None,
        seed=a.seed,
    )
