"""Post-augmentation validation checks."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


def _dominant_hue(frame: torch.Tensor) -> np.ndarray:
    """Return mean HSV hue for the non-grey pixels in a frame (HWC uint8)."""
    arr = frame.numpy().astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    max_c = np.max(arr, axis=-1)
    min_c = np.min(arr, axis=-1)
    delta = max_c - min_c

    # Only use pixels that are sufficiently saturated (not grey)
    mask = delta > 0.15
    if not mask.any():
        return np.array([])

    hue = np.zeros_like(max_c)
    m_r = (max_c == r) & mask
    m_g = (max_c == g) & mask
    m_b = (max_c == b) & mask
    hue[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6
    hue[m_g] = (b[m_g] - r[m_g]) / delta[m_g] + 2
    hue[m_b] = (r[m_b] - g[m_b]) / delta[m_b] + 4
    return hue[mask] * 60  # degrees [0, 360)


def check_episode(
    original: Dict,
    augmented: Dict,
    max_hue_shift_degrees: float = 25.0,
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, list_of_warnings).

    Checks:
    1. Frame count matches action count in augmented episode.
    2. No null / zero-size frames.
    3. Hue shift between original and augmented is within bound (uses a
       sample of frames to keep cost low).
    """
    errors: List[str] = []

    aug_df = augmented["df"]
    n_frames_df = len(aug_df)
    for vk, frames in augmented["videos"].items():
        if frames.shape[0] != n_frames_df:
            errors.append(
                f"Frame count mismatch for {vk}: video has {frames.shape[0]}, "
                f"parquet has {n_frames_df}"
            )
        if frames.numel() == 0:
            errors.append(f"Empty video tensor for {vk}")

    # Hue-shift sanity check on a sample of frames
    for vk in augmented["videos"]:
        orig_frames = original["videos"].get(vk)
        aug_frames = augmented["videos"].get(vk)
        if orig_frames is None or aug_frames is None:
            continue

        sample_idx = np.linspace(0, orig_frames.shape[0] - 1, min(5, orig_frames.shape[0]), dtype=int)
        for i in sample_idx:
            orig_hues = _dominant_hue(orig_frames[i])
            aug_hues = _dominant_hue(aug_frames[i])
            if orig_hues.size == 0 or aug_hues.size == 0:
                continue
            shift = abs(np.mean(aug_hues) - np.mean(orig_hues))
            # Wrap-around correction (e.g. 350° vs 5°)
            shift = min(shift, 360 - shift)
            if shift > max_hue_shift_degrees:
                errors.append(
                    f"Excessive hue shift in {vk} frame {i}: {shift:.1f}° "
                    f"(limit {max_hue_shift_degrees}°)"
                )

    ok = len(errors) == 0
    return ok, errors
