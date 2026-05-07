"""Frame-level visual augmentation applied consistently across all frames in an episode."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from augmentation.config import AugmentationConfig


# ---------------------------------------------------------------------------
# Parameter sampling — sampled once per episode, then applied to every frame
# ---------------------------------------------------------------------------

@dataclass
class VisualParams:
    brightness: float
    contrast: float
    saturation: float
    hue_delta: float       # degrees
    gamma: float
    noise_std: float
    blur_radius: float     # 0 = no blur
    shadow: bool
    shadow_alpha: float
    shadow_x: float        # fraction of width [0,1]
    translate_dx: float    # fraction of width
    translate_dy: float    # fraction of height


def sample_params(cfg: AugmentationConfig, rng: random.Random) -> VisualParams:
    return VisualParams(
        brightness=rng.uniform(*cfg.brightness_range),
        contrast=rng.uniform(*cfg.contrast_range),
        saturation=rng.uniform(*cfg.saturation_range),
        hue_delta=rng.uniform(*cfg.hue_range),
        gamma=rng.uniform(*cfg.gamma_range),
        noise_std=rng.uniform(0.0, cfg.noise_std_max),
        blur_radius=rng.uniform(0.5, 1.5) if rng.random() < cfg.blur_prob else 0.0,
        shadow=rng.random() < cfg.shadow_prob,
        shadow_alpha=rng.uniform(0.15, 0.35),
        shadow_x=rng.uniform(0.2, 0.8),
        translate_dx=rng.uniform(*cfg.translate_range),
        translate_dy=rng.uniform(*cfg.translate_range),
    )


# ---------------------------------------------------------------------------
# Per-frame transforms
# ---------------------------------------------------------------------------

def _shift_hue(arr: np.ndarray, delta_degrees: float) -> np.ndarray:
    """Vectorized RGB→HSV hue shift→RGB. arr is uint8 HWC."""
    if abs(delta_degrees) < 0.5:
        return arr

    f = arr.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]

    max_c = np.max(f, axis=-1)
    min_c = np.min(f, axis=-1)
    delta = max_c - min_c

    v = max_c
    s = np.where(max_c > 1e-6, delta / max_c, 0.0)

    hue = np.zeros_like(v)
    m_r = (max_c == r) & (delta > 1e-6)
    m_g = (max_c == g) & (delta > 1e-6)
    m_b = (max_c == b) & (delta > 1e-6)
    hue[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6
    hue[m_g] = (b[m_g] - r[m_g]) / delta[m_g] + 2
    hue[m_b] = (r[m_b] - g[m_b]) / delta[m_b] + 4
    hue /= 6.0
    hue = (hue + delta_degrees / 360.0) % 1.0

    # HSV → RGB
    hi = (hue * 6).astype(np.int32) % 6
    f_h = hue * 6 - np.floor(hue * 6)
    p = v * (1 - s)
    q = v * (1 - f_h * s)
    t = v * (1 - (1 - f_h) * s)

    out = np.zeros_like(f)
    for i, (rv, gv, bv) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        mask = hi == i
        out[mask, 0] = rv[mask]
        out[mask, 1] = gv[mask]
        out[mask, 2] = bv[mask]

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _add_shadow(img: Image.Image, alpha: float, x_frac: float) -> Image.Image:
    """Simple vertical shadow strip on one side."""
    w, h = img.size
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, int(alpha * 255)))
    x_px = int(x_frac * w)
    mask = Image.new("L", (w, h), 0)
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[:, :x_px] = 255
    mask.putdata(mask_arr.flatten().tolist())
    shadow.paste(overlay, mask=mask)
    return Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")


def augment_frame(img: Image.Image, params: VisualParams) -> Image.Image:
    # Brightness
    img = ImageEnhance.Brightness(img).enhance(params.brightness)
    # Contrast
    img = ImageEnhance.Contrast(img).enhance(params.contrast)
    # Saturation
    img = ImageEnhance.Color(img).enhance(params.saturation)
    # Hue shift (vectorized NumPy)
    arr = np.array(img)
    arr = _shift_hue(arr, params.hue_delta)
    # Gamma
    gamma_arr = (np.power(arr.astype(np.float32) / 255.0, 1.0 / params.gamma) * 255.0)
    arr = np.clip(gamma_arr, 0, 255).astype(np.uint8)
    # Gaussian noise
    if params.noise_std > 0:
        noise = np.random.normal(0, params.noise_std, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Blur
    if params.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))
    # Shadow
    if params.shadow:
        img = _add_shadow(img, params.shadow_alpha, params.shadow_x)
    # Translation (fill edge with border pixels via pad-then-crop)
    w, h = img.size
    dx = int(params.translate_dx * w)
    dy = int(params.translate_dy * h)
    if dx != 0 or dy != 0:
        arr = np.array(img)
        arr = np.roll(arr, dy, axis=0)
        arr = np.roll(arr, dx, axis=1)
        img = Image.fromarray(arr)
    return img


# ---------------------------------------------------------------------------
# Episode-level augmentation (applies params consistently across all frames)
# ---------------------------------------------------------------------------

def augment_video(
    frames: torch.Tensor,
    params: VisualParams,
) -> torch.Tensor:
    """Apply augmentation to a THWC uint8 tensor. Returns same shape."""
    augmented = []
    for i in range(frames.shape[0]):
        pil = Image.fromarray(frames[i].numpy())
        pil = augment_frame(pil, params)
        augmented.append(torch.from_numpy(np.array(pil)))
    return torch.stack(augmented)
