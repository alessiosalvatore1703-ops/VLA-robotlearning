"""
Per-frame portrait-swap compositing with occlusion-aware blending.

For each portrait quad:
  1. Warp the new celebrity texture into the quad via perspective transform.
  2. Build a 3-layer occlusion mask (hard occluder + shadow).
  3. Apply shadow-aware compositing: shadows from the arm fall naturally onto
     the new celebrity face via luminance multiplication.

All functions operate on BGR uint8 numpy arrays.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def load_texture(bank_dir: Path, celeb_slug: str) -> np.ndarray:
    """Load a pre-prepared A5 texture as a BGR uint8 array."""
    tex_path = bank_dir / "textures" / f"{celeb_slug}.png"
    if not tex_path.exists():
        raise FileNotFoundError(f"Texture not found: {tex_path}")
    bgr = cv2.imread(str(tex_path))
    if bgr is None:
        raise ValueError(f"Could not read texture: {tex_path}")
    return bgr  # (H, W, 3) BGR uint8


def _texture_corners(tex: np.ndarray) -> np.ndarray:
    """Return the 4 corners of a texture in TL, TR, BR, BL order."""
    h, w = tex.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def _warp_texture(
    tex: np.ndarray,
    quad: np.ndarray,
    frame_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective-warp *tex* into *quad*.

    Returns:
        warped_tex  : (H, W, 3) BGR — texture warped to frame size
        warped_mask : (H, W)   uint8 — 255 inside quad, 0 outside
    """
    H, W = frame_hw
    src_pts = _texture_corners(tex)
    dst_pts = quad.astype(np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_tex = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_src = np.full(tex.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask_src, M, (W, H), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_tex, warped_mask


def _build_occlusion_mask(
    frame_bgr: np.ndarray,
    clean_bgr: np.ndarray,
    quad: np.ndarray,
    chroma_thresh: float = 12.0,
    lum_ratio_thresh: float = 0.82,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Within the region of *quad*, separate pixels into:
      hard_occ  : chromaticity changed significantly → real object on top (arm/can)
      shadow    : luminance dropped but chromaticity stable → cast shadow

    Returns:
        hard_occ (H, W) uint8 — 255 = hard occluder, keep original pixel
        shadow   (H, W) float32 — luminance ratio [0,1] to apply to swapped texture
    """
    H, W = frame_bgr.shape[:2]

    # Build a boolean mask for the portrait quad ROI
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(roi_mask, [pts], 255)

    # -- Layer 1: LAB colour difference -----------------------------------------
    lab_t = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_0 = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_t, A_t, B_t = lab_t[..., 0], lab_t[..., 1], lab_t[..., 2]
    L_0, A_0, B_0 = lab_0[..., 0], lab_0[..., 1], lab_0[..., 2]

    chroma_diff = np.hypot(A_t - A_0, B_t - B_0)
    lum_ratio   = (L_t + 1.0) / (L_0 + 1.0)

    hard = (chroma_diff > chroma_thresh) & (roi_mask > 0)

    # -- Layer 2: Coke-can colour prior (Coca-Cola red + silver crumple) ---------
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_c, S_c, V_c = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    cola_red  = ((H_c <= 10) | (H_c >= 170)) & (S_c > 110) & (V_c > 60)
    # Silver/grey crumpled regions close to the red area
    cola_grey = (S_c < 60) & (V_c > 90)
    red_nearby = cv2.dilate(cola_red.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    coke_mask  = (cola_red | (cola_grey & red_nearby)) & (roi_mask > 0)

    # -- Layer 3: Robot gripper/arm (dark, low-saturation pixels that moved) -----
    gripper_dark = (S_c < 70) & (V_c < 60)
    gripper_mask = gripper_dark & hard  # only where something changed

    # -- Layer 4: KNN background subtraction already applied upstream
    #    (bg_fgmask injected per-frame by episode_worker via compose_frame_with_bg)

    # Combine hard occluder mask
    hard_occ = (hard | coke_mask.astype(bool) | gripper_mask.astype(bool)).astype(np.uint8) * 255
    hard_occ = cv2.morphologyEx(hard_occ, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    hard_occ = cv2.morphologyEx(hard_occ, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    hard_occ = cv2.dilate(hard_occ, np.ones((5, 5), np.uint8))  # safety margin

    # -- Shadow map: stable chromaticity + luminance drop -----------------------
    soft_region = (chroma_diff <= chroma_thresh) & (roi_mask > 0)
    shadow_ratio = np.ones((H, W), dtype=np.float32)
    shadow_ratio[soft_region] = np.clip(lum_ratio[soft_region], 0.3, 1.2)

    return hard_occ, shadow_ratio


def composite_portraits(
    frame_bgr: np.ndarray,
    clean_bgr: np.ndarray,
    quads: list[np.ndarray],
    textures: list[np.ndarray],
    bg_fgmask: np.ndarray | None = None,
    edge_feather_px: int = 3,
) -> np.ndarray:
    """
    Replace each portrait quad with its corresponding texture, preserving
    occluding objects (arm, coke can) and shadow lighting.

    Args:
        frame_bgr   : current video frame (H, W, 3) BGR uint8
        clean_bgr   : clean reference frame (no arm in portrait regions)
        quads       : list of (4,2) float32 corner arrays (TL,TR,BR,BL)
        textures    : list of BGR textures, one per quad
        bg_fgmask   : (H, W) uint8 background-subtractor foreground mask (optional)
        edge_feather_px : pixels to feather at occluder boundary for seamless blending
    """
    assert len(quads) == len(textures), "quads and textures must have the same length"

    H, W = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    for quad, tex in zip(quads, textures):
        warped_tex, warped_mask = _warp_texture(tex, quad, (H, W))
        hard_occ, shadow_ratio = _build_occlusion_mask(frame_bgr, clean_bgr, quad)

        # Incorporate background-subtractor foreground if provided
        if bg_fgmask is not None:
            roi_mask = np.zeros((H, W), dtype=np.uint8)
            pts = quad.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(roi_mask, [pts], 255)
            bg_fg_roi = (bg_fgmask > 0) & (roi_mask > 0)
            hard_occ = cv2.bitwise_or(hard_occ, (bg_fg_roi.astype(np.uint8) * 255))

        # Apply shadow luminance to the warped texture
        # Work in LAB to manipulate L channel independently
        warped_lab = cv2.cvtColor(warped_tex, cv2.COLOR_BGR2LAB).astype(np.float32)
        warped_lab[..., 0] = np.clip(warped_lab[..., 0] * shadow_ratio, 0, 255)
        warped_shaded = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Compute replace mask: inside quad AND not a hard occluder
        inside = (warped_mask > 0)
        replace = inside & (hard_occ == 0)

        # Feather the boundary between replace and non-replace for seamlessness
        if edge_feather_px > 0:
            replace_f = replace.astype(np.float32)
            hard_f    = (hard_occ > 0).astype(np.float32)
            kernel_sz = edge_feather_px * 2 + 1
            replace_soft = cv2.GaussianBlur(replace_f, (kernel_sz, kernel_sz), edge_feather_px)
            hard_soft    = cv2.GaussianBlur(hard_f,    (kernel_sz, kernel_sz), edge_feather_px)
            # Only feather where inside quad; outside stays untouched
            alpha = replace_soft * inside.astype(np.float32)
            alpha[hard_soft > 0.5] = 0.0  # no blending behind hard occluders

            for c in range(3):
                out[..., c] = (
                    alpha * warped_shaded[..., c].astype(np.float32)
                    + (1.0 - alpha) * out[..., c].astype(np.float32)
                ).clip(0, 255).astype(np.uint8)
        else:
            out[replace] = warped_shaded[replace]

    return out


def composite_frame_with_bg(
    frame_bgr: np.ndarray,
    clean_bgr: np.ndarray,
    quads: list[np.ndarray],
    textures: list[np.ndarray],
    bg_subtractor,
) -> np.ndarray:
    """Convenience wrapper that applies the KNN background subtractor per-frame."""
    fgmask = bg_subtractor.apply(frame_bgr, learningRate=0.0)
    return composite_portraits(frame_bgr, clean_bgr, quads, textures, bg_fgmask=fgmask)
