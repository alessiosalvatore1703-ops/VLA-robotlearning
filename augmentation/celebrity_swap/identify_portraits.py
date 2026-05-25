"""
Identify which celebrity is printed in each detected portrait region.

Uses CLIP (openai/clip-vit-base-patch32) in zero-shot mode:
  - Text prompts: "a portrait photo of <name>"
  - For each portrait crop, pick the name with highest cosine similarity.

Also exposes `compute_clip_embeddings` for pre-computing reference
embeddings that can be cached to disk (used by prepare_celeb_bank).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from augmentation.celebrity_swap.prepare_celeb_bank import CELEB_NAMES

logger = logging.getLogger(__name__)

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP model %s …", _CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).eval()
    return _clip_model, _clip_processor


def _crop_to_pil(frame_bgr: np.ndarray, quad: np.ndarray) -> Image.Image:
    """Warp the portrait quad to a rectangular crop."""
    x0 = int(quad[:, 0].min())
    x1 = int(quad[:, 0].max())
    y0 = int(quad[:, 1].min())
    y1 = int(quad[:, 1].max())
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(frame_bgr.shape[1], x1)
    y1 = min(frame_bgr.shape[0], y1)
    crop = frame_bgr[y0:y1, x0:x1]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def identify_portraits(
    frame_bgr: np.ndarray,
    quads: list[np.ndarray],
    candidate_names: Optional[list[str]] = None,
) -> list[str]:
    """
    For each quad in *quads*, return the best-matching celebrity name from
    *candidate_names* (defaults to all 30 in CELEB_NAMES).

    Returns a list of str, same length as quads.
    """
    if not quads:
        return []

    if candidate_names is None:
        candidate_names = CELEB_NAMES

    model, processor = _load_clip()

    # Build text prompts
    texts = [f"a portrait photo of {n}" for n in candidate_names]

    def _to_tensor(feat) -> torch.Tensor:
        """Handle both bare tensor (old transformers) and output object (new transformers)."""
        if isinstance(feat, torch.Tensor):
            return feat
        # BaseModelOutputWithPooling / similar — pooler_output is the right field
        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            return feat.pooler_output
        if hasattr(feat, "last_hidden_state"):
            return feat.last_hidden_state[:, 0]
        # Fallback: first element of the object
        return feat[0]

    # Encode texts once
    with torch.no_grad():
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        text_features = _to_tensor(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results: list[str] = []
    for quad in quads:
        crop = _crop_to_pil(frame_bgr, quad)
        with torch.no_grad():
            img_inputs = processor(images=crop, return_tensors="pt")
            img_features = _to_tensor(model.get_image_features(**img_inputs))
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        sims = (img_features @ text_features.T).squeeze(0)
        best_idx = int(sims.argmax())
        best_name = candidate_names[best_idx]
        best_score = float(sims[best_idx])
        logger.debug("  Quad centroid (%.0f, %.0f) → %s (score=%.3f)",
                     quad[:, 0].mean(), quad[:, 1].mean(), best_name, best_score)
        results.append(best_name)

    return results


def identify_from_task(
    frame_bgr: np.ndarray,
    quads: list[np.ndarray],
    task_str: str,
    known_celebs: list[str] = None,
) -> dict[int, str]:
    """
    Identify which quad index corresponds to which celeb.
    *task_str* tells us which celeb is the target; we still run CLIP
    for the others to build the full quad→celeb mapping.

    Returns: {quad_idx: celeb_name}
    """
    if known_celebs is None:
        known_celebs = ["Taylor Swift", "Barack Obama", "Yann LeCun"]

    names = identify_portraits(frame_bgr, quads, candidate_names=known_celebs)

    # Sanity check: the target celeb name from task_str should be among results
    target_name = _extract_target_name(task_str, known_celebs)
    if target_name and target_name not in names:
        logger.warning(
            "Task target '%s' not found in CLIP results %s; "
            "CLIP may have misidentified — using as-is.",
            target_name, names,
        )

    return {i: name for i, name in enumerate(names)}


def _extract_target_name(task_str: str, known_celebs: list[str]) -> Optional[str]:
    for name in known_celebs:
        if name.lower() in task_str.lower():
            return name
    return None
