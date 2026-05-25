"""
Portrait-quad detection: find the 3 printed celebrity portrait rectangles.

Two-stage pipeline:
  1. YOLO (yolov8n.pt, "person" class) → coarse face bounding boxes.
     Printed portrait photos contain recognisable faces that YOLO detects.
  2. Gradient + contour + minAreaRect → refine each face bbox to the precise
     paper corners.  The boundary between the printed photo and the table
     produces a strong gradient ring; we find it with Canny → contours →
     minimum-area rotated rectangle.

No white-border assumption.  Handles dark photo edges, slight paper tilts,
and small portraits (≈50–80 px wide in a 640×480 frame).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# DIN A5 portrait aspect ratio: width / height = 148/210 ≈ 0.7048
A5_RATIO = 148 / 210
# Reciprocal: height / width ≈ 1.419  (what minAreaRect reports)
A5_ASPECT_HW = 210 / 148   # ≈ 1.419

# ─────────────────────────────────────────────────────────────────────────────
# YOLO model (singleton, loaded once)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_yolo():
    from ultralytics import YOLO
    logger.info("Loading YOLOv8n …")
    return YOLO("yolov8n.pt")  # auto-downloads on first use


def _yolo_person_boxes(
    frame_bgr: np.ndarray,
    conf_hi: float = 0.25,
    conf_lo: float = 0.08,
) -> list[tuple[int, int, int, int]]:
    """
    Run YOLO and return bounding boxes of detected persons (COCO class 0).
    Tries conf_hi first; if too few results, retries at conf_lo.
    Returns list of (x1, y1, x2, y2) in pixel coords.
    """
    model  = _load_yolo()

    def _run(conf):
        res = model(frame_bgr, verbose=False, conf=conf,
                    imgsz=640, classes=[0])
        boxes = []
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
        return boxes

    boxes = _run(conf_hi)
    if len(boxes) < 3:
        boxes = _dedup_boxes(_run(conf_lo))
    return boxes


def _dedup_boxes(
    boxes: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    """Remove duplicate detections by IoU."""
    if not boxes:
        return []
    kept: list[tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True):
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-based portrait-boundary refinement
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_refine(
    frame_bgr: np.ndarray,
    face_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Given a rough face bounding box, return a precise (4, 2) float32 quad
    of the containing DIN A5 portrait paper.

    Strategy:
      - Expand the face bbox to an estimated paper region (+generous margin).
      - Compute gradient magnitude (Sobel) → Canny edges.
      - Find the contour that (a) encloses the face centre, (b) is roughly
        A5-shaped, and has the highest area×aspect_score.
      - Fit cv2.minAreaRect to that contour → precise rotated corners.
    """
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = face_xyxy
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    # ── Estimate full paper bounds from face size ──────────────────────────
    # Face occupies ~55 % of portrait height; placed ~25 % from top.
    ph_est = fh / 0.55
    pw_est = ph_est * A5_RATIO
    px_est = x1 + fw / 2 - pw_est / 2
    py_est = y1 - ph_est * 0.25

    # ── Generous search region ──────────────────────────────────────────────
    marg = max(fw, fh) * 1.0
    sx0 = max(0,  int(px_est - marg))
    sy0 = max(0,  int(py_est - marg))
    sx1 = min(W,  int(px_est + pw_est + marg))
    sy1 = min(H,  int(py_est + ph_est + marg))

    region = frame_bgr[sy0:sy1, sx0:sx1]
    if region.size == 0:
        return _fallback_quad(x1, y1, x2, y2)

    # ── Gradient magnitude ─────────────────────────────────────────────────
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Slight blur to suppress interior texture of the portrait
    gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    # Normalise to uint8
    mx = float(mag.max())
    if mx < 1e-3:
        return _fallback_quad(x1, y1, x2, y2)
    mag_u8 = np.clip(mag / mx * 255, 0, 255).astype(np.uint8)

    # ── Canny edges on gradient image ─────────────────────────────────────
    # Working on the gradient image (rather than the raw frame) suppresses
    # uniform-texture regions (table) while highlighting boundaries (paper edge).
    edges = cv2.Canny(mag_u8, 40, 120)

    # Close small gaps so the paper boundary forms a closed ring
    k_close = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    # ── Find best contour ──────────────────────────────────────────────────
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _fallback_quad(x1, y1, x2, y2)

    face_cx_r = x1 + fw / 2 - sx0   # face centre in region coords
    face_cy_r = y1 + fh / 2 - sy0

    min_area = fw * fh * 1.5
    max_area = fw * fh * 25.0

    best_pts: Optional[np.ndarray] = None
    best_score = -1.0

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        # Must enclose the face centre
        if cv2.pointPolygonTest(cnt, (face_cx_r, face_cy_r), False) < 0:
            continue

        # Fit minimum-area rotated rectangle
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if min(rw, rh) < 1:
            continue
        # Aspect: longer side / shorter side
        aspect_hw = max(rw, rh) / min(rw, rh)

        # Score: area × how close aspect is to A5 (allows ±50 % for perspective)
        asp_err = abs(aspect_hw - A5_ASPECT_HW) / A5_ASPECT_HW
        asp_score = max(0.0, 1.0 - asp_err * 2.0)
        # Prefer smaller but correctly-shaped contours (avoids over-large loose ones)
        score = np.sqrt(area) * (0.2 + 0.8 * asp_score)

        if score > best_score:
            best_score = score
            pts = cv2.boxPoints(rect)              # (4, 2) in region coords
            best_pts = pts + np.array([[sx0, sy0]], dtype=np.float32)

    if best_pts is None:
        return _fallback_quad(x1, y1, x2, y2)

    return _order_quad_corners(best_pts)


def _fallback_quad(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Return an estimated paper quad purely from the face bbox."""
    fw, fh = x2 - x1, y2 - y1
    ph = fh / 0.55
    pw = ph * A5_RATIO
    px = x1 + fw / 2 - pw / 2
    py = y1 - ph * 0.25
    return _order_quad_corners(np.array([
        [px, py], [px + pw, py],
        [px + pw, py + ph], [px, py + ph],
    ], dtype=np.float32))


def _order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 corners in TL, TR, BR, BL order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_portrait_quads(
    frame_bgr: np.ndarray,
    n_expected: int = 3,
) -> list[np.ndarray]:
    """
    Detect DIN A5 portrait quads in *frame_bgr*.

    Returns a list of (4, 2) float32 arrays (TL, TR, BR, BL),
    sorted left-to-right by centroid X.
    """
    H, W = frame_bgr.shape[:2]

    # ── Stage 1: YOLO person detection ────────────────────────────────────
    face_boxes = _yolo_person_boxes(frame_bgr)
    logger.debug("YOLO found %d person boxes.", len(face_boxes))

    if not face_boxes:
        logger.warning("YOLO found no persons in frame.")
        return []

    # ── Stage 2: gradient refinement per face ─────────────────────────────
    quads: list[np.ndarray] = []
    for bbox in face_boxes:
        quad = _gradient_refine(frame_bgr, bbox)
        quads.append(quad)

    # Deduplicate quads whose centres are very close
    quads = _dedup_quads(quads, min_dist_frac=0.05, frame_w=W)

    # Sort left-to-right
    quads.sort(key=lambda q: float(q[:, 0].mean()))

    if len(quads) < n_expected:
        logger.warning("Only found %d/%d portrait quads.", len(quads), n_expected)

    return quads[:n_expected]


def _dedup_quads(
    quads: list[np.ndarray],
    min_dist_frac: float,
    frame_w: int,
) -> list[np.ndarray]:
    """Remove quads whose centres are within min_dist_frac * frame_w of each other."""
    min_dist = min_dist_frac * frame_w
    out: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.reshape(-1, 1, 2))):
        c = q.mean(axis=0)
        if any(np.linalg.norm(c - ec) < min_dist for ec in centres):
            continue
        out.append(q)
        centres.append(c)
    return out


def find_clean_frame(
    frames_bgr: np.ndarray,
    quad_positions: list[np.ndarray],
    scan_range: int = 15,
) -> np.ndarray:
    """
    Among the first *scan_range* frames, return the one with minimum
    frame-to-frame change inside the portrait ROIs (i.e. least occlusion).
    """
    if len(frames_bgr) <= 1 or not quad_positions:
        return frames_bgr[0]

    n = min(scan_range, len(frames_bgr))
    best_idx, best_score = 0, float("inf")

    for t in range(n - 1):
        diff = np.abs(
            frames_bgr[t].astype(np.float32) - frames_bgr[t + 1].astype(np.float32)
        )
        H_f, W_f = diff.shape[:2]
        roi_scores = []
        for q in quad_positions:
            y0r = max(0, int(q[:, 1].min()))
            y1r = min(H_f, int(q[:, 1].max()))
            x0r = max(0, int(q[:, 0].min()))
            x1r = min(W_f, int(q[:, 0].max()))
            roi = diff[y0r:y1r, x0r:x1r]
            if roi.size > 0:
                roi_scores.append(float(roi.mean()))
        total = sum(roi_scores) if roi_scores else float("inf")
        if total < best_score:
            best_score, best_idx = total, t

    return frames_bgr[best_idx]
