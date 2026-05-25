"""
One-time script: download ielminawi/celeb30, crop each celebrity portrait to
a DIN A5 aspect ratio with white border, and save to bank/textures/<name>.png.

Run:
    python -m augmentation.celebrity_swap.prepare_celeb_bank [--bank-dir ./bank]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# All 30 celeb names in label order (matches ielminawi/celeb30 class labels 0-29)
CELEB_NAMES = [
    "Taylor Swift",       # 0  — in-distribution
    "Barack Obama",       # 1  — in-distribution
    "Yann LeCun",         # 2  — in-distribution
    "Beyonce",            # 3
    "Rihanna",            # 4
    "Drake",              # 5
    "Ed Sheeran",         # 6
    "Billie Eilish",      # 7
    "Bad Bunny",          # 8
    "Kanye West",         # 9
    "Leonardo DiCaprio",  # 10
    "Brad Pitt",          # 11
    "Tom Cruise",         # 12
    "Scarlett Johansson", # 13
    "Will Smith",         # 14
    "Dwayne Johnson",     # 15
    "Emma Watson",        # 16
    "Margot Robbie",      # 17
    "Elon Musk",          # 18
    "Bill Gates",         # 19
    "Mark Zuckerberg",    # 20
    "Sam Altman",         # 21
    "Jensen Huang",       # 22
    "Donald Trump",       # 23
    "Michelle Obama",     # 24
    "Lionel Messi",       # 25
    "Cristiano Ronaldo",  # 26
    "Lamine Yamal",       # 27
    "LeBron James",       # 28
    "Lewis Hamilton",     # 29
]

# DIN A5 portrait ratio: width/height = 148/210
A5_RATIO = 148 / 210       # ≈ 0.7048
# Target height for the texture (pixels); width = round(A5_RATIO * TARGET_H)
TARGET_H = 420
TARGET_W = round(A5_RATIO * TARGET_H)   # ≈ 296
# White border as fraction of short side
BORDER_FRAC = 0.06


def _name_to_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("é", "e")


def _detect_face_opencv(img_bgr: np.ndarray):
    """Return (x, y, w, h) of the largest detected face, or None."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    # Pick the largest face
    return max(faces, key=lambda f: f[2] * f[3])


def _crop_to_a5(img_pil: Image.Image, face_box=None) -> Image.Image:
    """
    Crop the image to DIN A5 portrait aspect ratio, centred on the face if found.
    Then resize to TARGET_W × TARGET_H and add a white border.
    """
    w, h = img_pil.size
    img_np = np.array(img_pil.convert("RGB"))

    # Desired crop dimensions for A5 ratio
    if h / w >= 1 / A5_RATIO:
        # Image is taller than A5 → crop height
        crop_w = w
        crop_h = round(w / A5_RATIO)
    else:
        # Image is wider → crop width
        crop_h = h
        crop_w = round(h * A5_RATIO)

    # Centre on face if available, else on image centre
    if face_box is not None:
        fx, fy, fw, fh = face_box
        cx = fx + fw // 2
        cy = fy + fh // 2
    else:
        cx, cy = w // 2, h // 2

    # Clamp so we don't go out of bounds
    x0 = max(0, min(cx - crop_w // 2, w - crop_w))
    y0 = max(0, min(cy - crop_h // 2, h - crop_h))
    cropped = img_np[y0 : y0 + crop_h, x0 : x0 + crop_w]

    # Resize to target size
    resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)

    # Add white border (mimics the printed DIN A5 sheet with white margins)
    border = round(TARGET_W * BORDER_FRAC)
    bordered = cv2.copyMakeBorder(
        resized,
        border, border, border, border,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return Image.fromarray(bordered)


def build_bank(bank_dir: Path, hf_repo: str = "ielminawi/celeb30") -> None:
    """Download celeb30 and write one A5-texture PNG per celebrity."""
    from datasets import load_dataset

    tex_dir = bank_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {hf_repo} …")
    ds = load_dataset(hf_repo, split="train")

    # Group images by label
    by_label: dict[int, list] = {}
    for row in ds:
        lbl = int(row["label"])
        by_label.setdefault(lbl, []).append(row["image"])

    for label, images in sorted(by_label.items()):
        if label >= len(CELEB_NAMES):
            continue
        name = CELEB_NAMES[label]
        slug = _name_to_slug(name)
        out_path = tex_dir / f"{slug}.png"

        if out_path.exists():
            logger.info(f"  [{label:02d}] {name}: already exists, skipping.")
            continue

        best_img = None
        best_face = None

        for pil_img in images:
            img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            face = _detect_face_opencv(img_bgr)
            if face is not None:
                # Prefer the image whose face is most centred and large
                fx, fy, fw, fh = face
                iw, ih = pil_img.size
                cx_err = abs(fx + fw / 2 - iw / 2) / iw
                score = (fw * fh) / (iw * ih) - cx_err
                if best_img is None or score > getattr(best_img, "_score", -1):
                    best_img = pil_img
                    best_img._score = score  # type: ignore[attr-defined]
                    best_face = face

        if best_img is None:
            # Fall back to first image, no face detected
            logger.warning(f"  [{label:02d}] {name}: no face detected, using first image.")
            best_img = images[0]

        texture = _crop_to_a5(best_img, best_face)
        texture.save(out_path)
        logger.info(f"  [{label:02d}] {name}: saved {out_path.name} "
                    f"({texture.width}×{texture.height}px)")

    # Write name lookup: slug → display name
    lookup_path = bank_dir / "names.txt"
    with open(lookup_path, "w") as f:
        for label, name in enumerate(CELEB_NAMES):
            slug = _name_to_slug(name)
            tex_exists = (tex_dir / f"{slug}.png").exists()
            f.write(f"{label}\t{slug}\t{name}\t{'ok' if tex_exists else 'missing'}\n")

    logger.info(f"Bank ready at {bank_dir}. {len(by_label)} celebrities.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", default="./bank", type=Path)
    ap.add_argument("--hf-repo", default="ielminawi/celeb30")
    args = ap.parse_args()
    build_bank(args.bank_dir, args.hf_repo)
