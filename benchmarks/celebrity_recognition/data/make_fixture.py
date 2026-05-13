"""
Generates a synthetic test fixture (no real photos needed).
Each celebrity is a solid-color tile with their name printed on it.
Run once to get a self-contained dataset you can sanity-check the pipeline with.

Usage:
  python -m benchmarks.celebrity_recognition.data.make_fixture --out /tmp/celeb_fixture
"""

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELEBRITIES = [
    ("Taylor Swift",    (255, 182, 193)),  # pink
    ("Elon Musk",       (173, 216, 230)),  # light blue
    ("Beyoncé",         (255, 228, 181)),  # peach
    ("Cristiano Ronaldo", (144, 238, 144)),  # light green
    ("Barack Obama",    (221, 160, 221)),  # plum
    ("Rihanna",         (255, 255, 153)),  # yellow
]

CELL_SIZE = 256
POSITIONS = ["top_left", "top_right", "bottom_left", "bottom_right"]
GRID_XY = [(0, 0), (CELL_SIZE, 0), (0, CELL_SIZE), (CELL_SIZE, CELL_SIZE)]


def make_tile(name: str, color: tuple) -> Image.Image:
    img = Image.new("RGB", (CELL_SIZE, CELL_SIZE), color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()
    # wrap long names
    words = name.split()
    lines = [" ".join(words[:2]), " ".join(words[2:])] if len(words) > 2 else [name]
    y = CELL_SIZE // 2 - 20
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((CELL_SIZE - w) // 2, y), line, fill=(0, 0, 0), font=font)
        y += 30
    return img


def build_fixture(out_dir: str):
    out = Path(out_dir)
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    rows = []

    # build one scene per combination of 4 from the 6 celebrities (15 combos)
    from itertools import combinations
    for scene_id, combo in enumerate(combinations(CELEBRITIES, 4)):
        order = list(combo)
        random.shuffle(order)

        canvas = Image.new("RGB", (CELL_SIZE * 2, CELL_SIZE * 2))
        for (name, color), (x, y) in zip(order, GRID_XY):
            canvas.paste(make_tile(name, color), (x, y))

        filename = f"scene_{scene_id:04d}.jpg"
        canvas.save(scenes_dir / filename)

        names = [c[0] for c in order]
        all_json = json.dumps(names)
        for pos, (name, _) in zip(POSITIONS, order):
            rows.append({
                "scene_path": f"scenes/{filename}",
                "target_celebrity": name,
                "target_position": pos,
                "all_celebrities": all_json,
            })

    csv_path = out / "dataset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_path", "target_celebrity", "target_position", "all_celebrities"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixture: {scene_id + 1} scenes, {len(rows)} samples → {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/celeb_fixture")
    args = p.parse_args()
    build_fixture(args.out)
