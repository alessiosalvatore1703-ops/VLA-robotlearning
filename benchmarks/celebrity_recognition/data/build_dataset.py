"""
Build scene dataset from a folder of per-celebrity images.

Expected source layout:
  source_dir/
    taylor_swift/   <- folder name becomes the celebrity label
      001.jpg
      002.jpg
    elon_musk/
      001.jpg
    ...

Output:
  out_dir/
    scenes/
      scene_0001.jpg   <- 2x2 composite of 4 celebrity crops
      ...
    dataset.csv        <- one row per (scene, target_celebrity) pair
"""

import argparse
import csv
import json
import random
from itertools import combinations
from pathlib import Path

from PIL import Image

CELL_SIZE = 256  # px per celebrity cell in the 2x2 grid
POSITIONS = ["top_left", "top_right", "bottom_left", "bottom_right"]
GRID_XY = [(0, 0), (CELL_SIZE, 0), (0, CELL_SIZE), (CELL_SIZE, CELL_SIZE)]


def make_scene(images: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (CELL_SIZE * 2, CELL_SIZE * 2))
    for img, (x, y) in zip(images, GRID_XY):
        canvas.paste(img.resize((CELL_SIZE, CELL_SIZE)), (x, y))
    return canvas


def build(source_dir: str, out_dir: str, scenes_per_combo: int = 2, seed: int = 42):
    random.seed(seed)
    source = Path(source_dir)
    out = Path(out_dir)
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # index: celebrity_name -> list of image paths
    index: dict[str, list[Path]] = {}
    for celeb_dir in sorted(source.iterdir()):
        if not celeb_dir.is_dir():
            continue
        imgs = [p for p in celeb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if imgs:
            index[celeb_dir.name] = imgs

    celebrities = list(index)
    assert len(celebrities) >= 4, "Need at least 4 celebrities to build scenes."

    rows = []
    scene_id = 0

    for combo in combinations(celebrities, 4):
        for _ in range(scenes_per_combo):
            order = list(combo)
            random.shuffle(order)

            images = [Image.open(random.choice(index[c])).convert("RGB") for c in order]
            scene = make_scene(images)

            scene_filename = f"scene_{scene_id:04d}.jpg"
            scene.save(scenes_dir / scene_filename)

            all_json = json.dumps(order)
            for pos, celeb in zip(POSITIONS, order):
                rows.append({
                    "scene_path": f"scenes/{scene_filename}",
                    "target_celebrity": celeb,
                    "target_position": pos,
                    "all_celebrities": all_json,
                })

            scene_id += 1

    csv_path = out / "dataset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_path", "target_celebrity", "target_position", "all_celebrities"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Built {scene_id} scenes → {len(rows)} samples  [{csv_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Folder with per-celebrity subfolders")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--scenes-per-combo", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build(args.source, args.out, args.scenes_per_combo, args.seed)
