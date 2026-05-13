"""
Downloads the Kaggle celebrity face dataset, samples images, and builds scenes.

Requirements:
  pip install kaggle pillow
  Environment variables:
    KAGGLE_USERNAME=<your kaggle username>
    KAGGLE_KEY=<your kaggle API key>

Usage:
  python -m benchmarks.celebrity_recognition.data.download_and_build

Outputs (relative to repo root):
  benchmarks/celebrity_recognition/data/
    raw/          <- extracted Kaggle zip
    scenes/       <- 512x512 composite scene images
    dataset.csv   <- benchmark CSV
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

KAGGLE_DATASET = "vishesh1412/celebrity-face-image-dataset"
IMAGES_PER_CELEBRITY = 10
NUM_SCENES = 200
CELL_SIZE = 256
POSITIONS = ["top_left", "top_right", "bottom_left", "bottom_right"]
GRID_XY = [(0, 0), (CELL_SIZE, 0), (0, CELL_SIZE), (CELL_SIZE, CELL_SIZE)]

DATA_DIR = Path(__file__).parent


def _check_kaggle_credentials() -> None:
    missing = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.environ.get(v)]
    if missing:
        print(f"Error: missing environment variable(s): {', '.join(missing)}")
        print("Set them before running:")
        print("  export KAGGLE_USERNAME=<your_username>")
        print("  export KAGGLE_KEY=<your_api_key>")
        sys.exit(1)


def download(raw_dir: Path) -> None:
    _check_kaggle_credentials()
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading from Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(raw_dir),
            "--quiet",
        ],
        check=True,
    )

    # kaggle saves it as <dataset-slug>.zip
    candidates = list(raw_dir.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No zip found in {raw_dir} after download.")
    zip_path = candidates[0]

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)
    zip_path.unlink()


def index_celebrities(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Returns {celebrity_name: [image_paths]} from the extracted folder.
    Handles one extra nesting level (e.g. raw/Celebrity Faces Dataset/Angelina Jolie/).
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    index: dict[str, list[Path]] = {}

    # find the first directory that itself contains subdirectories of images
    search_roots = [raw_dir] + [d for d in raw_dir.iterdir() if d.is_dir()]
    for root in search_roots:
        for celeb_dir in sorted(root.iterdir()):
            if not celeb_dir.is_dir():
                continue
            imgs = [p for p in celeb_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            if imgs:
                index[celeb_dir.name] = imgs
        if index:
            break

    return index


def make_scene(images: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (CELL_SIZE * 2, CELL_SIZE * 2))
    for img, (x, y) in zip(images, GRID_XY):
        canvas.paste(img.resize((CELL_SIZE, CELL_SIZE)), (x, y))
    return canvas


def build_scenes(
    index: dict[str, list[Path]],
    scenes_dir: Path,
    num_scenes: int,
    seed: int,
) -> list[dict]:
    random.seed(seed)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    celebrities = list(index)
    rows = []

    for scene_id in range(num_scenes):
        order = random.sample(celebrities, 4)
        images = [
            Image.open(random.choice(index[c])).convert("RGB")
            for c in order
        ]
        scene = make_scene(images)
        filename = f"scene_{scene_id:04d}.jpg"
        scene.save(scenes_dir / filename)

        all_json = json.dumps(order)
        for pos, celeb in zip(POSITIONS, order):
            rows.append({
                "scene_path": f"scenes/{filename}",
                "target_celebrity": celeb,
                "target_position": pos,
                "all_celebrities": all_json,
            })

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA_DIR), help="Output directory (default: data/)")
    p.add_argument("--images-per-celebrity", type=int, default=IMAGES_PER_CELEBRITY)
    p.add_argument("--num-scenes", type=int, default=NUM_SCENES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true", help="Reuse existing raw/ folder")
    args = p.parse_args()

    out = Path(args.out)
    raw_dir = out / "raw"

    if not args.skip_download:
        download(raw_dir)
    else:
        print(f"Skipping download, using {raw_dir}")

    print("Indexing celebrities...")
    full_index = index_celebrities(raw_dir)
    print(f"Found {len(full_index)} celebrities: {', '.join(sorted(full_index))}")
    assert len(full_index) >= 4, "Need at least 4 celebrities."

    # sample N images per celebrity
    index = {
        name: random.Random(args.seed).sample(paths, min(args.images_per_celebrity, len(paths)))
        for name, paths in full_index.items()
    }
    total_images = sum(len(v) for v in index.values())
    print(f"Sampled {total_images} images ({args.images_per_celebrity} per celebrity)")

    print(f"Building {args.num_scenes} scenes...")
    scenes_dir = out / "scenes"
    rows = build_scenes(index, scenes_dir, args.num_scenes, args.seed)

    csv_path = out / "dataset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_path", "target_celebrity", "target_position", "all_celebrities"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone.")
    print(f"  Scenes  : {args.num_scenes}  →  {scenes_dir}")
    print(f"  Samples : {len(rows)} rows (4 per scene)  →  {csv_path}")
    print(f"\nTo run the benchmark:")
    print(f"  python run_benchmark.py --model moondream --dataset {csv_path}")
    print(f"  python run_benchmark.py --model smolvlm_500m --dataset {csv_path}")


if __name__ == "__main__":
    main()
