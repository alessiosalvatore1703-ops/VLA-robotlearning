import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

POSITIONS = ["top_left", "top_right", "bottom_left", "bottom_right"]


@dataclass
class SceneSample:
    scene: Image.Image
    scene_path: str
    target_celebrity: str       # the name that would appear in the VLA prompt
    target_position: str        # ground truth: one of POSITIONS
    all_celebrities: list[str]  # all 4, ordered by POSITIONS


def load_dataset(csv_path: str) -> list[SceneSample]:
    """
    Expects a CSV with columns:
      scene_path, target_celebrity, target_position, all_celebrities
    where all_celebrities is a JSON list of 4 names ordered by POSITIONS.
    """
    base = Path(csv_path).parent
    samples = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            path = Path(row["scene_path"])
            if not path.is_absolute():
                path = base / path
            samples.append(SceneSample(
                scene=Image.open(path).convert("RGB"),
                scene_path=str(path),
                target_celebrity=row["target_celebrity"].strip(),
                target_position=row["target_position"].strip(),
                all_celebrities=json.loads(row["all_celebrities"]),
            ))
    return samples
