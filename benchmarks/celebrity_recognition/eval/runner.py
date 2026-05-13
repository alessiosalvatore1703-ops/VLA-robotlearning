import csv
import time
from pathlib import Path

from tqdm import tqdm

from ..data.loader import SceneSample
from ..models.base import VLMBase


def run(model: VLMBase, samples: list[SceneSample], prompt_template: str, output_csv: str) -> list[dict]:
    results = []
    for sample in tqdm(samples, desc="Inferring", unit="img"):
        prompt = prompt_template.format(celebrity=sample.target_celebrity)
        t0 = time.perf_counter()
        prediction = model.predict(sample.scene, prompt)
        elapsed = time.perf_counter() - t0
        results.append({
            "scene_path": sample.scene_path,
            "target_celebrity": sample.target_celebrity,
            "target_position": sample.target_position,
            "prediction": prediction,
            "latency_s": round(elapsed, 3),
        })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene_path", "target_celebrity", "target_position", "prediction", "latency_s"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    return results
