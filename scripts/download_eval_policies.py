#!/usr/bin/env python3
"""Download final evaluation policy snapshots into policy_checkpoints/.

This is mainly useful if the local checkpoint folders were removed. The zip
submission should already include them.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


POLICIES = {
    "task1": "ETHrobotlearning/task1-dagger_038000",
    "task2": "ETHrobotlearning/smolvla_task2_colors_dagger_lr2e-5-step2000",
    "task3": "Alessio03/smolvla-task3-50k",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "policy_checkpoints"
    root.mkdir(exist_ok=True)
    for task, repo_id in POLICIES.items():
        dst = root / task
        print(f"Downloading {repo_id} -> {dst}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=dst,
            force_download=True,
        )


if __name__ == "__main__":
    main()
