#!/usr/bin/env python3
"""Delete selected episodes from a LeRobot v3 dataset repo.

The output dataset is rebuilt with consecutive episode/global frame indices,
compact task metadata, and a refreshed ``meta/stats.json``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from move_episodes_between_datasets import (
    _build_dataset,
    _download_hf_dataset,
    _load_episodes,
    _select_episodes,
    _upload_folder,
    _validate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", required=True, nargs="+", type=int)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    base_tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        base_tmp = tempfile.TemporaryDirectory(prefix="lerobot_delete_episodes_")
        work = Path(base_tmp.name)

    source_root = work / "source_latest"
    output_root = work / "dataset_updated"

    try:
        _download_hf_dataset(args.repo_id, source_root)
        episodes = _load_episodes(source_root)
        all_indices = [int(value) for value in episodes["episode_index"].tolist()]
        delete_set = set(args.episodes)
        missing = sorted(delete_set - set(all_indices))
        if missing:
            raise ValueError(f"Requested episode(s) not present: {missing}")

        keep_indices = [idx for idx in all_indices if idx not in delete_set]
        refs = _select_episodes(source_root, args.repo_id, keep_indices)
        stats = _build_dataset(source_root, refs, output_root)
        _validate(output_root)

        print("\nDelete summary")
        print(f"  deleted episodes: {sorted(delete_set)}")
        print(f"  remaining dataset: {stats}")

        if args.push_to_hub:
            _upload_folder(
                output_root,
                args.repo_id,
                f"Delete episodes {sorted(delete_set)}",
            )
    finally:
        if base_tmp is not None:
            base_tmp.cleanup()


if __name__ == "__main__":
    main()
