#!/usr/bin/env python3
"""Prune local LeRobot episodes, push the cleaned dataset, and optionally delete local files.

Typical DAgger workflow:

1. Record with --dataset.push_to_hub=false.
2. Inspect which correction episodes are bad.
3. Run this script with --delete-episodes.
4. Push the cleaned result to the Hub.
5. Optionally remove local source/output folders after the push succeeds.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.utils.constants import HF_LEROBOT_HOME


def default_root(repo_id: str) -> Path:
    return HF_LEROBOT_HOME / repo_id


def validate_dataset_root(root: Path) -> None:
    missing = [name for name in ("meta", "data") if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} does not look like a LeRobot dataset root; missing: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Stamped dataset repo id, e.g. ETHrobotlearning/rollout_task1_20260520_231500.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Exact local dataset folder. Defaults to $HF_LEROBOT_HOME/<repo-id>.",
    )
    parser.add_argument(
        "--delete-episodes",
        nargs="*",
        type=int,
        default=[],
        help="Episode indices to remove before pushing. Use none to push as-is.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Where to write the cleaned dataset. Defaults to <root>_pruned when deleting episodes.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--delete-local-after-push",
        action="store_true",
        help="After a successful push, remove both source root and cleaned output root.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete an existing output-root before writing the cleaned dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.root).expanduser() if args.root else default_root(args.repo_id)
    source_root = source_root.resolve()
    validate_dataset_root(source_root)

    if args.delete_local_after_push and not args.push_to_hub:
        raise ValueError("--delete-local-after-push requires --push-to-hub.")

    print(f"Source repo id : {args.repo_id}")
    print(f"Source root    : {source_root}")
    print(f"Delete episodes: {args.delete_episodes if args.delete_episodes else 'none'}")

    if args.delete_episodes:
        output_root = (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else source_root.with_name(source_root.name + "_pruned")
        )
        if output_root == source_root:
            raise ValueError("output-root must be different from source root.")
        if output_root.exists():
            if not args.overwrite_output:
                raise FileExistsError(f"{output_root} already exists. Pass --overwrite-output to replace it.")
            shutil.rmtree(output_root)

        source_dataset = LeRobotDataset(args.repo_id, root=source_root)
        print(f"Original episodes: {source_dataset.meta.total_episodes}")
        cleaned_dataset = delete_episodes(
            source_dataset,
            episode_indices=args.delete_episodes,
            output_dir=output_root,
            repo_id=args.repo_id,
        )
        push_dataset = cleaned_dataset
        local_roots_to_delete = {source_root, output_root}
        print(f"Cleaned root    : {output_root}")
        print(f"Cleaned episodes: {cleaned_dataset.meta.total_episodes}")
    else:
        push_dataset = LeRobotDataset(args.repo_id, root=source_root)
        local_roots_to_delete = {source_root}
        print(f"Episodes: {push_dataset.meta.total_episodes}")

    if args.push_to_hub:
        print(f"Pushing cleaned dataset to Hub: {args.repo_id}")
        push_dataset.push_to_hub()
        print("Push complete.")

        if args.delete_local_after_push:
            for root in sorted(local_roots_to_delete):
                if root.exists():
                    print(f"Deleting local folder: {root}")
                    shutil.rmtree(root)
            print("Local cleanup complete.")
    else:
        print("Not pushed. Add --push-to-hub when ready.")


if __name__ == "__main__":
    main()
