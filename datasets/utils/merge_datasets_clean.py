#!/usr/bin/env python3
"""Cleanly merge LeRobot v3 datasets and optionally push to the Hub.

Unlike the older merge helper, this script overwrites stale files in the target
repo and writes a refreshed ``meta/stats.json``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from move_episodes_between_datasets import (
    _build_dataset,
    _download_hf_dataset,
    _load_episodes,
    _load_info,
    _select_episodes,
    _upload_folder,
    _validate,
)


def _is_hf_repo_id(value: str) -> bool:
    path = Path(value)
    if path.exists():
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def _validate_compatible(roots: list[Path]) -> None:
    ref = _load_info(roots[0])
    errors: list[str] = []
    for root in roots[1:]:
        info = _load_info(root)
        for key in ("fps", "robot_type"):
            if info.get(key) != ref.get(key):
                errors.append(f"{root.name}: {key} {info.get(key)!r} != {ref.get(key)!r}")

        ref_features = ref.get("features", {})
        features = info.get("features", {})
        if set(features) != set(ref_features):
            errors.append(f"{root.name}: feature keys differ")
            continue
        for feature_name, ref_feature in ref_features.items():
            feature = features[feature_name]
            if feature.get("dtype") != ref_feature.get("dtype"):
                errors.append(f"{root.name}: {feature_name} dtype differs")
            if feature.get("shape") != ref_feature.get("shape"):
                errors.append(f"{root.name}: {feature_name} shape differs")

    if errors:
        raise ValueError("Datasets are not compatible:\n  " + "\n  ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", help="HF repo IDs or local dataset paths")
    parser.add_argument("--output", required=True, help="HF repo ID or local output path")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("Provide at least two --inputs datasets.")
    if args.push_to_hub and not _is_hf_repo_id(args.output):
        parser.error("--push-to-hub requires --output to be a Hugging Face repo id.")

    base_tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        base_tmp = tempfile.TemporaryDirectory(prefix="lerobot_clean_merge_")
        work = Path(base_tmp.name)

    try:
        roots: list[Path] = []
        for i, source in enumerate(args.inputs):
            if _is_hf_repo_id(source):
                root = work / f"source_{i:03d}"
                _download_hf_dataset(source, root)
            else:
                root = Path(source)
                if not root.is_dir():
                    raise SystemExit(f"Input path does not exist: {root}")
            roots.append(root)

        _validate_compatible(roots)

        refs = []
        for source_name, root in zip(args.inputs, roots):
            episodes = _load_episodes(root)
            indices = [int(value) for value in episodes["episode_index"].tolist()]
            refs.extend(_select_episodes(root, source_name, indices))

        output_root = work / "merged_output" if args.push_to_hub else Path(args.output)
        stats = _build_dataset(roots[0], refs, output_root)
        _validate(output_root)

        print("\nMerge summary")
        print(f"  inputs: {args.inputs}")
        print(f"  output: {args.output}")
        print(f"  merged dataset: {stats}")

        if args.push_to_hub:
            _upload_folder(
                output_root,
                args.output,
                f"Clean merge of {', '.join(args.inputs)}",
            )
        else:
            print(f"Saved merged dataset to {output_root}")
    finally:
        if base_tmp is not None:
            base_tmp.cleanup()


if __name__ == "__main__":
    main()
