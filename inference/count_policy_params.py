#!/usr/bin/env python3
"""Print the number of parameters in a Hugging Face or local policy checkpoint.

Examples:

    python inference/count_policy_params.py Alessio03/smolvla-task3-50k

    python inference/count_policy_params.py ETHrobotlearning/my-policy-step10000

    python inference/count_policy_params.py outputs/train/my_local_policy
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
SKIP_WEIGHT_NAME_PARTS = {
    "optimizer",
    "optim",
    "scheduler",
    "trainer",
    "training_state",
    "rng_state",
    "random_states",
}


def is_hf_repo_id(value: str) -> bool:
    if Path(value).expanduser().exists():
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value:,} ({value / 1_000_000_000:.3f}B)"
    if value >= 1_000_000:
        return f"{value:,} ({value / 1_000_000:.3f}M)"
    if value >= 1_000:
        return f"{value:,} ({value / 1_000:.3f}K)"
    return f"{value:,}"


def should_skip_weight_file(path: Path) -> bool:
    name = path.name.lower()
    return any(part in name for part in SKIP_WEIGHT_NAME_PARTS)


def find_weight_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        files.extend(root.rglob(pattern))
    return sorted(path for path in files if path.is_file() and not should_skip_weight_file(path))


def download_weights(policy: str, revision: str | None, force_download: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=policy,
            repo_type="model",
            revision=revision,
            force_download=force_download,
            allow_patterns=WEIGHT_PATTERNS + ["config.json", "*.yaml", "*.md"],
        )
    )


def numel_from_shape(shape: list[int] | tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(math.prod(int(dim) for dim in shape))


def count_safetensors(path: Path) -> int:
    from safetensors import safe_open

    total = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            total += numel_from_shape(handle.get_slice(key).get_shape())
    return total


def iter_tensors(value: Any):
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def count_torch_checkpoint(path: Path) -> int:
    import torch

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    return sum(int(tensor.numel()) for tensor in iter_tensors(checkpoint))


def count_policy(policy: str, revision: str | None, force_download: bool, allow_pickle: bool) -> tuple[int, list[tuple[str, int]]]:
    root = download_weights(policy, revision, force_download) if is_hf_repo_id(policy) else Path(policy).expanduser()
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Policy path does not exist: {root}")

    weight_files = find_weight_files(root)
    if not weight_files:
        raise FileNotFoundError(f"No checkpoint weight files found under: {root}")

    counts: list[tuple[str, int]] = []
    for path in weight_files:
        rel = str(path.relative_to(root))
        if path.suffix == ".safetensors":
            count = count_safetensors(path)
        else:
            if not allow_pickle:
                print(
                    f"Skipping {rel}: pass --allow-pickle to count .bin/.pt/.pth files.",
                    file=sys.stderr,
                )
                continue
            count = count_torch_checkpoint(path)
        counts.append((rel, count))

    if not counts:
        raise RuntimeError("No checkpoint files were counted.")
    return sum(count for _, count in counts), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", help="Hugging Face model repo id or local policy directory.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--allow-pickle", action="store_true", help="Allow loading .bin/.pt/.pth checkpoints.")
    parser.add_argument("--files", action="store_true", help="Also print per-file counts.")
    args = parser.parse_args()

    total, counts = count_policy(args.policy, args.revision, args.force_download, args.allow_pickle)
    print(f"Policy: {args.policy}")
    print(f"Parameters: {format_count(total)}")

    if args.files:
        for rel, count in counts:
            print(f"  {rel}: {format_count(count)}")


if __name__ == "__main__":
    main()
