#!/usr/bin/env python3
"""Print parameter count of a HuggingFace SmolVLA policy, then run lerobot-rollout."""

import argparse
import subprocess
import sys

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def fmt(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HuggingFace repo id of the policy")
    args = parser.parse_args()

    policy = SmolVLAPolicy.from_pretrained(args.model)

    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"Policy: {args.model}")
    print(f"Total parameters    : {total:,} ({fmt(total)})")
    print(f"Trainable parameters: {trainable:,} ({fmt(trainable)})")

    cmd = [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={args.model}",
        "--policy.device=mps",
        "--robot.type=so101_follower",
        "--robot.port=/dev/tty.usbmodem5B140319121",
        "--robot.id=my_awesome_follower_arm",
        "--robot.cameras={ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10}}",
        "--task=Put the banana into the bowl that is not red and not green.",
        "--duration=20",
        "--display_data=true",
        "--fps=10",
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
