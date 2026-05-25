#!/usr/bin/env python3
"""Count a LeRobot policy checkpoint's parameters, then run a rollout.

Typical use:

    python inference/rollout_with_param_count.py \
      --policy-path Alessio03/smolvla-task3-50k \
      --task "Place the coke on Yann LeCun." \
      --duration 40 \
      --device mps

The parameter count is computed from checkpoint tensors. For standard LeRobot
policies this is the model checkpoint, usually ``model.safetensors``.
"""

import argparse
import math
import os
import signal
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_FOLLOWER_PORT = "/dev/tty.usbmodem5B140319121"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
DEFAULT_CAMERA_CONFIG = "{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10}}"

WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
]

SKIP_WEIGHT_NAMES = {
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
    lowered = path.name.lower()
    return any(token in lowered for token in SKIP_WEIGHT_NAMES)


def find_local_weight_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        files.extend(root.rglob(pattern))
    return sorted(path for path in files if path.is_file() and not should_skip_weight_file(path))


def download_policy_weights(policy_path: str, revision: str | None, force_download: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=policy_path,
            repo_type="model",
            revision=revision,
            force_download=force_download,
            allow_patterns=WEIGHT_PATTERNS + ["config.json", "*.yaml", "*.md"],
        )
    )


def tensor_numel_from_shape(shape: list[int] | tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(math.prod(int(dim) for dim in shape))


def count_safetensors(path: Path) -> int:
    from safetensors import safe_open

    total = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            # get_slice reads metadata/shape without materializing the full tensor.
            tensor_slice = handle.get_slice(key)
            total += tensor_numel_from_shape(tensor_slice.get_shape())
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


def count_pickle_checkpoint(path: Path) -> int:
    import torch

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")

    return sum(int(tensor.numel()) for tensor in iter_tensors(checkpoint))


def count_checkpoint_parameters(
    policy_path: str,
    revision: str | None,
    force_download: bool,
    allow_pickle_checkpoints: bool,
) -> tuple[int, list[tuple[str, int]]]:
    if is_hf_repo_id(policy_path):
        root = download_policy_weights(policy_path, revision=revision, force_download=force_download)
    else:
        root = Path(policy_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Local policy path does not exist: {root}")

    weight_files = find_local_weight_files(root)
    if not weight_files:
        raise FileNotFoundError(f"No checkpoint files found under {root}")

    per_file: list[tuple[str, int]] = []
    for path in weight_files:
        rel = str(path.relative_to(root))
        if path.suffix == ".safetensors":
            count = count_safetensors(path)
        else:
            if not allow_pickle_checkpoints:
                print(
                    f"Skipping {rel}: PyTorch pickle checkpoints are disabled. "
                    "Pass --allow-pickle-checkpoints to count it.",
                    file=sys.stderr,
                )
                continue
            count = count_pickle_checkpoint(path)
        per_file.append((rel, count))

    if not per_file:
        raise RuntimeError(
            "Found checkpoint files, but none were counted. "
            "If this policy only has .bin/.pt files, rerun with --allow-pickle-checkpoints."
        )

    return sum(count for _, count in per_file), per_file


def build_rollout_command(args: argparse.Namespace, extra_rollout_args: list[str]) -> list[str]:
    command = [
        args.lerobot_rollout_cmd,
        "--strategy.type=base",
        f"--policy.path={args.policy_path}",
        f"--policy.device={args.device}",
        f"--robot.type={args.robot_type}",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        f"--robot.cameras={args.robot_cameras}",
        f"--task={args.task}",
        f"--duration={args.duration}",
        f"--display_data={str(args.display_data).lower()}",
        f"--fps={args.fps}",
        f"--return_to_initial_position={str(args.return_to_initial_position).lower()}",
    ]

    if args.policy_use_amp is not None:
        command.append(f"--policy.use_amp={str(args.policy_use_amp).lower()}")

    command.extend(extra_rollout_args)
    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Count parameters for a Hugging Face/local LeRobot policy checkpoint, "
            "then run a normal rollout with the requested prompt."
        )
    )
    parser.add_argument("--policy-path", required=True, help="HF model repo id or local policy directory.")
    parser.add_argument("--task", required=True, help="Language prompt for the rollout.")
    parser.add_argument("--device", default="mps", help="Policy device passed to lerobot-rollout.")
    parser.add_argument("--duration", type=float, default=20, help="Rollout duration in seconds.")
    parser.add_argument("--fps", type=int, default=10, help="Rollout/control FPS.")
    parser.add_argument("--robot-type", default="so101_follower")
    parser.add_argument("--robot-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--robot-cameras", default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--revision", default=None, help="Optional HF model revision.")
    parser.add_argument("--force-download", action="store_true", help="Force-refresh HF checkpoint metadata/files.")
    parser.add_argument(
        "--allow-pickle-checkpoints",
        action="store_true",
        help="Allow counting .bin/.pt/.pth checkpoints via torch.load. Safetensors are always supported.",
    )
    parser.add_argument(
        "--policy-use-amp",
        choices=["true", "false"],
        default=None,
        help="Optional --policy.use_amp override for lerobot-rollout.",
    )
    parser.add_argument(
        "--lerobot-rollout-cmd",
        default=os.environ.get("LEROBOT_ROLLOUT_CMD", "lerobot-rollout"),
        help="Rollout executable to run. Defaults to lerobot-rollout.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the rollout command without running it.")
    parser.add_argument("--skip-count", action="store_true", help="Run rollout without counting parameters first.")
    parser.add_argument(
        "--return-to-initial-position",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ask LeRobot to return the robot to its startup joint position during teardown. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-display-data",
        action="store_false",
        dest="display_data",
        help="Pass --display_data=false to lerobot-rollout.",
    )
    parser.set_defaults(display_data=True)

    args, extra = parser.parse_known_args()
    if args.policy_use_amp is not None:
        args.policy_use_amp = args.policy_use_amp == "true"
    return args, extra


class TerminationRequested(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Received signal {signum}")


def forward_signal_to_rollout(process: subprocess.Popen, signum: int) -> None:
    if process.poll() is not None:
        return

    if hasattr(os, "killpg"):
        os.killpg(process.pid, signum)
    else:
        process.send_signal(signum)


def stop_rollout_process(process: subprocess.Popen, reason: str) -> int:
    """Stop lerobot-rollout while giving it a chance to run hardware cleanup."""
    if process.poll() is not None:
        return process.returncode

    print(f"\nStopping lerobot-rollout ({reason}). Waiting for LeRobot cleanup...", file=sys.stderr)
    forward_signal_to_rollout(process, signal.SIGINT)

    try:
        return process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("lerobot-rollout did not exit after SIGINT; sending SIGTERM.", file=sys.stderr)
        forward_signal_to_rollout(process, signal.SIGTERM)

    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("lerobot-rollout still did not exit; sending SIGKILL.", file=sys.stderr)
        if hasattr(signal, "SIGKILL"):
            forward_signal_to_rollout(process, signal.SIGKILL)
        else:
            process.kill()
        return process.wait()


def run_rollout_command(command: list[str]) -> int:
    process: subprocess.Popen | None = None

    def handle_sigterm(signum: int, _frame: Any) -> None:
        raise TerminationRequested(signum)

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        # Put rollout in its own process group so the wrapper can reliably stop
        # subprocesses spawned by LeRobot while still letting LeRobot handle SIGINT.
        process = subprocess.Popen(command, start_new_session=True)
        return process.wait()
    except KeyboardInterrupt:
        if process is not None:
            stop_rollout_process(process, "Ctrl-C")
        return 130
    except TerminationRequested as exc:
        if process is not None:
            stop_rollout_process(process, f"signal {exc.signum}")
        return 128 + exc.signum
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    args, extra_rollout_args = parse_args()

    if not args.skip_count:
        total, per_file = count_checkpoint_parameters(
            args.policy_path,
            revision=args.revision,
            force_download=args.force_download,
            allow_pickle_checkpoints=args.allow_pickle_checkpoints,
        )
        print(f"Policy: {args.policy_path}")
        print(f"Checkpoint tensor count: {format_count(total)}")
        for rel, count in per_file:
            print(f"  {rel}: {format_count(count)}")
        print()

    command = build_rollout_command(args, extra_rollout_args)
    print("Rollout command:")
    print(shlex.join(command))

    if args.dry_run:
        return

    raise SystemExit(run_rollout_command(command))


if __name__ == "__main__":
    main()
