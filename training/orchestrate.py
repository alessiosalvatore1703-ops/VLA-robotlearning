#!/usr/bin/env python3
"""
Local orchestrator: provisions a Brev A10G instance, runs SmolVLA fine-tuning
remotely, waits for the checkpoint to be pushed to HF Hub, then deletes the
instance to stop billing — automatically, on success or failure.

Prerequisites:
    brev login          # authenticate the Brev CLI once
    pip install brev    # if not already installed

Usage:
    python training/orchestrate.py \
        --dataset-repo-id USERNAME/my-lerobot-dataset \
        --output-repo-id  USERNAME/my-smolvla \
        [--hf-token TOKEN]          # or set $HF_TOKEN
        [--train-steps 20000]
        [--batch-size 64]
        [--instance-name smolvla-training]
        [--wandb-enable]

The script never prints the HF token to stdout; it is written to a local
temp file and copied to the instance over SCP, then deleted remotely
after training completes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# g5.xlarge: 1x NVIDIA A10G (24 GB VRAM), 4 vCPU, 16 GB RAM, 125 GB SSD
INSTANCE_TYPE = "g5.xlarge"

POLL_INTERVAL_S = 30       # seconds between brev ls polls
READY_TIMEOUT_S = 600      # max seconds to wait for RUNNING state
POST_RUNNING_GRACE_S = 15  # extra buffer after RUNNING before SSH is responsive


# ── helpers ───────────────────────────────────────────────────────────────────


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=capture,
    )


def brev(args: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return _run(["brev"] + args, check=check, capture=capture)


def wait_for_running(instance_name: str) -> None:
    print(f"\nPolling for '{instance_name}' RUNNING state (timeout {READY_TIMEOUT_S}s)...")
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        result = brev(["ls", "--json"], capture=True, check=False)
        try:
            data = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            data = []

        instances = data if isinstance(data, list) else data.get("instances", [])
        for inst in instances:
            if inst.get("name") == instance_name:
                status = str(inst.get("status", inst.get("state", "UNKNOWN"))).upper()
                print(f"  {instance_name}: {status}", flush=True)
                if status == "RUNNING":
                    return
                if any(kw in status for kw in ("ERROR", "FAIL", "TERMINAT", "DELET")):
                    raise RuntimeError(f"Instance entered non-recoverable state: {status}")

        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(
        f"'{instance_name}' did not reach RUNNING within {READY_TIMEOUT_S}s. "
        "Run `brev ls` to inspect."
    )


def delete_instance(instance_name: str) -> None:
    print(f"\n=== Deleting instance '{instance_name}' to stop billing ===")
    brev(["delete", instance_name], check=False)


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end SmolVLA fine-tuning pipeline on a Brev A10G instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-repo-id",
        required=True,
        help="HF Hub dataset repo id used for training (e.g. username/my-dataset).",
    )
    p.add_argument(
        "--output-repo-id",
        required=True,
        help="HF Hub model repo where the checkpoint will be pushed (e.g. username/my-smolvla).",
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Defaults to $HF_TOKEN.",
    )
    p.add_argument(
        "--train-steps",
        type=int,
        default=20000,
        help="Number of fine-tuning gradient steps (~4 h on A10G at 20 k steps).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size. Reduce if OOM.",
    )
    p.add_argument(
        "--instance-name",
        default="smolvla-training",
        help="Name for the Brev instance.",
    )
    p.add_argument(
        "--wandb-enable",
        action="store_true",
        help="Enable Weights & Biases experiment tracking on the remote instance.",
    )
    p.add_argument(
        "--wandb-api-key",
        default=os.environ.get("WANDB_API_KEY"),
        help="Weights & Biases API key. Defaults to $WANDB_API_KEY. Required when --wandb-enable is set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.hf_token:
        sys.exit(
            "Error: a Hugging Face token is required.\n"
            "Pass --hf-token TOKEN or set the HF_TOKEN environment variable."
        )

    if args.wandb_enable and not args.wandb_api_key:
        sys.exit(
            "Error: --wandb-enable requires a Weights & Biases API key.\n"
            "Pass --wandb-api-key KEY or set the WANDB_API_KEY environment variable.\n"
            "Find your key at https://wandb.ai/settings"
        )

    instance_name = args.instance_name
    remote_script = HERE / "remote_train.sh"

    if not remote_script.exists():
        sys.exit(f"Error: remote training script not found at {remote_script}")

    # Write secrets to a local temp file — never pass them as CLI arguments.
    env_fd, env_path = tempfile.mkstemp(prefix=".lerobot_creds_", suffix=".env")
    try:
        with os.fdopen(env_fd, "w") as f:
            f.write(f'export HF_TOKEN="{args.hf_token}"\n')
            f.write(f'export DATASET_REPO_ID="{args.dataset_repo_id}"\n')
            f.write(f'export OUTPUT_REPO_ID="{args.output_repo_id}"\n')
            f.write(f'export TRAIN_STEPS="{args.train_steps}"\n')
            f.write(f'export BATCH_SIZE="{args.batch_size}"\n')
            f.write(f'export WANDB_ENABLE="{"true" if args.wandb_enable else "false"}"\n')
            if args.wandb_api_key:
                f.write(f'export WANDB_API_KEY="{args.wandb_api_key}"\n')

        # ── 1. Provision instance ─────────────────────────────────────────
        print(f"\n=== Provisioning Brev instance '{instance_name}' (type: {INSTANCE_TYPE}) ===")
        brev(["create", instance_name, "--type", INSTANCE_TYPE])

        # ── 2. Wait until SSH is reachable ────────────────────────────────
        wait_for_running(instance_name)
        print(f"Waiting {POST_RUNNING_GRACE_S}s for SSH daemon to become responsive...")
        time.sleep(POST_RUNNING_GRACE_S)

        # ── 3. Upload credentials file ────────────────────────────────────
        print("\n=== Uploading credentials ===")
        brev(["copy", env_path, f"{instance_name}:/tmp/.lerobot_env"])

        # ── 4. Execute remote training script (blocks until complete) ─────
        # ~4 hours for 20 k steps on an A10G. Ensure a stable network
        # connection, or run orchestrate.py inside a local tmux session.
        print("\n=== Starting remote training (this will take several hours) ===")
        brev(["exec", instance_name, f"@{remote_script}"])

        print("\n=== Training and upload completed successfully ===")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Tearing down instance...", file=sys.stderr)
        delete_instance(instance_name)
        os.unlink(env_path)
        sys.exit(130)

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Tearing down instance to stop billing...", file=sys.stderr)
        delete_instance(instance_name)
        os.unlink(env_path)
        sys.exit(1)

    # Success path: delete instance and temp file.
    delete_instance(instance_name)
    os.unlink(env_path)


if __name__ == "__main__":
    main()
