#!/usr/bin/env python3
"""
Local orchestrator: provisions a Brev H100 instance, runs SmolVLA fine-tuning
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
        [--train-steps 30000]
        [--batch-size 32]
        [--save-freq 10000]
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

# gpu-h100-sxm.1gpu-16vcpu-200gb: Nebius H100 SXM (80 GB VRAM), 16 vCPU, $3.54/hr, stoppable
INSTANCE_TYPE = "gpu-h100-sxm.1gpu-16vcpu-200gb"

CREATE_MAX_RETRIES = 3      # retries only if the instance never appears in brev ls
CREATE_RETRY_DELAY_S = 30   # seconds before retrying a create that left no trace

POLL_INTERVAL_S = 20        # seconds between brev ls polls
READY_TIMEOUT_S = 900       # max seconds to wait for RUNNING state (Brev can take ~7-10 min)
POST_RUNNING_GRACE_S = 30   # extra buffer after RUNNING before SSH is responsive

EXEC_MAX_RETRIES = 3        # retries for brev exec on transient SSH timeouts
EXEC_RETRY_DELAY_S = 30     # seconds before retrying a failed brev exec


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



def _instance_exists(instance_name: str) -> bool:
    result = brev(["ls", "--json"], capture=True, check=False)
    try:
        data = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return False
    instances = data if isinstance(data, list) else data.get("instances", [])
    return any(inst.get("name") == instance_name for inst in instances)


def wait_for_running(instance_name: str) -> None:
    """Poll `brev ls --json` until the instance is RUNNING. brev exec does NOT
    do this on its own — it just times out the SSH connect."""
    print(f"\nPolling for '{instance_name}' RUNNING state (timeout {READY_TIMEOUT_S}s)...")
    deadline = time.time() + READY_TIMEOUT_S
    last_status = None
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
                if status != last_status:
                    print(f"  {instance_name}: {status}", flush=True)
                    last_status = status
                if status == "RUNNING":
                    return
                if any(kw in status for kw in ("ERROR", "FAIL", "TERMINAT", "DELET")):
                    raise RuntimeError(f"Instance entered non-recoverable state: {status}")
        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(
        f"'{instance_name}' did not reach RUNNING within {READY_TIMEOUT_S}s. "
        "Run `brev ls` to inspect."
    )


def ssh_exec_script(instance_name: str, script_path: str, *, retries: int = 1) -> None:
    """Execute a local shell script on the remote instance via direct ssh.

    Bypasses `brev exec` to work around two macOS-specific bugs in its SSH
    wrapper:
      * stale hostname in brev's SSH config after instance recreation, and
      * Unix-socket path-length limit (~104 chars) hit by brev's ControlPath
        template, which includes the full EC2 hostname.

    Brev sets up an SSH alias matching the instance name; we pass our own
    options to disable ControlMaster (avoids the path issue) and add
    keepalives so long-running execs survive brief network blips.
    """
    with open(script_path, "r") as f:
        script = f.read()

    ssh_cmd = [
        "ssh",
        "-o", "ControlMaster=no",
        "-o", "ControlPath=none",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=120",     # tolerate ~1 h of silence
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=30",
        instance_name,
        "bash -s",
    ]

    for attempt in range(1, retries + 1):
        print(f"  $ ssh {instance_name} 'bash -s'  (script piped via stdin)", flush=True)
        result = subprocess.run(ssh_cmd, input=script, text=True, check=False)
        if result.returncode == 0:
            return
        if attempt < retries:
            print(
                f"  ssh exec failed (exit {result.returncode}). "
                f"Retrying in {EXEC_RETRY_DELAY_S}s (attempt {attempt + 1}/{retries})...",
                flush=True,
            )
            time.sleep(EXEC_RETRY_DELAY_S)

    raise RuntimeError(
        f"ssh exec failed {retries} time(s) for '{instance_name}'."
    )


def brev_refresh() -> None:
    """Refresh the local brev SSH config so aliases point to current hostnames."""
    print("\nRefreshing brev SSH config...", flush=True)
    result = brev(["refresh"], check=False)
    if result.returncode != 0:
        print(
            "  Warning: brev refresh exited non-zero — SSH alias may use a stale hostname.",
            flush=True,
        )


def create_instance(instance_name: str, instance_type: str) -> None:
    """
    Issue `brev create` and tolerate a transient EOF/timeout on the HTTP response.

    The Brev control plane sometimes drops the connection before sending a reply
    even though the instance was queued successfully.  After any CLI failure we
    wait briefly and then check `brev ls` — if the instance is already there we
    proceed straight to wait_for_running.  We only retry the create command if
    the instance never appeared in the listing.
    """
    for attempt in range(1, CREATE_MAX_RETRIES + 1):
        print(
            f"\n=== Provisioning Brev instance '{instance_name}' "
            f"(type: {instance_type}) — attempt {attempt}/{CREATE_MAX_RETRIES} ===",
            flush=True,
        )
        result = brev(["create", instance_name, "--type", instance_type], check=False)
        if result.returncode == 0:
            return

        # CLI exited non-zero — check whether the instance was created anyway.
        print(
            f"  brev create exited {result.returncode}. "
            "Checking brev ls to see if the instance was queued...",
            flush=True,
        )
        time.sleep(10)  # give the control plane a moment to register the new instance
        if _instance_exists(instance_name):
            print(
                f"  Instance '{instance_name}' found in brev ls — "
                "provisioning is underway despite the CLI error.",
                flush=True,
            )
            return  # wait_for_running will poll until RUNNING

        if attempt < CREATE_MAX_RETRIES:
            print(f"  Instance not found. Retrying in {CREATE_RETRY_DELAY_S}s...", flush=True)
            time.sleep(CREATE_RETRY_DELAY_S)

    raise RuntimeError(
        f"brev create failed {CREATE_MAX_RETRIES} time(s) and '{instance_name}' "
        "never appeared in `brev ls`. Check Brev's status page."
    )


def delete_instance(instance_name: str) -> None:
    print(f"\n=== Deleting instance '{instance_name}' to stop billing ===")
    brev(["delete", instance_name], check=False)


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end SmolVLA fine-tuning pipeline on a Brev H100 instance.",
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
        default=30000,
        help="Number of fine-tuning gradient steps.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size. Reduce if the instance runs out of memory.",
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save and push a separate Hub model repo every N training steps.",
    )
    p.add_argument(
        "--action-chunk-size",
        type=int,
        default=10,
        help="SmolVLA policy.chunk_size.",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="SmolVLA policy.n_action_steps.",
    )
    p.add_argument(
        "--vlm-num-layers",
        type=int,
        default=16,
        help="SmolVLA policy.num_vlm_layers.",
    )
    p.add_argument(
        "--job-name",
        default="smolvla_eval2_topview_chunk10_30k",
        help="LeRobot/W&B job name.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Remote output directory. Defaults to ~/outputs/train/$JOB_NAME in remote_train.sh.",
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

    # Build credentials lines; reused both to write a local temp file (for
    # cleanup) and to embed into the remote preamble script.
    # Use single quotes so values containing double-quotes don't break the
    # shell assignment.  Single quotes are safe here because none of these
    # values (HF tokens, wandb keys, repo ids) can contain a literal ' char.
    cred_lines = [
        f"export HF_TOKEN='{args.hf_token}'",
        f"export DATASET_REPO_ID='{args.dataset_repo_id}'",
        f"export OUTPUT_REPO_ID='{args.output_repo_id}'",
        f"export TRAIN_STEPS='{args.train_steps}'",
        f"export BATCH_SIZE='{args.batch_size}'",
        f"export SAVE_FREQ='{args.save_freq}'",
        f"export ACTION_CHUNK_SIZE='{args.action_chunk_size}'",
        f"export N_ACTION_STEPS='{args.n_action_steps}'",
        f"export VLM_NUM_LAYERS='{args.vlm_num_layers}'",
        f"export JOB_NAME='{args.job_name}'",
        f"export WANDB_ENABLE='{'true' if args.wandb_enable else 'false'}'",
    ]
    if args.output_dir:
        cred_lines.append(f"export OUTPUT_DIR='{args.output_dir}'")
    if args.wandb_api_key:
        cred_lines.append(f"export WANDB_API_KEY='{args.wandb_api_key}'")

    # Local sentinel so the except block can always clean up.
    preamble_path: str | None = None
    try:
        # ── 1. Provision instance ─────────────────────────────────────────
        create_instance(instance_name, INSTANCE_TYPE)

        # ── 2. Wait until the instance is RUNNING ─────────────────────────
        wait_for_running(instance_name)
        print(f"Waiting {POST_RUNNING_GRACE_S}s for SSH daemon to become responsive...")
        time.sleep(POST_RUNNING_GRACE_S)

        # ── 3. Refresh brev's SSH config to pick up the new hostname ──────
        # brev exec sometimes uses a stale hostname from a previous instance
        # with the same name.  `brev refresh` rewrites ~/.ssh/config.
        brev_refresh()

        # ── 4. Write credentials to the instance via direct ssh ───────────
        # We avoid `brev exec` because its ControlMaster Unix-socket path
        # exceeds macOS's ~104-char limit.  Direct ssh with ControlMaster=no
        # sidesteps the issue entirely.
        print("\n=== Writing credentials to instance ===")
        preamble_fd, preamble_path = tempfile.mkstemp(prefix=".cred_setup_", suffix=".sh")
        with os.fdopen(preamble_fd, "w") as pf:
            pf.write("#!/bin/bash\nset -euo pipefail\n")
            pf.write("cat > /tmp/.lerobot_env << 'CREDEOF'\n")
            pf.write("\n".join(cred_lines) + "\n")
            pf.write("CREDEOF\n")
            pf.write("chmod 600 /tmp/.lerobot_env\n")
        ssh_exec_script(instance_name, preamble_path, retries=EXEC_MAX_RETRIES)
        os.unlink(preamble_path)
        preamble_path = None

        # ── 5. Execute remote training script (blocks until complete) ─────
        # ~1 hour for 20 k steps on an H100 at batch 128. Ensure a stable network
        # connection, or run orchestrate.py inside a local tmux session.
        # No retry here — restarting a multi-hour training run from scratch
        # on a transient disconnect is rarely what we want.
        print("\n=== Starting remote training (this will take several hours) ===")
        ssh_exec_script(instance_name, str(remote_script), retries=1)

        print("\n=== Training and upload completed successfully ===")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Tearing down instance...", file=sys.stderr)
        delete_instance(instance_name)
        if preamble_path:
            os.unlink(preamble_path)
        sys.exit(130)

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Tearing down instance to stop billing...", file=sys.stderr)
        delete_instance(instance_name)
        if preamble_path:
            os.unlink(preamble_path)
        sys.exit(1)

    # Success path: delete instance.
    delete_instance(instance_name)


if __name__ == "__main__":
    main()
