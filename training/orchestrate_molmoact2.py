#!/usr/bin/env python3
"""
Local orchestrator for MolmoAct2 fine-tuning on a Brev H100 instance.

It provisions a Brev instance, writes credentials/settings to
/tmp/.molmoact2_env, runs setup_and_train_molmoact2.sh remotely, and deletes the
instance on completion or failure unless --keep-instance is set.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()

DEFAULT_INSTANCE_TYPE = "gpu-h100-sxm.1gpu-16vcpu-200gb"
CREATE_MAX_RETRIES = 3
CREATE_RETRY_DELAY_S = 30
POLL_INTERVAL_S = 20
READY_TIMEOUT_S = 900
POST_RUNNING_GRACE_S = 30
EXEC_MAX_RETRIES = 3
EXEC_RETRY_DELAY_S = 30


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, capture_output=capture, text=capture)


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
    raise TimeoutError(f"'{instance_name}' did not reach RUNNING within {READY_TIMEOUT_S}s.")


def ssh_exec_script(instance_name: str, script_path: str, *, retries: int = 1) -> None:
    script = Path(script_path).read_text()
    ssh_cmd = [
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=120",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=30",
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
                f"  ssh exec failed (exit {result.returncode}). Retrying in "
                f"{EXEC_RETRY_DELAY_S}s (attempt {attempt + 1}/{retries})...",
                flush=True,
            )
            time.sleep(EXEC_RETRY_DELAY_S)
    raise RuntimeError(f"ssh exec failed {retries} time(s) for '{instance_name}'.")


def brev_refresh() -> None:
    print("\nRefreshing brev SSH config...", flush=True)
    result = brev(["refresh"], check=False)
    if result.returncode != 0:
        print("  Warning: brev refresh exited non-zero.", flush=True)


def create_instance(instance_name: str, instance_type: str) -> None:
    for attempt in range(1, CREATE_MAX_RETRIES + 1):
        print(
            f"\n=== Provisioning Brev instance '{instance_name}' "
            f"(type: {instance_type}) - attempt {attempt}/{CREATE_MAX_RETRIES} ===",
            flush=True,
        )
        result = brev(["create", instance_name, "--type", instance_type], check=False)
        if result.returncode == 0:
            return
        print("  brev create failed. Checking whether the instance was queued...", flush=True)
        time.sleep(10)
        if _instance_exists(instance_name):
            print(f"  Instance '{instance_name}' found in brev ls.", flush=True)
            return
        if attempt < CREATE_MAX_RETRIES:
            time.sleep(CREATE_RETRY_DELAY_S)
    raise RuntimeError(f"brev create failed and '{instance_name}' never appeared in brev ls.")


def delete_instance(instance_name: str) -> None:
    print(f"\n=== Deleting instance '{instance_name}' to stop billing ===")
    brev(["delete", instance_name], check=False)


def shell_export(name: str, value: object) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end MolmoAct2 SO100/SO101 fine-tuning on a Brev H100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-repo-id", default="ETHrobotlearning/task3-TOY-clean")
    p.add_argument("--output-repo-id", required=True)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--policy-checkpoint-path", default="allenai/MolmoAct2-SO100_101")
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-freq", type=int, default=5000)
    p.add_argument("--action-chunk-size", type=int, default=10)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--image-keys", default='["observation.images.front"]')
    p.add_argument("--setup-type", default="single SO-100/SO-101 arm with one front RGB camera")
    p.add_argument("--control-mode", default="absolute joint pose")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--job-name", default="molmoact2_so101_task3_toy_lora_50k")
    p.add_argument("--output-dir", default="")
    p.add_argument("--instance-name", default="molmoact2-training")
    p.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    p.add_argument("--wandb-enable", action="store_true", default=True)
    p.add_argument("--wandb-disable", action="store_false", dest="wandb_enable")
    p.add_argument("--wandb-api-key", default=os.environ.get("WANDB_API_KEY"))
    p.add_argument("--wandb-project", default="molmoact2")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--keep-instance", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        sys.exit("Error: pass --hf-token TOKEN or set HF_TOKEN.")
    if args.wandb_enable and not args.wandb_api_key:
        sys.exit("Error: --wandb-enable requires --wandb-api-key or WANDB_API_KEY.")

    remote_script = HERE / "setup_and_train_molmoact2.sh"
    if not remote_script.exists():
        sys.exit(f"Error: remote script not found: {remote_script}")

    cred_lines = [
        shell_export("HF_TOKEN", args.hf_token),
        shell_export("DATASET_REPO_ID", args.dataset_repo_id),
        shell_export("OUTPUT_REPO_ID", args.output_repo_id),
        shell_export("POLICY_CHECKPOINT_PATH", args.policy_checkpoint_path),
        shell_export("TRAIN_STEPS", args.train_steps),
        shell_export("BATCH_SIZE", args.batch_size),
        shell_export("SAVE_FREQ", args.save_freq),
        shell_export("ACTION_CHUNK_SIZE", args.action_chunk_size),
        shell_export("N_ACTION_STEPS", args.n_action_steps),
        shell_export("IMAGE_KEYS", args.image_keys),
        shell_export("SETUP_TYPE", args.setup_type),
        shell_export("CONTROL_MODE", args.control_mode),
        shell_export("TRAIN_ACTION_EXPERT_ONLY", "false"),
        shell_export("ENABLE_LORA_VLM", "true"),
        shell_export("ENABLE_LORA_ACTION_EXPERT", "false"),
        shell_export("LORA_RANK", args.lora_rank),
        shell_export("LORA_ALPHA", args.lora_alpha),
        shell_export("LORA_DROPOUT", args.lora_dropout),
        shell_export("JOB_NAME", args.job_name),
        shell_export("WANDB_ENABLE", "true" if args.wandb_enable else "false"),
        shell_export("WANDB_PROJECT", args.wandb_project),
        shell_export("WANDB_ENTITY", args.wandb_entity),
    ]
    if args.output_dir:
        cred_lines.append(shell_export("OUTPUT_DIR", args.output_dir))
    if args.wandb_api_key:
        cred_lines.append(shell_export("WANDB_API_KEY", args.wandb_api_key))

    preamble_path: str | None = None
    try:
        create_instance(args.instance_name, args.instance_type)
        wait_for_running(args.instance_name)
        print(f"Waiting {POST_RUNNING_GRACE_S}s for SSH daemon to become responsive...")
        time.sleep(POST_RUNNING_GRACE_S)
        brev_refresh()

        print("\n=== Writing MolmoAct2 settings to instance ===")
        preamble_fd, preamble_path = tempfile.mkstemp(prefix=".molmoact2_setup_", suffix=".sh")
        with os.fdopen(preamble_fd, "w") as pf:
            pf.write("#!/bin/bash\nset -euo pipefail\n")
            pf.write("cat > /tmp/.molmoact2_env << 'CREDEOF'\n")
            pf.write("\n".join(cred_lines) + "\n")
            pf.write("CREDEOF\n")
            pf.write("chmod 600 /tmp/.molmoact2_env\n")
        ssh_exec_script(args.instance_name, preamble_path, retries=EXEC_MAX_RETRIES)
        os.unlink(preamble_path)
        preamble_path = None

        print("\n=== Starting remote MolmoAct2 training ===")
        ssh_exec_script(args.instance_name, str(remote_script), retries=1)
        print("\n=== Training completed successfully ===")

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        if not args.keep_instance:
            delete_instance(args.instance_name)
        if preamble_path:
            os.unlink(preamble_path)
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if not args.keep_instance:
            print("Tearing down instance to stop billing...", file=sys.stderr)
            delete_instance(args.instance_name)
        if preamble_path:
            os.unlink(preamble_path)
        sys.exit(1)

    if not args.keep_instance:
        delete_instance(args.instance_name)


if __name__ == "__main__":
    main()
