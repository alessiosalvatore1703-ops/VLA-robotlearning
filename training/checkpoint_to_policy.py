#!/usr/bin/env python3
"""
Convert a lerobot checkpoint stored on HuggingFace into a standalone policy repo.

Usage:
    python checkpoint_to_policy.py \
        --source Alessio03/smolvla-task2-colors \
        --checkpoint 005000 \
        --output Alessio03/smolvla-task2-colors-final \
        --token hf_...

The source repo is the one with checkpoints/ folders (created by setup_and_train.sh).
The output repo will be a clean model loadable with --policy.pretrained_path=<output>.
"""

import argparse
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# Enable fast Rust-based transfers if hf_transfer is installed
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def parse_repo_id(value: str) -> str:
    """Accept either 'owner/repo' or a full HuggingFace URL."""
    if value.startswith("https://huggingface.co/"):
        value = value.removeprefix("https://huggingface.co/")
    return value.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description="Convert a checkpoint to a standalone HuggingFace policy repo.")
    parser.add_argument("--source", required=True, help="Repo containing checkpoints (e.g. Alessio03/smolvla-task2-colors) OR local path to checkpoint folder")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint step to convert (e.g. 005000)")
    parser.add_argument("--output", required=True, help="Output repo for the policy (e.g. Alessio03/smolvla-task2-colors-final)")
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    args = parser.parse_args()

    source = args.source
    output = parse_repo_id(args.output)
    step = args.checkpoint.zfill(6)

    api = HfApi(token=args.token)

    # Determine if source is a local path or a HuggingFace repo
    source_path = Path(source)
    if source_path.exists():
        # Local path: skip download, resolve pretrained_model/ if needed
        print(f"Using local source: {source_path}")
        model_dir = source_path / "pretrained_model"
        if not model_dir.exists():
            model_dir = source_path
        _tmp_ctx = None
    else:
        source = parse_repo_id(source)
        _tmp_ctx = tempfile.TemporaryDirectory()
        tmp = _tmp_ctx.name
        print(f"Downloading checkpoint {step} from {source}...")
        print(f"  Tip: run this script on the instance with --source ~/outputs/smolvla/checkpoints/{step} to skip this download.")
        local_dir = snapshot_download(
            repo_id=source,
            repo_type="model",
            allow_patterns=f"checkpoints/{step}/*",
            local_dir=tmp,
            token=args.token,
        )
        model_dir = Path(local_dir) / "checkpoints" / step / "pretrained_model"
        if not model_dir.exists():
            model_dir = Path(local_dir) / "checkpoints" / step
            print(f"  No pretrained_model/ subfolder found, using checkpoint root.")

    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(f"No model files found at {model_dir}")

    print(f"  Model dir: {model_dir}")
    print(f"\nCreating output repo {output}...")
    api.create_repo(repo_id=output, repo_type="model", exist_ok=True)

    print(f"Pushing policy to {output}...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=output,
        repo_type="model",
        commit_message=f"Policy from checkpoint {step}",
    )

    if _tmp_ctx:
        _tmp_ctx.cleanup()

    print(f"\nDone.")
    print(f"  HuggingFace : https://huggingface.co/{output}")
    print(f"  Load with   : --policy.pretrained_path={output}")


if __name__ == "__main__":
    main()
