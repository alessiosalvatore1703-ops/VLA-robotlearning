#!/usr/bin/env bash
# Remote execution script for SmolVLA fine-tuning.
# Invoked by orchestrate.py via: brev exec <instance> @training/remote_train.sh
#
# Expects /tmp/.lerobot_env to exist on the instance (uploaded by orchestrate.py)
# with the following exports:
#   HF_TOKEN, DATASET_REPO_ID, OUTPUT_REPO_ID, TRAIN_STEPS, BATCH_SIZE, WANDB_ENABLE
set -euo pipefail

LEROBOT_DIR="$HOME/lerobot"
export OUTPUT_DIR="$HOME/outputs/smolvla"

# ── 1. Load credentials ────────────────────────────────────────────────────────
source /tmp/.lerobot_env

echo ""
echo "============================================================"
echo " SmolVLA Fine-Tuning Pipeline"
echo "============================================================"
echo " Dataset : $DATASET_REPO_ID"
echo " Output  : $OUTPUT_REPO_ID"
echo " Steps   : $TRAIN_STEPS  |  Batch: $BATCH_SIZE  |  WandB: $WANDB_ENABLE"
echo "============================================================"
echo ""

# ── 2. Install miniforge silently ─────────────────────────────────────────────
echo "==> [1/6] Installing miniforge (Python environment manager)..."
MINIFORGE_INSTALLER="/tmp/miniforge_install.sh"
wget -q \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
  -O "$MINIFORGE_INSTALLER"
bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
rm "$MINIFORGE_INSTALLER"

# Activate conda for this non-interactive shell session
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda config --set always_yes true   # skip all "Proceed ([y]/n)?" prompts

# ── 3. Create isolated Python 3.12 environment ────────────────────────────────
echo "==> [2/6] Creating 'lerobot' conda environment (Python 3.12)..."
conda create -n lerobot python=3.12
conda activate lerobot

# ffmpeg is required by TorchCodec for LeRobot video decoding
conda install ffmpeg -c conda-forge

# ── 4. Install LeRobot + SmolVLA VLA dependencies ─────────────────────────────
echo "==> [3/6] Installing LeRobot with SmolVLA extras..."
git clone --depth=1 https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
cd "$LEROBOT_DIR"
# [smolvla] installs the vision-language-action model dependencies
pip install --quiet -e ".[smolvla]"

# ── 5. Authenticate with Hugging Face (non-interactive) ───────────────────────
echo "==> [4/6] Authenticating with Hugging Face Hub..."
huggingface-cli login --token "$HF_TOKEN"

# ── 6. Fine-tune SmolVLA ──────────────────────────────────────────────────────
echo "==> [5/6] Starting fine-tuning on $DATASET_REPO_ID..."
mkdir -p "$OUTPUT_DIR"

cd "$LEROBOT_DIR"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --batch_size="$BATCH_SIZE" \
  --steps="$TRAIN_STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_finetuning \
  --policy.device=cuda \
  --wandb.enable="$WANDB_ENABLE"

echo "==> Fine-tuning complete."

# ── 7. Upload checkpoint to Hugging Face Hub ──────────────────────────────────
echo "==> [6/6] Uploading checkpoint to ${OUTPUT_REPO_ID}..."

# Use Python for the upload so we get reliable repo creation + folder upload.
# <<'PYEOF' (quoted) prevents bash from expanding $variables inside the heredoc;
# the Python code reads everything it needs from the environment instead.
python3 - <<'PYEOF'
import os
from huggingface_hub import HfApi

output_dir  = os.environ["OUTPUT_DIR"]
output_repo = os.environ["OUTPUT_REPO_ID"]
token       = os.environ["HF_TOKEN"]

api = HfApi(token=token)
api.create_repo(output_repo, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=output_dir,
    repo_id=output_repo,
    repo_type="model",
)
print(f"Checkpoint available at: https://huggingface.co/{output_repo}")
PYEOF

# ── 8. Remove credentials from the instance ───────────────────────────────────
rm -f /tmp/.lerobot_env
echo ""
echo "==> All done. Credentials removed from instance."
