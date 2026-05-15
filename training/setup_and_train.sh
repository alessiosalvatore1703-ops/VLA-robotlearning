#!/usr/bin/env bash
# Run this script on a fresh Brev H100 instance after SSHing in.
#
# Usage:
#   brev create my-instance --type gpu-h100-sxm.1gpu-16vcpu-200gb
#   brev shell my-instance
#   bash setup_and_train.sh
#
set -euo pipefail

# ── CONFIG — fill these in before running ─────────────────────────────────────
HF_TOKEN=""
DATASET_REPO_ID="ETHrobotlearning/task2-colors"
OUTPUT_REPO_ID="Alessio03/smolvla-colors"
TRAIN_STEPS=20000
BATCH_SIZE=512
WANDB_ENABLE="false"      # set to "true" to enable W&B logging
WANDB_API_KEY=""          # required when WANDB_ENABLE="true"
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: Set HF_TOKEN at the top of this script before running."
  exit 1
fi
if [ "$WANDB_ENABLE" = "true" ] && [ -z "$WANDB_API_KEY" ]; then
  echo "ERROR: Set WANDB_API_KEY at the top of this script when WANDB_ENABLE=true."
  exit 1
fi

LEROBOT_DIR="$HOME/lerobot"
OUTPUT_DIR="$HOME/outputs/smolvla"

echo ""
echo "============================================================"
echo " SmolVLA Fine-Tuning Pipeline"
echo "============================================================"
echo " Dataset : $DATASET_REPO_ID"
echo " Output  : $OUTPUT_REPO_ID"
echo " Steps   : $TRAIN_STEPS  |  Batch: $BATCH_SIZE  |  WandB: $WANDB_ENABLE"
echo "============================================================"
echo ""

# ── 1. Install miniforge ──────────────────────────────────────────────────────
echo "==> [1/5] Installing miniforge..."
MINIFORGE_INSTALLER="/tmp/miniforge_install.sh"
curl -fsSL \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
  -o "$MINIFORGE_INSTALLER"
bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
rm "$MINIFORGE_INSTALLER"

CONDA="$HOME/miniforge3/bin/conda"

# ── 2. Create Python 3.12 environment ─────────────────────────────────────────
echo "==> [2/5] Creating 'lerobot' conda environment (Python 3.12)..."
"$CONDA" create -y -n lerobot python=3.12 pip

ENV_BIN="$HOME/miniforge3/envs/lerobot/bin"
PIP="$ENV_BIN/pip"
PYTHON="$ENV_BIN/python"

if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: conda env is missing python or pip at $ENV_BIN"
  exit 1
fi

# ── 3. Install LeRobot + dependencies ─────────────────────────────────────────
echo "==> [3/5] Installing LeRobot with SmolVLA extras..."
rm -rf "$LEROBOT_DIR"
git clone --depth=1 https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
cd "$LEROBOT_DIR"
"$PIP" install --quiet -e ".[smolvla,dataset]"
"$PIP" install --quiet wandb av

# Patch: huggingface_hub>=1.0 broke RevisionNotFoundError ctor in lerobot
python3 -c "
import pathlib
p = pathlib.Path('src/lerobot/datasets/utils.py')
src = p.read_text()
if 'raise RevisionNotFoundError(' not in src:
    print('Patch not needed')
else:
    lines = src.splitlines(True)
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if 'raise RevisionNotFoundError(' in ln:
            ind = len(ln) - len(ln.lstrip())
            out.append(' ' * ind + 'return \"main\"  # patched\n')
            depth = ln.count('(') - ln.count(')')
            while depth > 0 and i + 1 < len(lines):
                i += 1
                depth += lines[i].count('(') - lines[i].count(')')
        else:
            out.append(ln)
        i += 1
    p.write_text(''.join(out))
    print('Patched get_safe_version')
"

# ── 4. Authenticate with Hugging Face ─────────────────────────────────────────
echo "==> [4/5] Authenticating with Hugging Face Hub..."
"$ENV_BIN/hf" auth login --token "$HF_TOKEN" --add-to-git-credential

# ── 4b. Authenticate with W&B (optional) ──────────────────────────────────────
if [ "$WANDB_ENABLE" = "true" ]; then
  export WANDB_INIT_TIMEOUT=600
  export WANDB_HTTP_TIMEOUT=120
  export WANDB_RESUME=allow
  export WANDB_DISABLE_SERVICE=true

  echo "==> [4b/5] Logging into Weights & Biases..."
  for attempt in 1 2 3 4 5; do
    if "$ENV_BIN/wandb" login --relogin "$WANDB_API_KEY"; then
      break
    fi
    echo "  attempt $attempt failed; retrying in 20s..."
    sleep 20
    if [ "$attempt" = "5" ]; then
      echo "ERROR: wandb login failed after 5 attempts."
      exit 1
    fi
  done
fi

# ── 5. Fine-tune SmolVLA ──────────────────────────────────────────────────────
echo "==> [5/5] Starting fine-tuning..."
rm -rf "$OUTPUT_DIR"

cd "$LEROBOT_DIR"
HF_TOKEN="$HF_TOKEN" "$ENV_BIN/lerobot-train" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision=main \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --grad_accumulation_steps=1 \
  --steps="$TRAIN_STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=true \
  --policy.repo_id="$OUTPUT_REPO_ID" \
  --job_name=smolvla_finetuning \
  --policy.device=cuda \
  --policy.use_amp=true \
  --wandb.enable="$WANDB_ENABLE"

echo ""
echo "==> All done. Model pushed to $OUTPUT_REPO_ID"
