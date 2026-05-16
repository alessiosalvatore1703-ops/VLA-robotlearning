#!/usr/bin/env bash
# Remote execution script for SmolVLA fine-tuning on an H100 (80 GB VRAM).
# Invoked by orchestrate.py via SSH exec.
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
echo "==> [1/5] Installing miniforge (Python environment manager)..."
MINIFORGE_INSTALLER="/tmp/miniforge_install.sh"
# -fsSL: -f exits non-zero on HTTP errors, -s silent, -S show errors, -L follow redirects.
curl -fsSL \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
  -o "$MINIFORGE_INSTALLER"
bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
rm "$MINIFORGE_INSTALLER"

CONDA="$HOME/miniforge3/bin/conda"

# ── 3. Create isolated Python 3.12 environment ────────────────────────────────
# Include `pip` explicitly: with the conda-forge channel used by miniforge,
# `python=3.12` alone does not always pull pip as a default dependency.
# Use -y instead of `always_yes` config to avoid edge cases on a fresh install.
echo "==> [2/5] Creating 'lerobot' conda environment (Python 3.12 + pip)..."
"$CONDA" create -y -n lerobot python=3.12 pip


# Use explicit paths so this works in non-interactive shells where
# `conda activate` silently fails to switch the active Python.
ENV_BIN="$HOME/miniforge3/envs/lerobot/bin"
PIP="$ENV_BIN/pip"
PYTHON="$ENV_BIN/python"

# Fail loudly if conda silently created the env without pip / python.
if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ]; then
    echo "ERROR: lerobot conda env is missing python or pip."
    echo "  expected:  $PIP  +  $PYTHON"
    ls -la "$ENV_BIN" 2>&1 || true
    exit 1
fi

# ── 4. Install LeRobot + SmolVLA VLA dependencies ─────────────────────────────
echo "==> [3/5] Installing LeRobot with SmolVLA extras..."
rm -rf "$LEROBOT_DIR"
git clone --depth=1 https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
cd "$LEROBOT_DIR"
# [smolvla] — VLA model deps; [dataset] — adds the `datasets` package required at import time
"$PIP" install --quiet -e ".[smolvla,dataset]"
# wandb is not bundled in any lerobot extra
"$PIP" install --quiet wandb
# av (PyAV) is the video backend we use; torchcodec is incompatible with PyTorch 2.10+cu128
"$PIP" install --quiet av

# Patch: factory.py doesn't guard against None stats (image features have no stats)
python3 -c "
import pathlib
p = pathlib.Path('src/lerobot/datasets/factory.py')
src = p.read_text()
old = 'dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)'
if old not in src:
    print('factory patch not needed')
else:
    for line in src.splitlines():
        if old in line:
            indent = ' ' * (len(line) - len(line.lstrip()))
            break
    new = f'if dataset.meta.stats is not None and dataset.meta.stats.get(key) is not None:\n{indent}    {old}'
    p.write_text(src.replace(indent + old, indent + new))
    print('Patched factory.py')
"

# Patch get_safe_version: huggingface_hub>=1.0 made HfHubHTTPError.__init__ require
# a keyword-only 'response' arg, so lerobot's bare RevisionNotFoundError(message) raise
# crashes with TypeError.  Datasets we push have no v* tags anyway, so returning
# "main" is the correct fallback.
python3 -c "
import pathlib
p = pathlib.Path('src/lerobot/datasets/utils.py')
src = p.read_text()
if 'raise RevisionNotFoundError(' not in src:
    print('Patch not needed (pattern not found)')
else:
    lines = src.splitlines(True)
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if 'raise RevisionNotFoundError(' in ln:
            ind = len(ln) - len(ln.lstrip())
            out.append(' ' * ind + 'return \"main\"  # patched: hf_hub>=1.0 broke RevisionNotFoundError ctor\n')
            depth = ln.count('(') - ln.count(')')
            while depth > 0 and i + 1 < len(lines):
                i += 1
                depth += lines[i].count('(') - lines[i].count(')')
        else:
            out.append(ln)
        i += 1
    p.write_text(''.join(out))
    print('Patched get_safe_version in', str(p))
"

# ── 5. Authenticate with Hugging Face (non-interactive) ───────────────────────
# `huggingface-cli` was deprecated; the new CLI is `hf` (same package).
echo "==> [4/5] Authenticating with Hugging Face Hub..."
"$ENV_BIN/hf" auth login --token "$HF_TOKEN" --add-to-git-credential

# ── 5b. Configure Weights & Biases for unstable networks ──────────────────────
# Brev instances occasionally show transient timeouts to api.wandb.ai during
# the first minutes after boot.  We:
#   (a) bump init / HTTP timeouts so a slow first request doesn't kill the run,
#   (b) log in explicitly so auth + DNS fail fast (before training starts),
#   (c) enable resume mode so a network blip mid-run doesn't tank training.
if [ "$WANDB_ENABLE" = "true" ]; then
  export WANDB_INIT_TIMEOUT=600       # default ~30s — too short on cold instances
  export WANDB_HTTP_TIMEOUT=120       # default ~10s
  export WANDB_RESUME=allow           # keep training even if the run reconnects
  export WANDB_DISABLE_SERVICE=true   # avoid wandb-service hangs on some hosts

  echo "==> [4b/5] Logging into Weights & Biases (verifies network + key)..."
  # Retry the login a few times — DNS / network sometimes isn't ready yet.
  wandb_ok=false
  for attempt in 1 2 3 4 5; do
    if "$ENV_BIN/wandb" login --relogin "$WANDB_API_KEY"; then
      wandb_ok=true
      break
    fi
    echo "  wandb login attempt $attempt failed; retrying in 20s..."
    sleep 20
  done
  if [ "$wandb_ok" = false ]; then
    echo "ERROR: wandb login failed after 5 attempts. Aborting to avoid wasting GPU time."
    exit 1
  fi
fi

# ── 6. Fine-tune SmolVLA ──────────────────────────────────────────────────────
echo "==> [5/5] Starting fine-tuning on $DATASET_REPO_ID..."
rm -rf "$OUTPUT_DIR"   # lerobot-train raises FileExistsError if the dir exists, even when empty

cd "$LEROBOT_DIR"
HF_TOKEN="$HF_TOKEN" "$ENV_BIN/lerobot-train" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision=main \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --steps="$TRAIN_STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=true \
  --policy.repo_id="$OUTPUT_REPO_ID" \
  --job_name=smolvla_finetuning \
  --policy.device=cuda \
  --policy.use_amp=true \
  --wandb.enable="$WANDB_ENABLE"

echo "==> Fine-tuning and upload complete."

# ── 7. Remove credentials from the instance ───────────────────────────────────
rm -f /tmp/.lerobot_env
echo ""
echo "==> All done. Credentials removed from instance."
