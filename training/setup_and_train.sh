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
HF_TOKEN="${HF_TOKEN:-}"  # export HF_TOKEN before running, or set here (do NOT commit a real token)
DATASET_REPO_ID="ETHrobotlearning/tv-task2-clean-aug5p-fixed-negation-relative"  # <-- update to your combined dataset (Taylor Swift + Obama + LeCun episodes)
OUTPUT_REPO_ID="ETHrobotlearning/fix"   # <-- update to your desired output repo
TRAIN_STEPS=10000         # ~400 episodes × 200 frames = 80k frames; 50k steps at batch 64 ≈ 40 epochs — sufficient for this task size
BATCH_SIZE=32             # fits natively on an 80GB H100; no gradient accumulation needed (lerobot-train uses GA=1)
LR="2e-5"
SAVE_FREQ=1000            # save a checkpoint every N steps (lets you stop at any time safely)
RESUME="false"            # set to "true" to resume a previous run from OUTPUT_DIR
WANDB_ENABLE="true"      # set to "true" to enable W&B logging
WANDB_API_KEY="${WANDB_API_KEY:-}"  # export WANDB_API_KEY before running, or set here (do NOT commit a real key)
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
DATASET_DIR="$HOME/datasets/$(basename "$DATASET_REPO_ID")"
OUTPUT_DIR="$HOME/outputs/smolvla"

echo ""
echo "============================================================"
echo " SmolVLA Fine-Tuning Pipeline"
echo "============================================================"
echo " Dataset : $DATASET_REPO_ID"
echo " Output  : $OUTPUT_REPO_ID"
echo " Steps   : $TRAIN_STEPS  |  Batch: $BATCH_SIZE  |  LR: $LR  |  Save every: $SAVE_FREQ  |  Resume: $RESUME  |  WandB: $WANDB_ENABLE"
echo "============================================================"
echo ""

# ── 1. Install miniforge ──────────────────────────────────────────────────────
CONDA="$HOME/miniforge3/bin/conda"
if [ -x "$CONDA" ]; then
  echo "==> [1/6] miniforge already installed, skipping."
else
  echo "==> [1/6] Installing miniforge..."
  MINIFORGE_INSTALLER="/tmp/miniforge_install.sh"
  curl -fsSL \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    -o "$MINIFORGE_INSTALLER"
  bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
  rm "$MINIFORGE_INSTALLER"
fi

# ── 2. Create Python 3.12 environment ─────────────────────────────────────────
ENV_BIN="$HOME/miniforge3/envs/lerobot/bin"
if [ -x "$ENV_BIN/python" ]; then
  echo "==> [2/6] conda env 'lerobot' already exists, skipping."
else
  echo "==> [2/6] Creating 'lerobot' conda environment (Python 3.12)..."
  "$CONDA" create -y -n lerobot python=3.12 pip
fi

PIP="$ENV_BIN/pip"
PYTHON="$ENV_BIN/python"

if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: conda env is missing python or pip at $ENV_BIN"
  exit 1
fi

# ── 3. Install LeRobot + dependencies ─────────────────────────────────────────
if [ -d "$LEROBOT_DIR" ] && "$PYTHON" -c "import lerobot" 2>/dev/null; then
  echo "==> [3/6] LeRobot already installed, skipping."
else
  echo "==> [3/6] Installing LeRobot with SmolVLA extras..."
  rm -rf "$LEROBOT_DIR"
  git clone --depth=1 https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
  cd "$LEROBOT_DIR"
  "$PIP" install --quiet -e ".[smolvla,dataset]"
  "$PIP" install --quiet wandb av

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
fi

# ── 4. Authenticate with Hugging Face ─────────────────────────────────────────
echo "==> [4/6] Authenticating with Hugging Face Hub..."
"$PYTHON" -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)"

# ── 4b. Authenticate with W&B (optional) ──────────────────────────────────────
if [ "$WANDB_ENABLE" = "true" ]; then
  export WANDB_INIT_TIMEOUT=600
  export WANDB_HTTP_TIMEOUT=120
  export WANDB_RESUME=allow
  export WANDB_DISABLE_SERVICE=true

  echo "==> [4b/6] Logging into Weights & Biases..."
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

# ── 5. Download dataset locally ───────────────────────────────────────────────
DATASET_CACHE_FLAG="$HOME/.cache/lerobot_datasets/$(echo "$DATASET_REPO_ID" | tr '/' '_').done"
if [ -f "$DATASET_CACHE_FLAG" ]; then
  echo "==> [5/6] Dataset already downloaded, skipping."
else
  echo "==> [5/6] Downloading dataset to local disk (avoids slow streaming during training)..."
  "$PYTHON" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$DATASET_REPO_ID', repo_type='dataset')"
  mkdir -p "$(dirname "$DATASET_CACHE_FLAG")" && touch "$DATASET_CACHE_FLAG"
fi

# Note: episode-level train/val split is skipped — LeRobot's frame sampler does not
# correctly respect the episodes filter and crashes mid-training. Monitor overfitting
# via W&B training loss and evaluate on the real robot.

# ── 6. Fine-tune SmolVLA ──────────────────────────────────────────────────────
echo "==> [6/6] Starting fine-tuning..."
if [ "$RESUME" != "true" ]; then
  rm -rf "$OUTPUT_DIR"
fi

# Build optional training args
TRAIN_EXTRA_ARGS=()
[ "$RESUME" = "true" ] && TRAIN_EXTRA_ARGS+=(--resume=true)

# ── Checkpoint watcher: pushes each new checkpoint to HuggingFace ─────────────
_push_checkpoints() {
  local seen=""
  while true; do
    sleep 60
    [ -d "$OUTPUT_DIR/checkpoints" ] || continue
    for ckpt_dir in "$OUTPUT_DIR/checkpoints"/*/; do
      [ -d "$ckpt_dir" ] || continue
      ckpt_name=$(basename "$ckpt_dir")
      # skip the 'last' symlink/dir — it's not a real numbered checkpoint
      [ "$ckpt_name" = "last" ] && continue
      [[ "$seen" == *"|$ckpt_name|"* ]] && continue
      ckpt_repo="${OUTPUT_REPO_ID}_${ckpt_name}"
      echo "==> [watcher] Pushing checkpoint $ckpt_name to $ckpt_repo ..."
      push_ok=false
      for attempt in 1 2 3; do
        if HF_TOKEN="$HF_TOKEN" CKPT_REPO="$ckpt_repo" "$PYTHON" - <<PYEOF; then
from huggingface_hub import HfApi
import os
from pathlib import Path
api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["CKPT_REPO"]
ckpt = Path("$ckpt_dir")
model_dir = ckpt / "pretrained_model"
if not model_dir.exists():
    model_dir = ckpt
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=str(model_dir),
    repo_id=repo_id,
    repo_type="model",
    path_in_repo="",
    commit_message="checkpoint $ckpt_name",
)
print(f"  Pushed to {repo_id}")
PYEOF
          push_ok=true
          break
        fi
        echo "  attempt $attempt failed; retrying in 30s..."
        sleep 30
      done
      if $push_ok; then
        seen="$seen|$ckpt_name|"
        rm -rf "$ckpt_dir" && echo "  Deleted local $ckpt_name"
      else
        echo "  Warning: push failed for $ckpt_name after 3 attempts, keeping local copy"
      fi
    done
  done
}
_push_checkpoints &
WATCHER_PID=$!
trap "kill \$WATCHER_PID 2>/dev/null; wait \$WATCHER_PID 2>/dev/null" EXIT INT TERM

cd "$LEROBOT_DIR"
HF_TOKEN="${HF_TOKEN:-}"  # export HF_TOKEN before running, or set here (do NOT commit a real token)
  --policy.type=smolvla \
  --policy.pretrained_path=ETHrobotlearning/task-2-fix-agumentations_005000 \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision=main \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --steps="$TRAIN_STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=true \
  --policy.repo_id="${OUTPUT_REPO_ID}_${TRAIN_STEPS}" \
  --job_name=fine-tune-task2-colneg \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_decay_steps="$TRAIN_STEPS" \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.num_vlm_layers=16 \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --wandb.enable="$WANDB_ENABLE" \
  "${TRAIN_EXTRA_ARGS[@]}"

echo ""
echo "==> All done. Model pushed to $OUTPUT_REPO_ID"

# ── Auto-delete this instance ─────────────────────────────────────────────────
INSTANCE_NAME=$(curl -sf http://169.254.169.254/latest/meta-data/tags/instance/name 2>/dev/null || hostname)

# Install the brev CLI *now* (before we need it) so a network hiccup at delete
# time can't stop us.
if ! command -v brev >/dev/null 2>&1; then
  echo "==> Installing brev CLI for self-delete..."
  curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/assets/install-brev.sh | bash 2>/dev/null || true
fi
BREV_BIN="$(command -v brev || echo "$HOME/.brev/bin/brev")"

# The delete MUST run detached from this shell. `brev delete` tears down the
# instance's network connection, which drops the SSH/tmux session that launched
# it. If the delete process is still a child of that session it gets SIGHUP'd
# and dies before the API call completes — so the instance survives (exactly the
# "disconnected from tmux but not deleted" symptom). setsid + nohup give it its
# own session so it outlives the disconnect; a retry loop covers transient fails.
DELETE_SCRIPT="$HOME/.auto_delete.sh"
DELETE_LOG="$HOME/auto_delete.log"
cat > "$DELETE_SCRIPT" <<EOF
#!/usr/bin/env bash
sleep 60
echo "\$(date) — deleting instance $INSTANCE_NAME"
for attempt in 1 2 3 4 5; do
  if "$BREV_BIN" delete "$INSTANCE_NAME" --force; then
    echo "\$(date) — instance $INSTANCE_NAME deleted"
    exit 0
  fi
  echo "\$(date) — delete attempt \$attempt failed; retrying in 30s..."
  sleep 30
done
echo "\$(date) — WARNING: auto-delete failed. Delete manually: brev delete $INSTANCE_NAME"
EOF
chmod +x "$DELETE_SCRIPT"

echo "==> Self-deleting instance ($INSTANCE_NAME) in 60s — detached, survives tmux/SSH disconnect."
echo "    Progress log: $DELETE_LOG   |   Abort with: pkill -f .auto_delete.sh"
setsid nohup bash "$DELETE_SCRIPT" >"$DELETE_LOG" 2>&1 </dev/null &
disown
