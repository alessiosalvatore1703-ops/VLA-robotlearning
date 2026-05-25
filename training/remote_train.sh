#!/usr/bin/env bash
# Remote execution script for SmolVLA fine-tuning on an H100 (80 GB VRAM).
# Invoked by orchestrate.py via SSH exec.
#
# Expects /tmp/.lerobot_env to exist on the instance (uploaded by orchestrate.py)
# with the following exports:
#   HF_TOKEN, DATASET_REPO_ID, OUTPUT_REPO_ID, TRAIN_STEPS, BATCH_SIZE, WANDB_ENABLE
set -euo pipefail

LEROBOT_DIR="$HOME/lerobot"

# ── 1. Load credentials ────────────────────────────────────────────────────────
source /tmp/.lerobot_env

POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ETHrobotlearning/tv-colors-task2}"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-ETHrobotlearning/smolvla_eval2_topview_chunk10_30k}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
VLM_NUM_LAYERS="${VLM_NUM_LAYERS:-16}"
JOB_NAME="${JOB_NAME:-smolvla_eval2_topview_chunk10_30k}"
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/train/$JOB_NAME}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
SKIP_SETUP="${SKIP_SETUP:-false}"
DELETE_BREV_ON_SUCCESS="${DELETE_BREV_ON_SUCCESS:-false}"
BREV_INSTANCE_NAME="${BREV_INSTANCE_NAME:-}"
BREV_DELETE_DELAY_SECONDS="${BREV_DELETE_DELAY_SECONDS:-60}"
BREV_DELETE_LOG="${BREV_DELETE_LOG:-$HOME/brev_delete_after_training.log}"

if [ "$DELETE_BREV_ON_SUCCESS" = "true" ]; then
  if [ -z "$BREV_INSTANCE_NAME" ]; then
    echo "ERROR: export BREV_INSTANCE_NAME when DELETE_BREV_ON_SUCCESS=true."
    exit 1
  fi
  if ! command -v brev >/dev/null 2>&1; then
    echo "ERROR: DELETE_BREV_ON_SUCCESS=true requires the Brev CLI on this instance."
    echo "Install it with:"
    echo '  sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"'
    echo "Then run: brev login"
    exit 1
  fi
  if ! brev ls >/dev/null 2>&1; then
    echo "ERROR: Brev CLI is installed but not authenticated."
    echo "Run: brev login"
    exit 1
  fi
  case "$BREV_DELETE_DELAY_SECONDS" in
    ''|*[!0-9]*)
      echo "ERROR: BREV_DELETE_DELAY_SECONDS must be an integer number of seconds."
      exit 1
      ;;
  esac
fi

schedule_brev_delete() {
  if [ "$DELETE_BREV_ON_SUCCESS" != "true" ]; then
    return
  fi

  echo ""
  echo "==> Training completed successfully. Scheduling Brev instance deletion..."
  echo "    Instance : $BREV_INSTANCE_NAME"
  echo "    Delay    : ${BREV_DELETE_DELAY_SECONDS}s"
  echo "    Log file : $BREV_DELETE_LOG"
  BREV_INSTANCE_NAME="$BREV_INSTANCE_NAME" \
  BREV_DELETE_DELAY_SECONDS="$BREV_DELETE_DELAY_SECONDS" \
  BREV_DELETE_LOG="$BREV_DELETE_LOG" \
    nohup bash -c 'sleep "$BREV_DELETE_DELAY_SECONDS"; brev delete "$BREV_INSTANCE_NAME" >> "$BREV_DELETE_LOG" 2>&1' \
    >/dev/null 2>&1 &
}

echo ""
echo "============================================================"
echo " SmolVLA Fine-Tuning Pipeline"
echo "============================================================"
echo " Policy  : $POLICY_PATH"
echo " Dataset : $DATASET_REPO_ID"
echo " Output base : $OUTPUT_REPO_ID"
echo " Steps   : $TRAIN_STEPS  |  Batch: $BATCH_SIZE  |  WandB: $WANDB_ENABLE"
echo " Save/push : every $SAVE_FREQ steps as ${OUTPUT_REPO_ID}-step<N>  |  Local dir: $OUTPUT_DIR"
echo " Chunk   : policy.chunk_size=$ACTION_CHUNK_SIZE  |  n_action_steps=$N_ACTION_STEPS"
echo " VLM layers : policy.num_vlm_layers=$VLM_NUM_LAYERS"
echo " Mode    : action expert training (VLM frozen)"
echo " Setup   : SKIP_SETUP=$SKIP_SETUP"
echo " Brev cleanup : DELETE_BREV_ON_SUCCESS=$DELETE_BREV_ON_SUCCESS"
echo "============================================================"
echo ""

CONDA="$HOME/miniforge3/bin/conda"
ENV_BIN="$HOME/miniforge3/envs/lerobot/bin"
PIP="$ENV_BIN/pip"
PYTHON="$ENV_BIN/python"

if [ "$SKIP_SETUP" = "true" ]; then
  echo "==> [setup] SKIP_SETUP=true; reusing existing LeRobot environment..."
  if [ ! -d "$LEROBOT_DIR" ]; then
    echo "ERROR: $LEROBOT_DIR does not exist. Run once without SKIP_SETUP=true first."
    exit 1
  fi
  if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ]; then
    echo "ERROR: lerobot conda env is missing python or pip at $ENV_BIN."
    echo "Run once without SKIP_SETUP=true first."
    exit 1
  fi
  if [ ! -x "$ENV_BIN/lerobot-train" ]; then
    echo "ERROR: lerobot-train is missing at $ENV_BIN/lerobot-train"
    echo "Run once without SKIP_SETUP=true first."
    exit 1
  fi
  cd "$LEROBOT_DIR"
else
  # ── 2. Install miniforge silently ───────────────────────────────────────────
  echo "==> [1/5] Installing miniforge (Python environment manager)..."
  MINIFORGE_INSTALLER="/tmp/miniforge_install.sh"
  # -fsSL: -f exits non-zero on HTTP errors, -s silent, -S show errors, -L follow redirects.
  curl -fsSL \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    -o "$MINIFORGE_INSTALLER"
  bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
  rm "$MINIFORGE_INSTALLER"

  # ── 3. Create isolated Python 3.12 environment ──────────────────────────────
  # Include `pip` explicitly: with the conda-forge channel used by miniforge,
  # `python=3.12` alone does not always pull pip as a default dependency.
  # Use -y instead of `always_yes` config to avoid edge cases on a fresh install.
  echo "==> [2/5] Creating 'lerobot' conda environment (Python 3.12 + pip)..."
  "$CONDA" create -y -n lerobot python=3.12 pip

  # Fail loudly if conda silently created the env without pip / python.
  if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ]; then
      echo "ERROR: lerobot conda env is missing python or pip."
      echo "  expected:  $PIP  +  $PYTHON"
      ls -la "$ENV_BIN" 2>&1 || true
      exit 1
  fi

  # ── 4. Install LeRobot + SmolVLA VLA dependencies ───────────────────────────
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
fi

# Patch: factory.py doesn't guard against None stats (image features have no stats)
"$PYTHON" -c "
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
"$PYTHON" -c "
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

# Patch: upstream LeRobot saves local checkpoints every save_freq steps, but only
# pushes the policy to the Hub at the end. For long Brev runs we instead push
# each saved checkpoint to a separate model repo:
#   OUTPUT_REPO_ID-step10000, OUTPUT_REPO_ID-step20000, ...
"$PYTHON" - <<'PY'
from pathlib import Path

p = Path("src/lerobot/scripts/lerobot_train.py")
src = p.read_text()

marker = """                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()
"""

replacement = """                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

                if getattr(active_cfg, "push_to_hub", False):
                    base_repo_id = active_cfg.repo_id
                    checkpoint_repo_id = f"{base_repo_id}-step{step}"
                    logging.info(f"Pushing policy checkpoint after step {step} to Hub repo {checkpoint_repo_id}")
                    unwrapped_model = accelerator.unwrap_model(policy)
                    original_active_repo_id = getattr(active_cfg, "repo_id", None)
                    original_policy_repo_id = getattr(cfg.policy, "repo_id", None) if hasattr(cfg, "policy") else None
                    original_model_repo_id = getattr(unwrapped_model.config, "repo_id", None)
                    try:
                        active_cfg.repo_id = checkpoint_repo_id
                        if hasattr(cfg, "policy"):
                            cfg.policy.repo_id = checkpoint_repo_id
                        unwrapped_model.config.repo_id = checkpoint_repo_id
                        if not cfg.is_reward_model_training and cfg.policy.use_peft:
                            unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
                        else:
                            unwrapped_model.push_model_to_hub(cfg)
                        preprocessor.push_to_hub(checkpoint_repo_id)
                        postprocessor.push_to_hub(checkpoint_repo_id)
                    finally:
                        active_cfg.repo_id = original_active_repo_id
                        if hasattr(cfg, "policy"):
                            cfg.policy.repo_id = original_policy_repo_id
                        unwrapped_model.config.repo_id = original_model_repo_id

            accelerator.wait_for_everyone()
"""

final_marker = """        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)
"""

final_replacement = """        if getattr(active_cfg, "push_to_hub", False):
            logging.info(
                "Skipping final push to base repo; saved checkpoints are pushed to per-step repos."
            )
"""

if "checkpoint_repo_id = f\"{base_repo_id}-step{step}\"" in src:
    print("per-step checkpoint Hub push patch already present")
elif marker in src:
    p.write_text(src.replace(marker, replacement))
    src = p.read_text()
    print("Patched lerobot_train.py to push each saved checkpoint to a per-step Hub repo")
else:
    raise RuntimeError("Could not find checkpoint save block in lerobot_train.py")

if "Skipping final push to base repo" in src:
    print("final base push skip patch already present")
elif final_marker in src:
    p.write_text(src.replace(final_marker, final_replacement))
    print("Patched lerobot_train.py to skip final base repo push")
else:
    raise RuntimeError("Could not find final Hub push block in lerobot_train.py")
PY

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
echo "==> Launch mode: single GPU"
HF_TOKEN="$HF_TOKEN" "$ENV_BIN/lerobot-train" \
  --policy.type=smolvla \
  --policy.pretrained_path="$POLICY_PATH" \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision=main \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --steps="$TRAIN_STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=true \
  --policy.repo_id="$OUTPUT_REPO_ID" \
  --job_name="$JOB_NAME" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.chunk_size="$ACTION_CHUNK_SIZE" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --policy.num_vlm_layers="$VLM_NUM_LAYERS" \
  --wandb.enable="$WANDB_ENABLE"

echo "==> Fine-tuning and upload complete."
schedule_brev_delete

# ── 7. Remove credentials from the instance ───────────────────────────────────
rm -f /tmp/.lerobot_env
echo ""
echo "==> All done. Credentials removed from instance."
