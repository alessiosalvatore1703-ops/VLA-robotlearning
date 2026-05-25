#!/usr/bin/env bash
# Run this script on a fresh Brev H100 instance after SSHing in.
#
# Minimal usage on the Brev instance:
#   export HF_TOKEN=hf_...
#   export OUTPUT_REPO_ID=YOUR_USER/molmoact2-task3-toy
#   export WANDB_API_KEY=...
#   bash setup_and_train_molmoact2.sh
#
# The script also sources /tmp/.molmoact2_env when present. The local
# orchestrator writes that file before executing this script remotely.
set -euo pipefail

ENV_FILE="${ENV_FILE:-/tmp/.molmoact2_env}"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

# -- MolmoAct2 defaults from the SO100/SO101 task3 plan ----------------------
HF_TOKEN="${HF_TOKEN:-}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ETHrobotlearning/task3-TOY-clean}"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-}"
POLICY_CHECKPOINT_PATH="${POLICY_CHECKPOINT_PATH:-allenai/MolmoAct2-SO100_101}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
ACTION_MODE="${ACTION_MODE:-continuous}"
IMAGE_KEYS="${IMAGE_KEYS:-[\"observation.images.front\"]}"
SETUP_TYPE="${SETUP_TYPE:-single SO-100/SO-101 arm with one front RGB camera}"
CONTROL_MODE="${CONTROL_MODE:-absolute joint pose}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
NUM_FLOW_TIMESTEPS="${NUM_FLOW_TIMESTEPS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FREEZE_EMBEDDING="${FREEZE_EMBEDDING:-true}"
NORMALIZE_GRIPPER="${NORMALIZE_GRIPPER:-false}"
ENABLE_KNOWLEDGE_INSULATION="${ENABLE_KNOWLEDGE_INSULATION:-false}"
TRAIN_ACTION_EXPERT_ONLY="${TRAIN_ACTION_EXPERT_ONLY:-false}"
ENABLE_LORA_VLM="${ENABLE_LORA_VLM:-true}"
ENABLE_LORA_ACTION_EXPERT="${ENABLE_LORA_ACTION_EXPERT:-false}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-20}"
EVAL_FREQ="${EVAL_FREQ:--1}"
JOB_NAME="${JOB_NAME:-molmoact2_so101_task3_toy_lora_50k}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/train/$JOB_NAME}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-molmoact2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
SKIP_SETUP="${SKIP_SETUP:-false}"
SKIP_DATASET_DOWNLOAD="${SKIP_DATASET_DOWNLOAD:-false}"
MOLMOACT2_REPO_URL="${MOLMOACT2_REPO_URL:-https://github.com/allenai/molmoact2.git}"
MOLMOACT2_GIT_REF="${MOLMOACT2_GIT_REF:-main}"
DELETE_BREV_ON_SUCCESS="${DELETE_BREV_ON_SUCCESS:-false}"
BREV_INSTANCE_NAME="${BREV_INSTANCE_NAME:-}"
BREV_DELETE_DELAY_SECONDS="${BREV_DELETE_DELAY_SECONDS:-60}"
BREV_DELETE_LOG="${BREV_DELETE_LOG:-$HOME/brev_delete_after_training.log}"

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: export HF_TOKEN before running."
  exit 1
fi
if [ "$PUSH_TO_HUB" = "true" ] && [ -z "$OUTPUT_REPO_ID" ]; then
  echo "ERROR: export OUTPUT_REPO_ID when PUSH_TO_HUB=true."
  exit 1
fi
if [ "$WANDB_ENABLE" = "true" ] && [ -z "$WANDB_API_KEY" ]; then
  echo "ERROR: export WANDB_API_KEY when WANDB_ENABLE=true."
  exit 1
fi
if [ "$DELETE_BREV_ON_SUCCESS" = "true" ]; then
  if [ -z "$BREV_INSTANCE_NAME" ]; then
    echo "ERROR: export BREV_INSTANCE_NAME when DELETE_BREV_ON_SUCCESS=true."
    exit 1
  fi
  if ! command -v brev >/dev/null 2>&1; then
    echo "ERROR: DELETE_BREV_ON_SUCCESS=true requires the Brev CLI on this instance."
    echo 'Install it with: sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"'
    exit 1
  fi
  if ! brev ls >/dev/null 2>&1; then
    echo "ERROR: Brev CLI is installed but not authenticated. Run: brev login"
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

MOLMOACT2_DIR="$HOME/molmoact2"
LEROBOT_DIR="$MOLMOACT2_DIR/lerobot"

echo ""
echo "============================================================"
echo " MolmoAct2 SO100/SO101 Fine-Tuning Pipeline"
echo "============================================================"
echo " Dataset       : $DATASET_REPO_ID"
echo " Checkpoint    : $POLICY_CHECKPOINT_PATH"
echo " Output repo   : ${OUTPUT_REPO_ID:-<hub push disabled>}"
echo " Steps/batch   : $TRAIN_STEPS / $BATCH_SIZE"
echo " Save/push     : every $SAVE_FREQ steps as ${OUTPUT_REPO_ID:-OUTPUT_REPO_ID}-step<N>"
echo " Camera keys   : $IMAGE_KEYS"
echo " Chunk/actions : policy.chunk_size=$ACTION_CHUNK_SIZE | n_action_steps=$N_ACTION_STEPS"
echo " Mode          : VLM LoRA + full action expert"
echo " LoRA          : vlm=$ENABLE_LORA_VLM action_expert=$ENABLE_LORA_ACTION_EXPERT r=$LORA_RANK alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo " Setup/control : $SETUP_TYPE | $CONTROL_MODE"
echo " W&B           : $WANDB_ENABLE"
echo " Setup flags   : SKIP_SETUP=$SKIP_SETUP | SKIP_DATASET_DOWNLOAD=$SKIP_DATASET_DOWNLOAD"
echo "============================================================"
echo ""

CONDA="$HOME/miniforge3/bin/conda"
ENV_BIN="$HOME/miniforge3/envs/lerobot/bin"
PIP="$ENV_BIN/pip"
PYTHON="$ENV_BIN/python"

if [ "$SKIP_SETUP" = "true" ]; then
  echo "==> [setup] SKIP_SETUP=true; reusing existing MolmoAct2 LeRobot environment..."
  if [ ! -d "$LEROBOT_DIR" ]; then
    echo "ERROR: $LEROBOT_DIR does not exist. Run once without SKIP_SETUP=true first."
    exit 1
  fi
  if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ] || [ ! -x "$ENV_BIN/lerobot-train" ]; then
    echo "ERROR: lerobot env is missing python, pip, or lerobot-train at $ENV_BIN."
    exit 1
  fi
  cd "$LEROBOT_DIR"
else
  echo "==> [1/6] Installing miniforge..."
  MINIFORGE_INSTALLER="/tmp/miniforge_install.sh"
  curl -fsSL \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    -o "$MINIFORGE_INSTALLER"
  bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
  rm "$MINIFORGE_INSTALLER"

  echo "==> [2/6] Creating 'lerobot' conda environment (Python 3.12)..."
  "$CONDA" create -y -n lerobot python=3.12 pip

  if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ]; then
    echo "ERROR: conda env is missing python or pip at $ENV_BIN"
    exit 1
  fi

  echo "==> [3/6] Cloning MolmoAct2 and installing LeRobot MolmoAct2 extras..."
  rm -rf "$MOLMOACT2_DIR"
  git clone --recurse-submodules "$MOLMOACT2_REPO_URL" "$MOLMOACT2_DIR"
  cd "$MOLMOACT2_DIR"
  if [ -n "$MOLMOACT2_GIT_REF" ]; then
    git fetch --all --tags
    git checkout "$MOLMOACT2_GIT_REF"
    git submodule update --init --recursive
  fi
  cd "$LEROBOT_DIR"
  "$PIP" install --quiet -e ".[training,molmoact2]"
  "$PIP" install --quiet av hf_transfer
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Patch: factory.py does not always guard against missing image stats.
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
    new = f'if dataset.meta.stats is not None and dataset.meta.stats.get(key) is not None:\\n{indent}    {old}'
    p.write_text(src.replace(indent + old, indent + new))
    print('Patched factory.py')
"

# Patch get_safe_version for huggingface_hub>=1.0 compatibility.
"$PYTHON" -c "
import pathlib
p = pathlib.Path('src/lerobot/datasets/utils.py')
src = p.read_text()
if 'raise RevisionNotFoundError(' not in src:
    print('get_safe_version patch not needed')
else:
    lines = src.splitlines(True)
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if 'raise RevisionNotFoundError(' in ln:
            ind = len(ln) - len(ln.lstrip())
            out.append(' ' * ind + 'return \"main\"  # patched: hf_hub>=1.0 RevisionNotFoundError ctor\\n')
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

# Patch LeRobot so each saved checkpoint is pushed to a separate Hub repo:
#   OUTPUT_REPO_ID-step5000, OUTPUT_REPO_ID-step10000, ...
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
                    policy_uses_peft = bool(getattr(cfg.policy, "use_peft", False))
                    try:
                        active_cfg.repo_id = checkpoint_repo_id
                        if hasattr(cfg, "policy"):
                            cfg.policy.repo_id = checkpoint_repo_id
                        unwrapped_model.config.repo_id = checkpoint_repo_id
                        if not cfg.is_reward_model_training and policy_uses_peft:
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

if 'checkpoint_repo_id = f"{base_repo_id}-step{step}"' in src:
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
    print("final base push block not found; leaving final push behavior unchanged")
PY

echo "==> [4/6] Authenticating with Hugging Face Hub..."
"$ENV_BIN/hf" auth login --token "$HF_TOKEN" --add-to-git-credential

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

if [ "$SKIP_DATASET_DOWNLOAD" = "true" ]; then
  echo "==> [5/6] SKIP_DATASET_DOWNLOAD=true; not running hf download."
else
  echo "==> [5/6] Downloading dataset to local cache..."
  "$ENV_BIN/hf" download "$DATASET_REPO_ID" --repo-type dataset
fi

echo "==> [6/6] Starting MolmoAct2 fine-tuning..."
rm -rf "$OUTPUT_DIR"
cd "$LEROBOT_DIR"

train_args=(
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.revision=main"
  "--dataset.video_backend=pyav"
  "--dataset.image_transforms.enable=true"
  "--policy.type=molmoact2"
  "--policy.checkpoint_path=$POLICY_CHECKPOINT_PATH"
  "--policy.device=cuda"
  "--policy.action_mode=$ACTION_MODE"
  "--policy.train_action_expert_only=$TRAIN_ACTION_EXPERT_ONLY"
  "--policy.chunk_size=$ACTION_CHUNK_SIZE"
  "--policy.n_action_steps=$N_ACTION_STEPS"
  "--policy.setup_type=$SETUP_TYPE"
  "--policy.control_mode=$CONTROL_MODE"
  "--policy.image_keys=$IMAGE_KEYS"
  "--policy.model_dtype=$MODEL_DTYPE"
  "--policy.num_flow_timesteps=$NUM_FLOW_TIMESTEPS"
  "--policy.gradient_checkpointing=$GRADIENT_CHECKPOINTING"
  "--policy.freeze_embedding=$FREEZE_EMBEDDING"
  "--policy.normalize_gripper=$NORMALIZE_GRIPPER"
  "--policy.enable_knowledge_insulation=$ENABLE_KNOWLEDGE_INSULATION"
  "--policy.enable_lora_vlm=$ENABLE_LORA_VLM"
  "--policy.enable_lora_action_expert=$ENABLE_LORA_ACTION_EXPERT"
  "--policy.lora_rank=$LORA_RANK"
  "--policy.lora_alpha=$LORA_ALPHA"
  "--policy.lora_dropout=$LORA_DROPOUT"
  "--batch_size=$BATCH_SIZE"
  "--steps=$TRAIN_STEPS"
  "--save_checkpoint=true"
  "--save_freq=$SAVE_FREQ"
  "--output_dir=$OUTPUT_DIR"
  "--job_name=$JOB_NAME"
  "--num_workers=$NUM_WORKERS"
  "--log_freq=$LOG_FREQ"
  "--eval_freq=$EVAL_FREQ"
  "--policy.push_to_hub=$PUSH_TO_HUB"
  "--wandb.enable=$WANDB_ENABLE"
)

if [ -n "$OUTPUT_REPO_ID" ]; then
  train_args+=("--policy.repo_id=$OUTPUT_REPO_ID")
fi
if [ -n "$WANDB_PROJECT" ]; then
  train_args+=("--wandb.project=$WANDB_PROJECT")
fi
if [ -n "$WANDB_ENTITY" ]; then
  train_args+=("--wandb.entity=$WANDB_ENTITY")
fi

HF_TOKEN="$HF_TOKEN" "$ENV_BIN/lerobot-train" "${train_args[@]}"

echo ""
if [ "$PUSH_TO_HUB" = "true" ]; then
  echo "==> All done. Checkpoint model repos were pushed as ${OUTPUT_REPO_ID}-step<N>"
else
  echo "==> All done. Local checkpoints are under $OUTPUT_DIR"
fi
schedule_brev_delete

if [ "$ENV_FILE" = "/tmp/.molmoact2_env" ]; then
  rm -f "$ENV_FILE"
  echo "==> Credentials removed from $ENV_FILE."
fi
