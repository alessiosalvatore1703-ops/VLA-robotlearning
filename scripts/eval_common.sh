#!/usr/bin/env bash
# Shared helpers for the evaluation rollout scripts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ensure_eval_env() {
  if command -v lerobot-rollout >/dev/null 2>&1; then
    return
  fi

  echo "lerobot-rollout was not found. Creating a local evaluation virtualenv..."
  cd "$ROOT_DIR"
  python3 -m venv .eval_venv
  # shellcheck disable=SC1091
  source .eval_venv/bin/activate
  python -m pip install --upgrade pip

  if [ -d "$ROOT_DIR/../lerobot" ]; then
    echo "Installing LeRobot from sibling checkout: $ROOT_DIR/../lerobot"
    python -m pip install -e "$ROOT_DIR/../lerobot[smolvla]"
  else
    echo "Installing LeRobot from PyPI."
    python -m pip install "lerobot[smolvla]"
  fi
fi

select_policy_path() {
  local local_path="$1"
  local hub_repo="$2"

  if [ -d "$local_path" ]; then
    printf "%s" "$local_path"
  else
    printf "%s" "$hub_repo"
  fi
}

run_base_rollout() {
  local policy_path="$1"
  local default_prompt="$2"
  local cli_prompt="${3:-}"
  local task_prompt="${cli_prompt:-${TASK_PROMPT:-$default_prompt}}"

  local robot_type="${ROBOT_TYPE:-so101_follower}"
  local robot_port="${ROBOT_PORT:-/dev/tty.usbmodem5B140319121}"
  local robot_id="${ROBOT_ID:-my_awesome_follower_arm}"
  local camera_config="${CAMERA_CONFIG:-{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10}}}"
  local policy_device="${POLICY_DEVICE:-mps}"
  local duration="${DURATION:-20}"
  local fps="${FPS:-10}"
  local display_data="${DISPLAY_DATA:-true}"
  local return_to_initial="${RETURN_TO_INITIAL_POSITION:-true}"

  echo "============================================================"
  echo "Evaluation rollout"
  echo "  policy : $policy_path"
  echo "  task   : $task_prompt"
  echo "  robot  : $robot_type on $robot_port"
  echo "  device : $policy_device"
  echo "  fps    : $fps"
  echo "============================================================"

  if [ -f "$ROOT_DIR/inference/count_policy_params.py" ]; then
    python "$ROOT_DIR/inference/count_policy_params.py" "$policy_path" --files || true
    echo ""
  fi

  lerobot-rollout \
    --strategy.type=base \
    --policy.path="$policy_path" \
    --policy.device="$policy_device" \
    --robot.type="$robot_type" \
    --robot.port="$robot_port" \
    --robot.id="$robot_id" \
    --robot.cameras="$camera_config" \
    --task="$task_prompt" \
    --duration="$duration" \
    --display_data="$display_data" \
    --fps="$fps" \
    --return_to_initial_position="$return_to_initial"
}
