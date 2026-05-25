#!/usr/bin/env bash
# Evaluation task 2 rollout.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/eval_common.sh
source "$ROOT_DIR/scripts/eval_common.sh"

ensure_eval_env

POLICY_PATH="$(select_policy_path \
  "$ROOT_DIR/policy_checkpoints/task2" \
  "ETHrobotlearning/smolvla_task2_colors_dagger_lr2e-5-step2000")"

run_base_rollout "$POLICY_PATH" "Put the banana in the green colored bowl." "${1:-}"
