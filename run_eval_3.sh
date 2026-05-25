#!/usr/bin/env bash
# Evaluation task 3 rollout.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/eval_common.sh
source "$ROOT_DIR/scripts/eval_common.sh"

ensure_eval_env

POLICY_PATH="$(select_policy_path \
  "$ROOT_DIR/policy_checkpoints/task3" \
  "Alessio03/smolvla-task3-50k")"

run_base_rollout "$POLICY_PATH" "Place the coke on Yann LeCun." "${1:-}"
