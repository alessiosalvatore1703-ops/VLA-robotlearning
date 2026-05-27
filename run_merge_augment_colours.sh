#!/usr/bin/env bash
# Merge 6 config datasets with relabeled colour prompts, then brightness-augment.
#
# Pipeline:
#   1. relabel_bowls_and_merge.py  : merge the 6 original configs + rewrite ordinal
#                                    bowl prompts to colour prompts
#                                    -> ETHrobotlearning/colours-task2-premerge  (tmp)
#   2. augmentation/augment_brightness_push.py : brightness-augment the merged dataset
#                                    -> ETHrobotlearning/colours-task2  (final)
#   3. Delete the temporary intermediate repo from the Hub.
#
# Order note: relabelling must happen BEFORE augmentation because the colour
# parser reads config names (e.g. config1-red-blue-green) from the repo ID.
# Augmented intermediates would no longer carry those names.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

INPUTS=(
    "ETHrobotlearning/config1-red-blue-green"
    "ETHrobotlearning/config2-green-red-blue"
    "ETHrobotlearning/config3-green-blue-red"
    "ETHrobotlearning/config4-red-green-blue"
    "ETHrobotlearning/config5-blue-green-red"
    "ETHrobotlearning/config6-blue-red-green"
)

TMP_REPO="ETHrobotlearning/colours-task2-premerge"
FINAL_REPO="ETHrobotlearning/colours-task2"

BRIGHTNESS_LEVELS="0.5 0.6 0.7 0.8 0.9 1.3 1.1 1.2"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
UTILS_DIR="$REPO_ROOT/datasets/utils"

# ── Conda environment ────────────────────────────────────────────────────────

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# ── Step 1 : Merge + relabel ─────────────────────────────────────────────────

echo "============================================================"
echo "Step 1/3 : Merging and relabelling 6 config datasets"
echo "  inputs  : ${INPUTS[*]}"
echo "  output  : $TMP_REPO"
echo "============================================================"

cd "$UTILS_DIR"

python relabel_bowls_and_merge.py \
    --inputs "${INPUTS[@]}" \
    --output "$TMP_REPO" \
    --allow-unchanged

cd "$REPO_ROOT"

echo ""
echo "Step 1 complete."

# ── Step 2 : Brightness augmentation ─────────────────────────────────────────

echo "============================================================"
echo "Step 2/3 : Brightness-augmenting merged dataset"
echo "  src     : $TMP_REPO"
echo "  dst     : $FINAL_REPO"
echo "  levels  : $BRIGHTNESS_LEVELS"
echo "============================================================"

# shellcheck disable=SC2086
python augmentation/augment_brightness_push.py \
    --src  "$TMP_REPO" \
    --dst  "$FINAL_REPO" \
    --brightness-levels $BRIGHTNESS_LEVELS

echo ""
echo "Step 2 complete."

# ── Step 3 : Clean up temporary repo ─────────────────────────────────────────

echo "============================================================"
echo "Step 3/3 : Deleting temporary repo $TMP_REPO"
echo "============================================================"

python - <<'PYEOF'
import os
from huggingface_hub import HfApi
import sys

tmp = "ETHrobotlearning/colours-task2-premerge"
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
try:
    api.delete_repo(repo_id=tmp, repo_type="dataset")
    print(f"Deleted {tmp}")
except Exception as exc:
    print(f"Warning: could not delete {tmp}: {exc}", file=sys.stderr)
PYEOF

echo ""
echo "============================================================"
echo "Done!  Final dataset: https://huggingface.co/datasets/$FINAL_REPO"
echo "============================================================"
