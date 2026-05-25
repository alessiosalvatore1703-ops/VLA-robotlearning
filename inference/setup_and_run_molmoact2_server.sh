#!/usr/bin/env bash
set -euo pipefail

# Install the dependencies needed for MolmoAct2 inference on a Brev Ubuntu
# instance, then start the FastAPI server.
#
# Usage on Brev:
#   bash ~/setup_and_run_molmoact2_server.sh
#
# Optional overrides:
#   SERVER_SCRIPT=~/molmoact2_server.py
#   VENV_DIR=~/venvs/molmoact2
#   HOST=127.0.0.1
#   PORT=8000
#   DTYPE=bfloat16
#   REPO_ID=allenai/MolmoAct2-SO100_101
#   NORM_TAG=so100_so101_molmoact2
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#   REINSTALL_TORCH=true
#   DISABLE_CUDA_GRAPH=true

SERVER_SCRIPT="${SERVER_SCRIPT:-$HOME/molmoact2_server.py}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/molmoact2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
REPO_ID="${REPO_ID:-allenai/MolmoAct2-SO100_101}"
NORM_TAG="${NORM_TAG:-so100_so101_molmoact2}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
REINSTALL_TORCH="${REINSTALL_TORCH:-false}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-false}"

if [[ ! -f "$SERVER_SCRIPT" ]]; then
  echo "ERROR: server script not found at: $SERVER_SCRIPT"
  echo "Copy it first, for example:"
  echo "  scp inference/molmoact2_server.py <brev-host>:~/molmoact2_server.py"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed."
  exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "Installing python3-venv and python3-pip with apt..."
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [[ "$REINSTALL_TORCH" == "true" ]]; then
  python -m pip uninstall -y torch torchvision torchaudio || true
fi

if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "CUDA-enabled PyTorch already works."
else
  echo "Installing CUDA-compatible PyTorch from: $TORCH_INDEX_URL"
  python -m pip uninstall -y torch torchvision torchaudio || true
  python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

python -m pip install \
  transformers \
  pillow \
  numpy \
  huggingface_hub \
  hf_transfer \
  fastapi \
  "uvicorn[standard]" \
  pydantic \
  accelerate \
  safetensors \
  einops \
  requests

# Guard against dependency resolution replacing the CUDA-compatible PyTorch
# stack after installing the non-torch packages above.
if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "CUDA-enabled PyTorch still works after dependency install."
else
  echo "Reinstalling PyTorch/torchvision from $TORCH_INDEX_URL after dependency install."
  python -m pip uninstall -y torch torchvision torchaudio || true
  python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    raise SystemExit(
        "ERROR: CUDA is not available from PyTorch. "
        "Use a GPU Brev instance or reinstall a CUDA-enabled torch wheel."
    )
PY

export HF_HUB_ENABLE_HF_TRANSFER=1

server_args=(
  "$SERVER_SCRIPT"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --repo-id "$REPO_ID"
  --norm-tag "$NORM_TAG"
)

if [[ "$DISABLE_CUDA_GRAPH" == "true" ]]; then
  server_args+=(--disable-cuda-graph)
fi

echo "Starting MolmoAct2 server:"
echo "  repo: $REPO_ID"
echo "  host: $HOST"
echo "  port: $PORT"
echo "  dtype: $DTYPE"
echo
echo "Health check once running:"
echo "  curl http://127.0.0.1:$PORT/health"
echo

exec python "${server_args[@]}"
