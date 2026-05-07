# VLA-robotlearning

Vision-language-action (VLA) training pipeline for a pick-and-place task using the SO-101 robot. The repo has two main sections: **data augmentation** (runs locally) and **SmolVLA fine-tuning** (runs on a remote Brev GPU instance).

---

## Repository structure

```
augmentation/       Local data augmentation pipeline (LeRobot format)
training/
  orchestrate.py    Local script — provisions Brev instance, drives the pipeline
  remote_train.sh   Remote script — runs on the GPU instance (do not run locally)
run_augmentation.py Entry point for the augmentation pipeline
requirements.txt    Local Python dependencies
```

---

## 1. Data augmentation

The augmentation pipeline takes a LeRobot-format dataset, applies text and visual augmentations, and pushes the expanded dataset to the Hugging Face Hub.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python run_augmentation.py
```

Configuration (dataset path, HF repo, augmentation parameters) is set in `augmentation/config.py`.

---

## 2. SmolVLA fine-tuning on Brev

`training/orchestrate.py` is a fully automated pipeline that:

1. Provisions a Brev `g5.xlarge` instance (NVIDIA A10G, 24 GB VRAM, 125 GB SSD)
2. Uploads your Hugging Face credentials securely to the instance
3. Installs miniforge, creates a Python 3.12 environment, and installs LeRobot with SmolVLA dependencies — all non-interactively
4. Runs `lerobot-train` to fine-tune [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) on your dataset
5. Uploads the resulting checkpoint to your Hugging Face Hub model repo
6. Deletes the Brev instance automatically (stopping all billing), whether training succeeds or fails

> **Note:** `remote_train.sh` is executed remotely by `orchestrate.py`. You never need to run it directly.

### Prerequisites

**Brev CLI** — install and authenticate once:

```bash
pip install brev
brev login
```

**Hugging Face token** — create a token with write access at <https://huggingface.co/settings/tokens>, then either export it:

```bash
export HF_TOKEN=hf_...
```

or pass it with `--hf-token` each time.

**Your dataset must already be on the Hugging Face Hub** in LeRobot format before running the pipeline. Follow the [LeRobot dataset recording guide](https://huggingface.co/docs/lerobot/il_robots) to collect and push your data.

### Run

```bash
python training/orchestrate.py \
    --dataset-repo-id USERNAME/my-lerobot-dataset \
    --output-repo-id  USERNAME/my-smolvla
```

This will take approximately **4 hours** for the default 20 000 steps on an A10G. Keep the terminal session alive (or run inside a local `tmux` window) for the duration.

### All options

| Flag | Default | Description |
|---|---|---|
| `--dataset-repo-id` | *(required)* | HF Hub dataset to train on, e.g. `user/my-dataset` |
| `--output-repo-id` | *(required)* | HF Hub model repo for the checkpoint, e.g. `user/my-smolvla` |
| `--hf-token` | `$HF_TOKEN` | Hugging Face access token |
| `--train-steps` | `20000` | Number of gradient steps (~4 h on A10G) |
| `--batch-size` | `64` | Training batch size — reduce to `32` if you hit OOM |
| `--instance-name` | `smolvla-training` | Name of the Brev instance |
| `--wandb-enable` | off | Pass this flag to enable Weights & Biases logging |

### Example with all options

```bash
python training/orchestrate.py \
    --dataset-repo-id alessiosalvatore44/so101-pickplace \
    --output-repo-id  alessiosalvatore44/smolvla-so101 \
    --hf-token        hf_... \
    --train-steps     20000 \
    --batch-size      32 \
    --instance-name   smolvla-run1
```

### What happens step by step

```
[local]   brev create smolvla-training --type g5.xlarge
[local]   poll brev ls --json until RUNNING
[local]   brev copy credentials → instance:/tmp/.lerobot_env
[local]   brev exec @training/remote_train.sh  ← blocks here
  [remote]  install miniforge (silent)
  [remote]  conda create -n lerobot python=3.12
  [remote]  conda install ffmpeg
  [remote]  pip install lerobot[smolvla]
  [remote]  huggingface-cli login --token $HF_TOKEN
  [remote]  lerobot-train --policy.path=lerobot/smolvla_base ...
  [remote]  upload checkpoint → HF Hub
  [remote]  rm /tmp/.lerobot_env
[local]   brev delete smolvla-training
```

### Error handling

If any remote step fails (`set -euo pipefail` is active throughout `remote_train.sh`), `brev exec` returns a non-zero exit code, `orchestrate.py` catches the error, **deletes the instance immediately**, and exits with code 1. The same teardown happens on `Ctrl-C`.

