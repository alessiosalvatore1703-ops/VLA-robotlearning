# VLA-robotlearning

Vision-language-action (VLA) training pipeline for a pick-and-place task using the SO-101 robot. The repo has two main sections: **data augmentation** (runs locally) and **SmolVLA fine-tuning** (runs on a remote Brev GPU instance).

---

## Repository structure

```
augmentation/         Local data augmentation pipeline (LeRobot format)
datasets/
  utils/
    check_datasets_uniform.py  Verify that multiple datasets share the same format
    downsample_fps.py          Downsample a dataset to a lower FPS
    to_av1.py                  Re-encode video streams from H.264 to AV1
    retask.py                  Replace the task prompt of a single-task dataset
    merge_datasets.py          Merge multiple datasets into one
training/
  orchestrate.py      Local script — provisions Brev instance, drives the pipeline
  remote_train.sh     Remote script — runs on the GPU instance (do not run locally)
run_augmentation.py   Entry point for the augmentation pipeline
requirements.txt      Local Python dependencies
```

---

## 1. Dataset utilities

All scripts live in `datasets/utils/` and are run from the repo root. They all accept local paths or Hugging Face repo IDs as input/output.

### check_datasets_uniform.py — verify format compatibility

```bash
python datasets/utils/check_datasets_uniform.py user/ds1 user/ds2
```

Compares fps, robot_type, feature keys/dtypes/shapes, and video codec across all listed datasets. Exits 0 if all fields match, 1 otherwise.

### downsample_fps.py — reduce frame rate

```bash
python datasets/utils/downsample_fps.py \
    --input  user/my_dataset \
    --output my_dataset_10fps \
    --fps    10
```

Supports LeRobot v2.x and v3.x. Re-encodes videos and rewrites parquet indices.

### to_av1.py — re-encode videos to AV1

```bash
python datasets/utils/to_av1.py \
    --input  user/my_dataset \
    --output my_dataset_av1
```

Uses SVT-AV1 via ffmpeg. Optional `--crf` (default 30) and `--preset` (default 8, range 0–13).

### retask.py — replace the task prompt

```bash
python datasets/utils/retask.py \
    --input  user/my_dataset \
    --output my_dataset_retask \
    --task   "Put the banana in the green bowl."
```

Works only on single-prompt datasets. Rewrites `tasks.parquet` and the `tasks` column in every episode parquet.

### merge_datasets.py — merge multiple datasets

```bash
python datasets/utils/merge_datasets.py \
    --inputs user/ds1 user/ds2 user/ds3 \
    --output merged_dataset
```

Requires datasets to pass the uniformity check (fps, features, codec). Tasks do not need to match — each source's prompts are merged into a unified task list and `task_index` values are remapped automatically.

---

## 2. Data augmentation

The augmentation pipeline takes a LeRobot-format dataset, applies text and visual augmentations, and pushes the expanded dataset to the Hugging Face Hub.

### Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python run_augmentation.py
```

Configuration (dataset path, HF repo, augmentation parameters) is set in `augmentation/config.py`.

---

## 3. SmolVLA fine-tuning on Brev

`training/orchestrate.py` is a fully automated pipeline that:

1. Provisions a Brev `g5.xlarge` instance (NVIDIA A10G, 24 GB VRAM, 125 GB SSD)
2. Writes your credentials securely to the instance via SSH (no `brev copy`/SCP)
3. Installs miniforge, creates a Python 3.12 environment, and installs LeRobot with SmolVLA, dataset, wandb, and PyAV dependencies — all non-interactively
4. Runs `lerobot-train` to fine-tune [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) on your dataset, and pushes the checkpoint directly to your HF Hub model repo
5. Deletes the Brev instance automatically (stopping all billing), whether training succeeds or fails

> **Note:** `remote_train.sh` is executed remotely by `orchestrate.py`. You never need to run it directly.

### Prerequisites

**No local virtual environment needed** — `orchestrate.py` uses only Python standard library modules. The remote instance sets up its own isolated conda environment from scratch.

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
| `--wandb-api-key` | `$WANDB_API_KEY` | WandB API key — required when `--wandb-enable` is set. Get it at [wandb.ai/settings](https://wandb.ai/settings) |

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
[local]   poll brev ls --json until RUNNING  (+30 s SSH grace period)
[local]   brev refresh  (update SSH alias to current hostname)
[local]   ssh smolvla-training 'bash -s' < preamble.sh  ← writes /tmp/.lerobot_env
[local]   ssh smolvla-training 'bash -s' < remote_train.sh  ← blocks here (~hours)
  [remote]  curl miniforge installer + bash install
  [remote]  conda create -n lerobot python=3.12 pip
  [remote]  pip install lerobot[smolvla,dataset] wandb av
  [remote]  hf auth login --token $HF_TOKEN
  [remote]  wandb login (up to 5 retries for network readiness)
  [remote]  lerobot-train --policy.type=smolvla
                          --policy.pretrained_path=lerobot/smolvla_base
                          --dataset.video_backend=pyav
                          --policy.push_to_hub=true ...
  [remote]  rm /tmp/.lerobot_env
[local]   brev delete smolvla-training
```

### Using Weights & Biases

Pass both flags together:

```bash
python training/orchestrate.py \
    --dataset-repo-id USERNAME/my-dataset \
    --output-repo-id  USERNAME/my-smolvla \
    --wandb-enable \
    --wandb-api-key   YOUR_WANDB_API_KEY   # or export WANDB_API_KEY=... beforehand
```

Find your API key at <https://wandb.ai/settings>. The script will exit with an error if `--wandb-enable` is set but no key is provided.

The remote script runs `wandb login` with up to 5 retries (20 s apart) before training starts, because Brev instances sometimes take a minute to reach `api.wandb.ai` after boot. `WANDB_INIT_TIMEOUT` and `WANDB_HTTP_TIMEOUT` are also increased to tolerate slow cold-start networks.

### Error handling

If any remote step fails (`set -euo pipefail` is active throughout `remote_train.sh`), `brev exec` returns a non-zero exit code, `orchestrate.py` catches the error, **deletes the instance immediately**, and exits with code 1. The same teardown happens on `Ctrl-C`.

