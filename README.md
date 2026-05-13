# VLA-robotlearning

Vision-language-action (VLA) training pipeline for a pick-and-place task using the SO-101 robot. The repo has two main sections: **data augmentation** (runs locally) and **SmolVLA fine-tuning** (runs on a remote Brev GPU instance).

---

## Repository structure

```
augmentation/         Local data augmentation pipeline (LeRobot format)
benchmarks/
  celebrity_recognition/  VLM spatial-grounding benchmark (see VLM backbone study)
datasets/
  utils/
    check_datasets_uniform.py  Verify that multiple datasets share the same format
    downsample_fps.py          Downsample a dataset to a lower FPS
    to_av1.py                  Re-encode video streams from H.264 to AV1
    retask.py                  Replace the task prompt of a single-task dataset
    merge_datasets.py          Merge multiple datasets into one
lerobot-doctor/       Dataset quality diagnostics tool (vendored)
tracelr/              Desktop episode viewer and annotation tool (own README)
training/
  orchestrate.py      Local script — provisions Brev instance, drives the pipeline
  remote_train.sh     Remote script — runs on the GPU instance (do not run locally)
run_augmentation.py   Entry point for the augmentation pipeline
run_benchmark.py      Entry point for VLM benchmarks
trim_and_push.py      Trim frozen-action frames from a Hub dataset and re-push
requirements.txt      Local Python dependencies
```

---

## VLM backbone study

To evaluate whether a small VLM could act as a perception backbone for the robot (e.g. "locate object X and move near it"), we built a spatial-grounding benchmark: four celebrity cards are arranged in a 2×2 grid on a table, and the model must name the quadrant (`top_left`, `top_right`, `bottom_left`, `bottom_right`) containing a given person.

**Models tested:** `SmolVLM-256M-Instruct` and `SmolVLM-500M-Instruct` (HuggingFaceTB).

| Model | Accuracy (20 samples) | Observed behaviour |
|---|---|---|
| SmolVLM-256M | 25% | Almost always predicts `top_left` regardless of input |
| SmolVLM-500M | 25% | Almost always predicts `top_right` regardless of input |

Both models score at chance level (random = 25%) and show no celebrity understanding: they ignore the identity prompt entirely and default to a fixed positional bias. Neither is viable as a VLM backbone without substantial fine-tuning.

The benchmark code lives in `benchmarks/celebrity_recognition/`. Run it with:

```bash
python run_benchmark.py --model smolvlm_256m --dataset benchmarks/celebrity_recognition/data/dataset.csv
python run_benchmark.py --model smolvlm_500m --dataset benchmarks/celebrity_recognition/data/dataset.csv
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

### trim_and_push.py — remove frozen-action frames

```bash
python trim_and_push.py --src user/my_dataset --dst user/my_dataset_clean
```

Trims leading and trailing frames where the robot arm is not yet moving (frozen actions) from every episode in a LeRobot v3 Hub dataset, then pushes the cleaned dataset to a new HF repo. Episodes shorter than 50 frames after trimming are left untouched. Videos are re-encoded with SVT-AV1.

---

## 1b. Dataset diagnostics — lerobot-doctor

`lerobot-doctor/` is a vendored copy of the [lerobot-doctor](https://github.com/jashshah999/lerobot-doctor) tool. It catches common dataset quality issues: corrupted timestamps, dropped frames, frozen actions, clipped values, metadata inconsistencies, and video problems.

```bash
pip install lerobot-doctor          # or: pip install ./lerobot-doctor
lerobot-doctor /path/to/dataset
lerobot-doctor user/my_hf_dataset
```

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

