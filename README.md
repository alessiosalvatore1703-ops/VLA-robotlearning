# VLA-robotlearning

Vision-language-action (VLA) training pipeline for pick-and-place tasks on the **SO-101** robot, built on top of [🤗 LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base).

The repo covers the full loop:

1. **Dataset utilities** — relabel, merge, downsample, re-encode and validate LeRobot datasets.
2. **Data augmentation** — text/visual augmentation, brightness partitioning, and an experimental celebrity face-swap pipeline.
3. **SmolVLA fine-tuning** — train on a cloud GPU and push checkpoints to the Hub.

All scripts are run from the repo root and accept either local paths or Hugging Face repo IDs.

---

## Evaluation tasks

The policy is evaluated on three setups of increasing difficulty. The robot is mounted at the edge of a white table with objects placed in a semicircle in front of it; each rollout has a 20-second time limit.

### Task 1 — Color-conditioned pick-and-place

Three bowls are arranged left-to-right (blue, red, green) with a small toy banana placed in front of the arm. Given a prompt of the form **"Put the banana in the [blue/red/green] colored bowl."**, the policy must drop the banana into the bowl of the named color. This is direct color lookup — the prompt names the target color explicitly.

### Task 2 — Compositional instruction following

Same bowl setup (with varying colors), but the prompts require reasoning beyond a direct color lookup, e.g. **"Put the banana into the 2nd bowl from the left."**, **"Put the banana into the bowl on the right of the red bowl."**, or **"Put the banana into the bowl that is not green and not blue."** The exact prompts are not known in advance. The relabeling utilities in `datasets/utils/` (ordinal → color, color → negation, ordinal → relative) generate training data for these phrasings.

### Task 3 — Place the can on a celebrity

DIN A5 portrait prints of celebrities are placed in a semicircle with a 330 ml slim coke can standing in the middle. Given a prompt of the form **"Place the coke on [celebrity name]"**, the policy must place the can on top of the correct portrait. In-distribution identities are Taylor Swift, Barack Obama and Yann LeCun; some rollouts use out-of-distribution celebrities (e.g. Roger Federer, Angela Merkel). The `augmentation/celebrity_swap/` pipeline expands identity coverage by compositing different faces onto the printed cards.

**Autonomous rollouts — prompt: _"Put the coke can on Barack Obama"_**

<table>
  <tr>
    <td width="50%"><img src="https://github.com/alessiosalvatore1703-ops/VLA-robotlearning/raw/main/docs/eval3_coke_obama_1.gif" width="100%" alt="Autonomous rollout 1 — place the coke can on Barack Obama"></td>
    <td width="50%"><img src="https://github.com/alessiosalvatore1703-ops/VLA-robotlearning/raw/main/docs/eval3_coke_obama_2.gif" width="100%" alt="Autonomous rollout 2 — place the coke can on Barack Obama"></td>
  </tr>
</table>

---

## Our approach

All three policies are fine-tuned from [SmolVLA](https://huggingface.co/lerobot/smolvla_base) with [`training/setup_and_train.sh`](training/setup_and_train.sh), which wraps `lerobot-train`. Model size is set by `--policy.num_vlm_layers` (relevant to the smallest-model bonus), each refinement stage starts from a previous checkpoint via `--policy.pretrained_path`, and the vision encoder / VLM are kept frozen (`--policy.freeze_vision_encoder`, `--policy.train_expert_only`) so we mainly train the action expert on top of the pretrained VLM representation. Finished checkpoints are promoted to standalone, loadable policies with [`training/checkpoint_to_policy.py`](training/checkpoint_to_policy.py), and their parameter counts are reported by [`training/count_policy_params.py`](training/count_policy_params.py).

| Task | Policy | Dataset |
|---|---|---|
| 1 | [`ETHrobotlearning/task1-dagger_038000`](https://huggingface.co/ETHrobotlearning/task1-dagger_038000) | [`ETHrobotlearning/tv-task1-clean-4prompts-fixed`](https://huggingface.co/datasets/ETHrobotlearning/tv-task1-clean-4prompts-fixed) |
| 2 | [`ETHrobotlearning/smolvla_task2_colors_dagger_lr2e-5-step2000`](https://huggingface.co/ETHrobotlearning/smolvla_task2_colors_dagger_lr2e-5-step2000) | [`ETHrobotlearning/tv-task2-clean-aug5p-fixed`](https://huggingface.co/datasets/ETHrobotlearning/tv-task2-clean-aug5p-fixed) |
| 3 | [`Alessio03/smolvla-task3-50k`](https://huggingface.co/Alessio03/smolvla-task3-50k) | [`ETHrobotlearning/task3-TOY-clean`](https://huggingface.co/datasets/ETHrobotlearning/task3-TOY-clean) |

### Task 1

For Task 1 we used a straightforward imitation-learning setup: we collected demonstrations and trained a SmolVLA policy directly on them. Since the task is relatively simple, we used only **8 VLM layers**, as it does not require a highly expressive vision-language backbone. We then used **DAgger** to refine the policy: after rolling out the trained model, we collected corrective demonstrations for the failure cases and added them back into training.

*In this repo:* demonstrations are cleaned of frozen-action frames with [`trim_and_push.py`](trim_and_push.py) (the `clean` in the dataset name) and expanded to several prompt phrasings (the `4prompts`); the policy is trained at `--policy.num_vlm_layers=8`. The DAgger loop is a workflow rather than a single script — corrective episodes are folded in with [`datasets/utils/merge_datasets.py`](datasets/utils/merge_datasets.py) and training is continued from the prior checkpoint via `--policy.pretrained_path` (the `dagger_038000` checkpoint).

### Task 2

For Task 2 we increased the complexity of the training data. We collected multiple demonstrations and augmented the language supervision by associating each episode with several prompt formulations (~5 prompts per episode), exposing the policy to more linguistic variety while keeping the same underlying behavior. We trained a standard SmolVLA policy with a **16-layer** VLM backbone and used it as a stronger base policy for further fine-tuning. We again applied **DAgger** — evaluating the policy, identifying weak or inconsistent behaviors, collecting corrective demonstrations, and fine-tuning on those examples. This refinement can be seen as a form of **curriculum learning**: the policy is first trained on a broad dataset and then specialized on harder cases.

*In this repo:* the compositional prompt variants are generated by the relabeling utilities in [`datasets/utils/`](datasets/utils/) — ordinal→colour ([`relabel_bowls_and_merge.py`](datasets/utils/relabel_bowls_and_merge.py)), colour→negation ([`color_prompts_to_negation.py`](datasets/utils/color_prompts_to_negation.py)), and ordinal→relative ([`relabel_relative_bowls_and_merge.py`](datasets/utils/relabel_relative_bowls_and_merge.py)); the ~5-prompt expansion is the `aug5p` in the dataset name, with visual robustness added by [`augment_brightness_push.py`](augment_brightness_push.py). Training uses `--policy.num_vlm_layers=16`, and the DAgger stage fine-tunes from the previous checkpoint via `--policy.pretrained_path` (the `dagger_lr2e-5-step2000` run).

### Task 3

For Task 3 we followed the same general strategy as Task 2: collecting demonstrations, increasing prompt diversity, training a SmolVLA policy, and fine-tuning from a stronger pretrained checkpoint. We also experimented with fine-tuning the VLM backbone itself, but this did not lead to clear improvements — so we kept the Task 2 methodology, focusing on training and fine-tuning the action policy while relying on the pretrained VLM representation.

*In this repo:* identity diversity is expanded with the [`augmentation/celebrity_swap/`](augmentation/celebrity_swap/) pipeline, which composites different celebrity faces onto the printed cards; the policy is trained for 50k steps with `setup_and_train.sh`. The "fine-tune the VLM backbone" experiment corresponds to relaxing `--policy.freeze_vision_encoder` / `--policy.train_expert_only`, which we reverted after seeing no clear gain.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Datasets, caches, model weights and training outputs are not tracked in git** (see `.gitignore`). They are re-pullable from the [`ETHrobotlearning`](https://huggingface.co/ETHrobotlearning) Hub org or regenerated by the scripts below. Keep the working tree to source only.

---

## Evaluation Submission

This repository includes one runnable script per evaluation task. Each script:

1. checks that `lerobot-rollout` is available,
2. installs a local evaluation environment if needed,
3. prints the selected policy checkpoint and parameter count,
4. runs a normal SO-101 rollout with the requested task prompt.

Final policy checkpoints are not committed to this repository. They are loaded from the Hugging Face Hub by default. If you want to pre-download them, run:

```bash
python scripts/download_eval_policies.py
```

That command creates a local `policy_checkpoints/` cache, which is ignored by git. The evaluation scripts use the local cache when present and otherwise fall back to the explicit Hub model IDs below:

| Eval | Local cache path | Hub model ID | Default prompt |
|---|---|---|---|
| Task 1 | `policy_checkpoints/task1` | `ETHrobotlearning/task1-dagger_038000` | `Place the banana to the red colored bowl.` |
| Task 2 | `policy_checkpoints/task2` | `ETHrobotlearning/smolvla_task2_colors_dagger_lr2e-5-step2000` | `Put the banana in the green colored bowl.` |
| Task 3 | `policy_checkpoints/task3` | `Alessio03/smolvla-task3-50k` | `Place the coke on Yann LeCun.` |

### Hardware Defaults

The scripts assume the same SO-101 setup used for collection/evaluation:

```bash
ROBOT_PORT=/dev/tty.usbmodem5B140319121
ROBOT_ID=my_awesome_follower_arm
POLICY_DEVICE=mps
FPS=10
DURATION=20
```

Override any value with an environment variable, for example:

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export POLICY_DEVICE=cuda
export DURATION=40
```

### Run Evaluation

Run the default prompt for each task:

```bash
./run_eval_1.sh
./run_eval_2.sh
./run_eval_3.sh
```

Run with a custom prompt by passing it as the first argument:

```bash
./run_eval_2.sh "Put the banana into the bowl that is not red and not blue."
./run_eval_3.sh "Place the coke on Barack Obama."
```

The scripts return the robot to its startup joint position during teardown with:

```bash
--return_to_initial_position=true
```

### Policy Method Summary

For Task 1, we collected demonstrations and trained SmolVLA directly with an 8-layer VLM backbone, since the task did not require a highly expressive visual-language representation. For Task 2, we trained on multiple demonstrations augmented with diverse prompt variants, roughly five prompts per episode, then used a standard 16-layer SmolVLA backbone. The resulting policy was treated as a base policy and further fine-tuned on weaker cases, which acts as a small curriculum-learning pipeline. For Task 3, we followed the same methodology as Task 2. We also tried co-training/fine-tuning the VLM backbone, but this did not improve performance, so the final approach kept the pretrained VLM representation and focused on action-policy fine-tuning.

---

## Repository structure

```
augment_brightness_push.py    Brightness-partition augmentation of a v3 Hub dataset
trim_and_push.py              Trim frozen-action frames from a Hub dataset, re-push
run_augmentation.py           Entry point for the text/visual augmentation pipeline
run_benchmark.py              Entry point for the VLM grounding benchmark
run_merge_augment_colours.sh  End-to-end: relabel → merge → brightness for the 6 colour configs
debug_pipeline.py             Read-only pre-flight checks for the merge→augment pipeline
preview_brightness.py         Preview brightness levels on a single video
validate_final_dataset.py     Post-pipeline validation of the merged colours dataset
training_smolvla.py           Reference Colab notebook export (SmolVLA training)
requirements.txt              Local Python dependencies

augmentation/                 Text + visual augmentation pipeline (LeRobot format)
  config.py, pipeline.py, text_augment.py, visual_augment.py, dataset_io.py, validate.py
  celebrity_swap/             Experimental face-swap augmentation (own module CLI)

bank/                         Celebrity portrait textures + names.txt (inputs for celebrity_swap)

benchmarks/
  celebrity_recognition/      VLM spatial-grounding benchmark (see "VLM backbone study")

datasets/
  utils/                      Dataset transform/merge utilities (see "Dataset utilities")

training/
  setup_and_train.sh          Standalone fine-tuning script — run on a fresh GPU instance
  orchestrate.py              Local automation — provisions a Brev instance and drives training
  remote_train.sh             Remote script driven by orchestrate.py (do not run locally)
  checkpoint_to_policy.py     Convert a checkpoint repo into a standalone, loadable policy repo
  count_policy_params.py      Print a policy's parameter count and run lerobot-rollout
  orchestrate_molmoact2.py    Local Brev runner for MolmoAct2 SO100/SO101 fine-tuning
  setup_and_train_molmoact2.sh
                              Standalone MolmoAct2 setup + training script for Brev H100

lerobot-doctor/               Vendored dataset-quality diagnostics tool
tracelr/                      Git submodule — desktop episode viewer / annotation tool
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

The benchmark code lives in `benchmarks/celebrity_recognition/`:

```bash
python run_benchmark.py --model smolvlm_256m --dataset benchmarks/celebrity_recognition/data/dataset.csv
python run_benchmark.py --model smolvlm_500m --dataset benchmarks/celebrity_recognition/data/dataset.csv
```

---

## 1. Dataset utilities

All scripts live in `datasets/utils/`. They accept local paths or Hugging Face repo IDs as input/output.

### check_datasets_uniform.py — verify format compatibility

```bash
python datasets/utils/check_datasets_uniform.py user/ds1 user/ds2
```

Compares fps, robot_type, feature keys/dtypes/shapes, and video codec across all listed datasets. Exits 0 if all fields match, 1 otherwise.

### downsample_fps.py — reduce frame rate

```bash
python datasets/utils/downsample_fps.py --input user/my_dataset --output my_dataset_10fps --fps 10
```

Supports LeRobot v2.x and v3.x. Re-encodes videos and rewrites parquet indices.

### to_av1.py — re-encode videos to AV1

```bash
python datasets/utils/to_av1.py --input user/my_dataset --output my_dataset_av1
```

Uses SVT-AV1 via ffmpeg. Optional `--crf` (default 30) and `--preset` (default 8, range 0–13).

### retask.py — replace the task prompt

```bash
python datasets/utils/retask.py --input user/my_dataset --output my_dataset_retask \
    --task "Put the banana in the green bowl."
```

Single-prompt datasets only. Rewrites `tasks.parquet` and the `tasks` column in every episode parquet.

### merge_datasets.py — merge multiple datasets

```bash
python datasets/utils/merge_datasets.py --inputs user/ds1 user/ds2 user/ds3 --output merged_dataset
```

Requires the datasets to pass the uniformity check (fps, features, codec). Tasks need not match — each source's prompts are merged into a unified task list and `task_index` values are remapped automatically.

### relabel_bowls_and_merge.py — ordinal bowl prompts → colour prompts, then merge

```bash
python datasets/utils/relabel_bowls_and_merge.py --output ETHrobotlearning/banana-bowls-color-prompts
```

Defaults to the six `ETHrobotlearning/config*-...` datasets. Each config name encodes bowl colours left→right, so `config1-red-blue-green` maps `1st bowl`→red, `2nd bowl`→blue, `3rd bowl`→green. Rewrites prompts like `Put the banana into the 2nd bowl from the left ...` into `Put the banana in the blue colored bowl`, then merges the relabeled datasets.

### color_prompts_to_negation.py — colour prompts → negation prompts

```bash
python datasets/utils/color_prompts_to_negation.py
```

Defaults `ETHrobotlearning/task2-colors` → `ETHrobotlearning/task2-negation`. Rewrites `Put the banana in the red colored bowl` into `Put the banana into the bowl that is not green and not blue.`, leaving frames/videos unchanged.

### relabel_relative_bowls_and_merge.py — ordinal bowl prompts → relative prompts

```bash
python datasets/utils/relabel_relative_bowls_and_merge.py
```

Defaults the six `config*` datasets → `ETHrobotlearning/task2-relative`. Creates adjacent-reference prompts such as `Put the banana into the bowl on the right of the red bowl ...`. Middle-bowl targets are duplicated so both valid phrasings are present.

### make_training_ready.py — repair stats/metadata for training

```bash
# local → local
python datasets/utils/make_training_ready.py --source dataset --output dataset_trainready

# HF repo → local → push back
python datasets/utils/make_training_ready.py --source ETHrobotlearning/my-dataset --output /tmp/fixed --push
```

Recomputes `meta/stats.json` and rewrites `meta/info.json` (`total_*`, `splits`) so a structurally-valid LeRobot v3 dataset becomes training-ready for SmolVLA. SmolVLA loads normalization stats from `meta/stats.json`; without it, training crashes or trains unnormalized, and a `splits["train"]` that doesn't cover every episode silently hides data from the dataloader.

### Dataset diagnostics — lerobot-doctor

`lerobot-doctor/` is a vendored copy of [lerobot-doctor](https://github.com/jashshah999/lerobot-doctor). It catches corrupted timestamps, dropped/frozen frames, clipped values, metadata inconsistencies and video problems.

```bash
pip install ./lerobot-doctor
lerobot-doctor /path/to/dataset      # or: lerobot-doctor user/my_hf_dataset
```

---

## 2. Data augmentation

### Text + visual pipeline

Applies text and visual augmentations to a LeRobot dataset and optionally pushes the expanded dataset to the Hub. Augmentation parameters default from `augmentation/config.py`.

```bash
python run_augmentation.py --input /path/to/eval1_raw --output /path/to/eval1_augmented \
    --text-variants 5 --visual-variants 3 \
    --push-to-hub --hub-repo-id your-username/eval1-augmented
```

### Brightness augmentation — augment_brightness_push.py

```bash
python augment_brightness_push.py --src user/my_dataset --dst user/my_dataset_bright \
    --brightness-levels 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
```

Pulls a LeRobot v3 Hub dataset, splits the source episodes into N contiguous partitions (N = number of brightness levels), assigns each partition one brightness multiplier, and emits one augmented copy per source episode (final size = 2 × source). Only pixels change — actions, `observation.state`, timestamps, `task_index` and per-episode stats pass through unchanged. Videos are re-encoded with SVT-AV1. Use `preview_brightness.py --src ...` to eyeball a brightness range on one video first.

### Frozen-frame trimming — trim_and_push.py

```bash
python trim_and_push.py --src user/my_dataset --dst user/my_dataset_clean
```

Trims leading/trailing frames where the arm is not yet moving (frozen actions) from every episode of a LeRobot v3 Hub dataset, then pushes the cleaned dataset. Episodes shorter than 50 frames after trimming are left untouched. Videos re-encoded with SVT-AV1.

### Colours pipeline — run_merge_augment_colours.sh

```bash
bash run_merge_augment_colours.sh
```

End-to-end for the six colour configs: (1) relabel ordinal→colour prompts and merge the configs into a temporary repo, (2) brightness-augment into the final `ETHrobotlearning/colours-task2`, (3) delete the temporary repo. Relabelling must precede augmentation because the colour parser reads config names from the repo ID. Run `debug_pipeline.py` for read-only pre-flight checks and `validate_final_dataset.py` afterwards to verify the result.

### Celebrity face-swap (experimental) — augmentation/celebrity_swap/

Detects portrait cards in episode frames and composites a different celebrity face onto them, multiplying identity diversity. Run as a module from the repo root:

```bash
# 1. One-time: build the celeb texture bank from ielminawi/celeb30 → bank/textures/
python -m augmentation.celebrity_swap.prepare_celeb_bank

# 2. Visualise portrait detection on random episodes (no swapping)
python -m augmentation.celebrity_swap.run_pipeline viz --src ETHrobotlearning/task3-TOY-clean --n-episodes 20

# 3. Run the swap pipeline
python -m augmentation.celebrity_swap.run_pipeline run \
    --src ETHrobotlearning/task3-TOY-clean --dst ./output/task3-celebswap --n-aug 3 --workers 4

# 4. Push the finished dataset to the Hub
python -m augmentation.celebrity_swap.run_pipeline push \
    --dst ./output/task3-celebswap --hub-repo ETHrobotlearning/task3-TOY-celebswap
```

Extra dependencies are listed in `augmentation/celebrity_swap/requirements_swap.txt`.

---

## 3. SmolVLA fine-tuning

The training dataset must already be on the Hugging Face Hub in LeRobot format. Create a write-enabled HF token at <https://huggingface.co/settings/tokens>. **Never commit a real token** — `setup_and_train.sh` reads `HF_TOKEN` / `WANDB_API_KEY` from the environment.

### Option A — standalone script (recommended)

`training/setup_and_train.sh` runs on a fresh GPU instance (e.g. a Brev/Nebius **H100**). SSH in, export your credentials (or edit the CONFIG block at the top), then run it:

```bash
export HF_TOKEN=hf_...            # write access
export WANDB_API_KEY=...          # only if WANDB_ENABLE="true"
bash training/setup_and_train.sh
```


It is idempotent and runs six stages: install miniforge → create a Python 3.12 `lerobot` env → install LeRobot + SmolVLA extras → authenticate with HF (and optionally W&B) → download the dataset locally → run `lerobot-train` on `lerobot/smolvla_base`. A background watcher pushes each new checkpoint to `OUTPUT_REPO_ID` as it is written, then frees the local copy. Edit `DATASET_REPO_ID`, `OUTPUT_REPO_ID`, `TRAIN_STEPS`, `BATCH_SIZE`, `LR`, `SAVE_FREQ`, `RESUME` and `WANDB_ENABLE` in the CONFIG block.

> No episode-level train/val split is applied — LeRobot's frame sampler does not respect the episodes filter reliably. Monitor overfitting via the W&B training loss and evaluate on the real robot.

### Option B — automated Brev orchestration

`training/orchestrate.py` provisions a Brev instance, writes credentials over SSH, drives `remote_train.sh`, and **deletes the instance on completion or failure** (stopping billing). It uses only the Python standard library — no local venv needed. `remote_train.sh` is executed remotely; never run it directly.

```bash
pip install brev && brev login
export HF_TOKEN=hf_...
python training/orchestrate.py \
    --dataset-repo-id USERNAME/my-lerobot-dataset \
    --output-repo-id  USERNAME/my-smolvla
```

| Flag | Default | Description |
|---|---|---|
| `--dataset-repo-id` | *(required)* | HF Hub dataset to train on |
| `--output-repo-id` | *(required)* | HF Hub model repo for the checkpoint |
| `--hf-token` | `$HF_TOKEN` | Hugging Face access token |
| `--train-steps` | `20000` | Number of gradient steps |
| `--batch-size` | `64` | Training batch size — reduce to `32` on OOM |
| `--instance-name` | `smolvla-training` | Brev instance name |
| `--wandb-enable` | off | Enable Weights & Biases logging |
| `--wandb-api-key` | `$WANDB_API_KEY` | Required when `--wandb-enable` is set ([wandb.ai/settings](https://wandb.ai/settings)) |

### Post-training utilities

```bash
# Promote one checkpoint into a clean, loadable policy repo
python training/checkpoint_to_policy.py \
    --source USER/smolvla-run --checkpoint 005000 --output USER/smolvla-final --token hf_...

# Inspect a policy's parameter count (and run a rollout)
python training/count_policy_params.py USER/smolvla-final
```

The output of `checkpoint_to_policy.py` is loadable directly via `--policy.pretrained_path=<output>`.

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

---

## 4. MolmoAct2 SO100/SO101 fine-tuning on Brev

MolmoAct2 training uses the Ai2 MolmoAct2 LeRobot fork, not upstream
Hugging Face LeRobot. The helper scripts are:

- `training/orchestrate_molmoact2.py` — local Brev provisioner/runner
- `training/setup_and_train_molmoact2.sh` — standalone script to run inside a Brev H100 instance

Default MolmoAct2 settings:

| Setting | Value |
|---|---|
| Dataset | `ETHrobotlearning/task3-TOY-clean` |
| Initial checkpoint | `allenai/MolmoAct2-SO100_101` |
| Camera key | `["observation.images.front"]` |
| Training mode | VLM LoRA + fully trainable action expert |
| Batch size | `32` |
| Steps | `50000` |
| Checkpoint frequency | every `5000` steps |
| Action chunk | `10` |
| Action mode | `continuous` |
| Setup/control prompt | `single SO-100/SO-101 arm with one front RGB camera` / `absolute joint pose` |

### Local Brev orchestration

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...

python training/orchestrate_molmoact2.py \
    --output-repo-id ETHrobotlearning/molmoact2-task3-toy-lora
```

This creates a Brev H100 instance, installs MolmoAct2/LeRobot, launches
training, pushes checkpoints to:

```text
ETHrobotlearning/molmoact2-task3-toy-lora-step5000
ETHrobotlearning/molmoact2-task3-toy-lora-step10000
...
ETHrobotlearning/molmoact2-task3-toy-lora-step50000
```

and then deletes the instance unless `--keep-instance` is passed.

### Standalone script inside Brev

On a fresh Brev H100 instance:

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export OUTPUT_REPO_ID=ETHrobotlearning/molmoact2-task3-toy-lora

bash setup_and_train_molmoact2.sh
```

Useful overrides:

```bash
BATCH_SIZE=16 TRAIN_STEPS=1000 WANDB_ENABLE=false PUSH_TO_HUB=false \
  bash setup_and_train_molmoact2.sh
```

For a rerun on the same instance after setup has completed once:

```bash
SKIP_SETUP=true SKIP_DATASET_DOWNLOAD=true bash setup_and_train_molmoact2.sh
```
