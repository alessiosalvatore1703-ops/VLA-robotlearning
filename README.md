# VLA-robotlearning

**Vision-language-action policies for language-conditioned pick-and-place on the SO-101 arm.**

We fine-tune [SmolVLA](https://huggingface.co/lerobot/smolvla_base) into three policies of increasing difficulty — from naming a color, to reasoning about spatial relations between bowls, to placing a can on the portrait of a *named celebrity*. This repository is the record of the **design decisions** behind those policies: how we shaped the data, sized the models, refined behavior with DAgger, and stress-tested what a small VLM can and cannot ground.

Built on [🤗 LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base). Every dataset and checkpoint we reference lives on the [`ETHrobotlearning`](https://huggingface.co/ETHrobotlearning) Hub org.

**Autonomous rollout — _"Put the coke can on Barack Obama"_** (in-distribution identity)

<table>
  <tr>
    <td width="50%"><img src="https://github.com/alessiosalvatore1703-ops/VLA-robotlearning/raw/main/docs/eval3_coke_obama_1.gif" width="100%" alt="Autonomous rollout 1 — place the coke can on Barack Obama"></td>
    <td width="50%"><img src="https://github.com/alessiosalvatore1703-ops/VLA-robotlearning/raw/main/docs/eval3_coke_obama_2.gif" width="100%" alt="Autonomous rollout 2 — place the coke can on Barack Obama"></td>
  </tr>
</table>

**Autonomous rollout — _"Put the coke can on Michael Jackson"_** (out-of-distribution identity)

Michael Jackson was never in the training identities (Taylor Swift, Barack Obama, Yann LeCun) — this rollout shows the policy generalizing to an **OOD** celebrity portrait it never saw during data collection.

<p align="center">
  <img src="https://github.com/alessiosalvatore1703-ops/VLA-robotlearning/raw/main/docs/eval3_coke_michael_jackson.gif" width="70%" alt="Autonomous rollout — place the coke can on Michael Jackson (out-of-distribution identity)">
</p>

---

## The challenge

The robot sits at the edge of a white table with objects placed in a semicircle in front of it. Each rollout has a 20-second time limit. We targeted three setups, each one a step harder than the last in *what the language has to do*.

| | Task | What the prompt asks | Why it's harder |
|---|---|---|---|
| **1** | Color-conditioned pick-and-place | *"Place the banana to the **red** colored bowl."* | Direct lookup — the prompt names the target color. |
| **2** | Compositional instruction following | *"Put the banana into the **2nd bowl from the left**"*, *"…the bowl **on the right of the red bowl**"*, *"…the bowl that is **not green and not blue**"* | The policy must resolve ordinal, relative and negated references — and the exact phrasings aren't known in advance. |
| **3** | Place the can on a celebrity | *"Place the coke on **Barack Obama**"* | The policy must locate the right face among portrait cards, including **out-of-distribution** identities never seen in training. |

For Task 3, DIN A5 portrait prints are arranged in a semicircle with a 330 ml slim coke can in the middle. In-distribution identities are Taylor Swift, Barack Obama and Yann LeCun; some rollouts deliberately use OOD celebrities such as Michael Jackson, Roger Federer and Angela Merkel.

---

## How we approached it

Rather than treat each task as a one-off, we settled on a small set of principles and applied them with increasing intensity as the tasks got harder. These are the choices that mattered.

### 1. Train the action expert, trust the pretrained VLM

For every policy we keep the vision encoder and the VLM backbone **frozen** and train only the action expert on top of the pretrained representation (`--policy.freeze_vision_encoder`, `--policy.train_expert_only`). The bet is that SmolVLA's pretrained perception is already good enough to ground our prompts, and that the hard part is mapping that representation to SO-101 actions. On Task 3 we explicitly tested the opposite — unfreezing the VLM backbone — and saw **no clear improvement**, so we reverted. (Our [VLM backbone study](#did-we-even-need-the-vlm-to-be-smart) below is the wider version of that question.)

### 2. Right-size the model to the task

Model capacity is set by `--policy.num_vlm_layers`, and we deliberately spent only as much as each task needed:

- **Task 1** uses **8 VLM layers** — color lookup doesn't need an expressive backbone, and a smaller model is cheaper to run (and counts toward the smallest-model bonus).
- **Tasks 2 & 3** use **16 layers** — compositional and identity reasoning justify the extra capacity.

### 3. Teach language variety, not just behavior

The hardest part of Task 2 is linguistic, not motor: the same physical trajectory ("put the banana in *that* bowl") has to be reachable from many phrasings. So instead of collecting new demonstrations for every wording, we **relabel** existing episodes into new prompt families, keeping the trajectory fixed:

- *ordinal → color* — config names encode bowl order (`config1-red-blue-green`), so `2nd bowl from the left` becomes `blue colored bowl`.
- *color → negation* — `red colored bowl` becomes `the bowl that is not green and not blue`.
- *ordinal → relative* — generates adjacency references like `the bowl on the right of the red bowl`.

Each episode ends up associated with ~5 prompt formulations, exposing the policy to linguistic variety at near-zero collection cost.

### 4. Augment for the world, not the lab

Two visual augmentations push the policies toward robustness:

- **Brightness partitioning** splits a dataset into contiguous slices and applies a different brightness multiplier to each — *in place*, so no episodes are duplicated and only pixels change. This buys lighting robustness for free.
- **Celebrity face-swap** (Task 3) detects the portrait cards in each frame and composites *different* faces onto them, multiplying identity coverage so the policy can generalize to celebrities it never physically saw during collection.

### 5. Refine with DAgger, as a curriculum

A first policy trained on the broad dataset is rarely the final one. We roll it out, identify the failure cases, collect **corrective demonstrations** for exactly those situations, fold them back in, and continue training from the previous checkpoint (`--policy.pretrained_path`). Seen end to end this is a curriculum: train broad, then specialize on what's actually hard.

### 6. Clean data beats more data

Demonstrations are full of frames where the arm hasn't started moving yet. We trim those leading/trailing **frozen-action frames** from every episode before training, so the policy isn't taught to sit still. We also repair dataset metadata — recomputing normalization stats (`meta/stats.json`) and the episode `splits` — because SmolVLA silently trains unnormalized, or hides episodes from the dataloader, when those are wrong.

### The resulting policies

| Task | Policy | Training dataset | Notes |
|---|---|---|---|
| 1 | [`task1-dagger_038000`](https://huggingface.co/ETHrobotlearning/task1-dagger_038000) | [`tv-task1-clean-4prompts-fixed`](https://huggingface.co/datasets/ETHrobotlearning/tv-task1-clean-4prompts-fixed) | 8 layers · cleaned · 4 prompt variants · DAgger-refined |
| 2 | [`smolvla_task2_colors_dagger_lr2e-5-step2000`](https://huggingface.co/ETHrobotlearning/smolvla_task2_colors_dagger_lr2e-5-step2000) | [`tv-task2-clean-aug5p-fixed`](https://huggingface.co/datasets/ETHrobotlearning/tv-task2-clean-aug5p-fixed) | 16 layers · ~5 prompts/episode · brightness aug · DAgger-refined |
| 3 | [`Alessio03/smolvla-task3-50k`](https://huggingface.co/Alessio03/smolvla-task3-50k) | [`task3-TOY-clean`](https://huggingface.co/datasets/ETHrobotlearning/task3-TOY-clean) | 16 layers · celebrity face-swap aug · 50k steps |

---

## Did we even need the VLM to be smart?

A natural question underlies all of this: could a small VLM act as the perception backbone on its own — "find person X, move near them" — without us training an action expert at all? We built a **spatial-grounding benchmark** to find out. Four celebrity cards are arranged in a 2×2 grid; the model is asked which quadrant (`top_left`, `top_right`, `bottom_left`, `bottom_right`) contains a named person.

| Model | Accuracy (20 samples) | Observed behavior |
|---|---|---|
| `SmolVLM-256M-Instruct` | 25% | Almost always predicts `top_left`, regardless of input |
| `SmolVLM-500M-Instruct` | 25% | Almost always predicts `top_right`, regardless of input |

Both score at **chance** (25%) and show no celebrity understanding whatsoever — they ignore the identity prompt and fall back to a fixed positional bias. The takeaway shaped our whole approach: a small VLM is not a usable grounding oracle off the shelf, which is exactly why we lean on SmolVLA's representation through the *action expert* rather than asking the VLM to reason explicitly. Benchmark code lives in `benchmarks/celebrity_recognition/`.

---

## Infrastructure choices

A few decisions were about *running* the project sanely rather than the policies themselves:

- **Cost-aware cloud training.** `training/orchestrate.py` provisions a Brev GPU instance, drives training over SSH, and **deletes the instance on completion, failure, or Ctrl-C** — so a crashed run never quietly bills for hours. A background watcher streams each checkpoint to the Hub as it's written and frees the local copy.
- **Source-only repository.** Datasets, caches, model weights and training outputs are never tracked in git; everything is re-pullable from the Hub or regenerable by the scripts here.
- **MolmoAct2 exploration.** We also wired up fine-tuning of Ai2's MolmoAct2 (`training/*_molmoact2*`) as an alternative backbone for Task 3, kept here for reference.

---

## Running the evaluation

Each task ships one runnable rollout script. The scripts check that `lerobot-rollout` is available, install a local eval environment if needed, print the checkpoint and its parameter count, and run an SO-101 rollout with the requested prompt.

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# default prompt per task
./run_eval_1.sh
./run_eval_2.sh
./run_eval_3.sh

# custom prompt as the first argument
./run_eval_2.sh "Put the banana into the bowl that is not red and not blue."
./run_eval_3.sh "Place the coke on Barack Obama."
```

Checkpoints are loaded from the Hub by default (or from a local `policy_checkpoints/` cache built by `python scripts/download_eval_policies.py`):

| Eval | Hub model ID | Default prompt |
|---|---|---|
| Task 1 | `ETHrobotlearning/task1-dagger_038000` | `Place the banana to the red colored bowl.` |
| Task 2 | `ETHrobotlearning/smolvla_task2_colors_dagger_lr2e-5-step2000` | `Put the banana in the green colored bowl.` |
| Task 3 | `Alessio03/smolvla-task3-50k` | `Place the coke on Yann LeCun.` |

The scripts assume our SO-101 setup (`ROBOT_PORT`, `ROBOT_ID`, `POLICY_DEVICE=mps`, `FPS=10`, `DURATION=20`) and return the arm to its startup pose on teardown. Override any value with an environment variable, e.g. `export POLICY_DEVICE=cuda`.

---

## Repository map

```
run_eval_1.sh / _2 / _3.sh    Per-task SO-101 evaluation rollouts
run_augmentation.py           Text + visual augmentation entry point
run_benchmark.py              VLM grounding benchmark entry point
run_merge_augment_colours.sh  End-to-end relabel → merge → brightness for the 6 colour configs

augmentation/                 Text + visual augmentation pipeline (LeRobot format)
  celebrity_swap/             Portrait detection + face-swap identity augmentation
benchmarks/                   VLM spatial-grounding benchmark (Task 3 study)
collection/                   On-robot data collection (start-pose capture + recording)
datasets/utils/               Dataset transforms: relabel, merge, trim, downsample, re-encode, stats repair
inference/                    Remote policy servers (incl. MolmoAct2) and rollout helpers
scripts/                      Shared eval helpers (eval_common.sh, download_eval_policies.py)
training/                     SmolVLA fine-tuning, Brev orchestration, checkpoint→policy utilities
validation/ + lerobot-doctor/ Dataset validation and quality diagnostics
docs/                         Rollout GIFs and media
```

The dataset, augmentation and training scripts are self-documenting (`--help`); the principles above explain *why* each exists. For the full command reference on any tool, run it with `--help` from the repo root — all scripts accept either local paths or Hugging Face repo IDs.
