"""
LeRobot Prompt Augmentation — 4 prompts per episode, rotating variants
6 config dataset
"""

from huggingface_hub import snapshot_download, HfApi
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import glob, os, json, shutil
import numpy as np
from collections import defaultdict

# ─── ALL 33 PROMPTS ──────────────────────────────────────────────────────────
ALL_PROMPTS = {
    0:  "Move the banana to the red colored bowl.",
    1:  "Put the banana into the left bowl from the robot perspective.",
    2:  "Put the banana into the bowl on the left of the blue bowl from the robot perspective.",
    3:  "Put the banana into the bowl that is not green and not blue.",
    4:  "Put the banana in the green bowl.",
    5:  "Put the banana into the 1st bowl from the left from the robot perspective.",
    6:  "Put the banana into the bowl on the left of the red bowl from the robot perspective.",
    7:  "Put the banana into the bowl that is not red and not blue.",
    8:  "Put the banana in the blue colored bowl.",
    9:  "Put the banana into the bowl that is not red and not green.",
    10: "Put the banana in the red colored bowl.",
    11: "Place the banana in the green bowl.",
    12: "Put the banana into the leftmost bowl from the robot perspective.",
    13: "Put the banana in the blue bowl.",
    14: "Put the banana in the red bowl.",
    15: "Move the banana to the green colored bowl.",
    16: "Place the banana in the blue bowl.",
    17: "Place the banana in the red bowl.",
    18: "Put the banana in the green colored bowl.",
    19: "Move the banana to the blue colored bowl.",
    20: "Put the banana into the center bowl from the robot perspective.",
    21: "Put the banana into the bowl on the left of the green bowl from the robot perspective.",
    22: "Put the banana into the 2nd bowl from the left from the robot perspective.",
    23: "Put the banana into the bowl between the blue bowl and the red bowl.",
    24: "Put the banana into the bowl between the red bowl and the green bowl.",
    25: "Put the banana into the bowl between the blue bowl and the green bowl.",
    26: "Put the banana into the middle bowl from the robot perspective.",
    27: "Put the banana into the bowl on the right of the blue bowl from the robot perspective.",
    28: "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
    29: "Put the banana into the right bowl from the robot perspective.",
    30: "Put the banana into the bowl on the right of the green bowl from the robot perspective.",
    31: "Put the banana into the 3rd bowl from the left from the robot perspective.",
    32: "Put the banana into the rightmost bowl from the robot perspective.",
}

# ─── PROMPT CATEGORIES ───────────────────────────────────────────────────────
COLOR_VARIANTS = {
    "red":   [0, 10, 14, 17],
    "green": [4, 11, 15, 18],
    "blue":  [8, 13, 16, 19],
}

POSITION_VARIANTS = {
    "left":   [1, 5, 12],
    "middle": [20, 22, 26],
    "right":  [29, 31, 32],
}

EXCLUSION_MAP = {
    "red":   3,
    "green": 7,
    "blue":  9,
}

BETWEEN_MAP = {
    ("blue",  "red"):   23,
    ("red",   "blue"):  23,
    ("red",   "green"): 24,
    ("green", "red"):   24,
    ("blue",  "green"): 25,
    ("green", "blue"):  25,
}

RELATIVE_RIGHT_OF = {
    "red":   28,
    "green": 30,
    "blue":  27,
}

RELATIVE_LEFT_OF = {
    "red":   6,
    "green": 21,
    "blue":  2,
}

# ─── 6 CONFIGS ───────────────────────────────────────────────────────────────
CONFIGS = {
    "ETHrobotlearning/tv-config1-green-red-blue-clean": ["green", "red",   "blue"],
    "ETHrobotlearning/tv-config2-red-green-blue-clean": ["red",   "green", "blue"],
    "ETHrobotlearning/tv-config3-red-blue-green-clean": ["red",   "blue",  "green"],
    "ETHrobotlearning/tv-config4-blue-red-green-clean": ["blue",  "red",   "green"],
    "ETHrobotlearning/tv-config5-green-blue-red-clean": ["green", "blue",  "red"],
    "ETHrobotlearning/tv-config6-blue-green-red-clean": ["blue",  "green", "red"],
}

COLOR_PROMPTS = {
    "green": "Put the banana in the green colored bowl.",
    "red":   "Put the banana in the red colored bowl.",
    "blue":  "Put the banana in the blue colored bowl.",
}

# ─── CORE PROMPT LOGIC ───────────────────────────────────────────────────────
def get_5_prompts_for_episode(position, color, left_color, right_color, episode_num):
    color_prompt_id     = COLOR_VARIANTS[color][episode_num % 4]
    position_prompt_id  = POSITION_VARIANTS[position][episode_num % 3]
    exclusion_prompt_id = EXCLUSION_MAP[color]

    if left_color and right_color:
        # Middle bowl:
        #   prompt 4 = always "between X and Y"
        #   prompt 5 = alternates between right-of-left-neighbor and left-of-right-neighbor
        between_id   = BETWEEN_MAP.get((left_color, right_color),
                       BETWEEN_MAP.get((right_color, left_color)))
        # cycle: even episodes → right-of-left, odd episodes → left-of-right
        directional_id = (RELATIVE_RIGHT_OF[left_color]
                          if episode_num % 2 == 0
                          else RELATIVE_LEFT_OF[right_color])
        return [
            ALL_PROMPTS[color_prompt_id],
            ALL_PROMPTS[position_prompt_id],
            ALL_PROMPTS[exclusion_prompt_id],
            ALL_PROMPTS[between_id],
            ALL_PROMPTS[directional_id],
        ]
    elif left_color:
        # Right bowl: single relative + second color variant
        relative_id     = RELATIVE_RIGHT_OF[left_color]
        color_id_2      = COLOR_VARIANTS[color][(episode_num + 2) % 4]
    else:
        # Left bowl: single relative + second color variant
        relative_id     = RELATIVE_LEFT_OF[right_color]
        color_id_2      = COLOR_VARIANTS[color][(episode_num + 2) % 4]

    return [
        ALL_PROMPTS[color_prompt_id],
        ALL_PROMPTS[position_prompt_id],
        ALL_PROMPTS[exclusion_prompt_id],
        ALL_PROMPTS[relative_id],
        ALL_PROMPTS[color_id_2],
    ]

# ─── PREVIEW ─────────────────────────────────────────────────────────────────
def preview_all_configs():
    for repo_id, layout in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"CONFIG: {repo_id}")
        print(f"Layout: [{layout[0]}] [{layout[1]}] [{layout[2]}]")
        print(f"{'='*70}")
        for pos_idx, position in enumerate(["left", "middle", "right"]):
            color       = layout[pos_idx]
            left_color  = layout[pos_idx - 1] if pos_idx > 0 else None
            right_color = layout[pos_idx + 1] if pos_idx < 2 else None
            print(f"\n  {position.upper()} bowl ({color}) — first 3 episodes:")
            for ep_num in range(3):
                prompts = get_5_prompts_for_episode(
                    position, color, left_color, right_color, ep_num
                )
                for p in prompts:
                    print(f"    - {p}")

# ─── AUGMENTATION ────────────────────────────────────────────────────────────
def augment_dataset(repo_id, layout):
    print(f"\n{'='*60}\nProcessing: {repo_id}\n{'='*60}")

    local_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    work_path  = os.path.join(
        os.path.expanduser("~"), "lerobot_aug", repo_id.split("/")[1]
    )
    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    shutil.copytree(local_path, work_path)

    # ── Read original tasks ──────────────────────────────────────────────────
    orig_tasks_df = pd.read_parquet(
        os.path.join(work_path, "meta", "tasks.parquet")
    ).reset_index()
    print(f"Original tasks:\n{orig_tasks_df.to_string()}")

    old_idx_to_bowl = {}
    for _, row in orig_tasks_df.iterrows():
        task_str = row["task"]
        for color, prompt in COLOR_PROMPTS.items():
            if task_str == prompt:
                pos_idx  = layout.index(color)
                position = ["left", "middle", "right"][pos_idx]
                old_idx_to_bowl[int(row["task_index"])] = position

    print(f"task_index -> bowl: {old_idx_to_bowl}")

    # ── Load data ────────────────────────────────────────────────────────────
    data_files = sorted(glob.glob(
        os.path.join(work_path, "data", "**", "*.parquet"), recursive=True
    ))
    orig_data = pd.concat(
        [pd.read_parquet(p) for p in data_files], ignore_index=True
    ).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # ── Map episode -> bowl ──────────────────────────────────────────────────
    ep_to_bowl = {}
    for ep in orig_data["episode_index"].unique():
        orig_task_idx = int(
            orig_data[orig_data["episode_index"] == ep]["task_index"].iloc[0]
        )
        bowl = old_idx_to_bowl.get(orig_task_idx)
        if bowl:
            ep_to_bowl[ep] = bowl
        else:
            print(f"  WARNING: episode {ep} task_index {orig_task_idx} unknown, skipping")

    bowl_counts = defaultdict(int)
    for bowl in ep_to_bowl.values():
        bowl_counts[bowl] += 1
    print(f"Episodes per bowl: {dict(bowl_counts)}")

    # ── Collect all unique prompts ───────────────────────────────────────────
    all_new_prompts_list = []
    seen = set()
    for pos_idx, position in enumerate(["left", "middle", "right"]):
        color       = layout[pos_idx]
        left_color  = layout[pos_idx - 1] if pos_idx > 0 else None
        right_color = layout[pos_idx + 1] if pos_idx < 2 else None
        for ep_num in range(bowl_counts[position]):
            for p in get_5_prompts_for_episode(
                position, color, left_color, right_color, ep_num
            ):
                if p not in seen:
                    all_new_prompts_list.append(p)
                    seen.add(p)

    task_to_idx = {t: i for i, t in enumerate(all_new_prompts_list)}
    n_prompts   = len(all_new_prompts_list)
    print(f"Total unique prompts: {n_prompts}")

    # ── Write new tasks.parquet ──────────────────────────────────────────────
    tasks_df = pd.DataFrame([
        {"task_index": i, "task": t}
        for i, t in enumerate(all_new_prompts_list)
    ]).set_index("task")
    tasks_df.to_parquet(os.path.join(work_path, "meta", "tasks.parquet"))

    # ── Load episodes metadata ───────────────────────────────────────────────
    ep_files = sorted(glob.glob(
        os.path.join(work_path, "meta", "episodes", "**", "*.parquet"),
        recursive=True
    ))

    orig_ep_df = pd.concat(
        [pd.read_parquet(p) for p in ep_files], ignore_index=True
    )
    orig_ep_lookup = {
        int(row["episode_index"]): row.to_dict()
        for _, row in orig_ep_df.iterrows()
    }

    # ── Build augmented data ─────────────────────────────────────────────────
    new_rows            = []
    ep_meta             = []
    new_ep_idx          = 0
    global_frame_cursor = 0
    ep_count_per_bowl   = defaultdict(int)

    for orig_ep in sorted(ep_to_bowl.keys()):
        bowl        = ep_to_bowl[orig_ep]
        ep_num      = ep_count_per_bowl[bowl]
        pos_idx     = ["left", "middle", "right"].index(bowl)
        color       = layout[pos_idx]
        left_color  = layout[pos_idx - 1] if pos_idx > 0 else None
        right_color = layout[pos_idx + 1] if pos_idx < 2 else None

        prompts      = get_5_prompts_for_episode(bowl, color, left_color, right_color, ep_num)
        ep_rows      = orig_data[orig_data["episode_index"] == orig_ep].copy()
        n_frames     = len(ep_rows)
        orig_ep_meta = orig_ep_lookup.get(orig_ep, {})

        for prompt_text in prompts:
            new_task_idx = task_to_idx[prompt_text]
            frame_start  = global_frame_cursor
            frame_end    = global_frame_cursor + n_frames

            # ── Duplicate data rows ──────────────────────────────────────────
            chunk = ep_rows.copy()
            chunk["episode_index"] = new_ep_idx
            chunk["task_index"]    = new_task_idx
            chunk["index"]         = range(frame_start, frame_end)
            new_rows.append(chunk)

            # ── Duplicate episode metadata ───────────────────────────────────
            new_ep_row = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in orig_ep_meta.items()
            }
            new_ep_row["episode_index"]      = new_ep_idx
            new_ep_row["task_index"]         = new_task_idx
            new_ep_row["length"]             = n_frames
            new_ep_row["dataset_from_index"] = frame_start
            new_ep_row["dataset_to_index"]   = frame_end
            new_ep_row["tasks"]              = [prompt_text]

            # ── Recalculate index stats ──────────────────────────────────────
            idx_range = np.arange(frame_start, frame_end, dtype=float)
            for stat, val in [
                ("stats/index/min",   [float(frame_start)]),
                ("stats/index/max",   [float(frame_end - 1)]),
                ("stats/index/mean",  [float(idx_range.mean())]),
                ("stats/index/std",   [float(idx_range.std())]),
                ("stats/index/count", [n_frames]),
                ("stats/index/q01",   [float(np.percentile(idx_range,  1))]),
                ("stats/index/q10",   [float(np.percentile(idx_range, 10))]),
                ("stats/index/q50",   [float(np.percentile(idx_range, 50))]),
                ("stats/index/q90",   [float(np.percentile(idx_range, 90))]),
                ("stats/index/q99",   [float(np.percentile(idx_range, 99))]),
            ]:
                if stat in new_ep_row:
                    new_ep_row[stat] = val

            # ── Recalculate episode_index stats (constant per episode) ───────
            for stat, val in [
                ("stats/episode_index/min",   [float(new_ep_idx)]),
                ("stats/episode_index/max",   [float(new_ep_idx)]),
                ("stats/episode_index/mean",  [float(new_ep_idx)]),
                ("stats/episode_index/std",   [0.0]),
                ("stats/episode_index/count", [n_frames]),
                ("stats/episode_index/q01",   [float(new_ep_idx)]),
                ("stats/episode_index/q10",   [float(new_ep_idx)]),
                ("stats/episode_index/q50",   [float(new_ep_idx)]),
                ("stats/episode_index/q90",   [float(new_ep_idx)]),
                ("stats/episode_index/q99",   [float(new_ep_idx)]),
            ]:
                if stat in new_ep_row:
                    new_ep_row[stat] = val

            # ── Recalculate task_index stats (constant per episode) ──────────
            for stat, val in [
                ("stats/task_index/min",   [float(new_task_idx)]),
                ("stats/task_index/max",   [float(new_task_idx)]),
                ("stats/task_index/mean",  [float(new_task_idx)]),
                ("stats/task_index/std",   [0.0]),
                ("stats/task_index/count", [n_frames]),
                ("stats/task_index/q01",   [float(new_task_idx)]),
                ("stats/task_index/q10",   [float(new_task_idx)]),
                ("stats/task_index/q50",   [float(new_task_idx)]),
                ("stats/task_index/q90",   [float(new_task_idx)]),
                ("stats/task_index/q99",   [float(new_task_idx)]),
            ]:
                if stat in new_ep_row:
                    new_ep_row[stat] = val

            ep_meta.append(new_ep_row)
            new_ep_idx          += 1
            global_frame_cursor += n_frames

        ep_count_per_bowl[bowl] += 1

    aug_data       = pd.concat(new_rows, ignore_index=True)
    total_episodes = new_ep_idx
    total_frames   = len(aug_data)
    print(f"\nAugmented: {total_episodes} episodes, {total_frames} frames")

    # ── Build ep_meta_df ─────────────────────────────────────────────────────
    ep_meta_df = pd.DataFrame(ep_meta)
    ep_meta_df["tasks"] = ep_meta_df["tasks"].apply(
        lambda x: x if isinstance(x, list) else [str(x)]
    )

    # ── Write data parquet files ─────────────────────────────────────────────
    for p in data_files:
        os.remove(p)
    data_dir = os.path.dirname(data_files[0])
    os.makedirs(data_dir, exist_ok=True)
    n_out  = max(1, total_frames // 5000 + 1)
    chunks = list(np.array_split(aug_data, n_out))
    for i, fc in enumerate(chunks):
        fc.to_parquet(os.path.join(data_dir, f"file-{i:03d}.parquet"), index=False)
        print(f"  wrote data file-{i:03d}.parquet ({len(fc)} rows)")

    ep_to_data_file = {}
    for file_i, fc in enumerate(chunks):
        for ep in fc["episode_index"].unique():
            ep_to_data_file[int(ep)] = file_i

    if "data/chunk_index" in ep_meta_df.columns:
        ep_meta_df["data/chunk_index"] = 0
    if "data/file_index" in ep_meta_df.columns:
        ep_meta_df["data/file_index"] = ep_meta_df["episode_index"].map(ep_to_data_file)

    # ── Write episodes metadata (single file, pyarrow schema preserved) ──────
    for p in ep_files:
        os.remove(p)
    ep_dir = os.path.dirname(ep_files[0])
    os.makedirs(ep_dir, exist_ok=True)

    if "meta/episodes/chunk_index" in ep_meta_df.columns:
        ep_meta_df["meta/episodes/chunk_index"] = 0
        ep_meta_df["meta/episodes/file_index"]  = 0

    # Drop per-episode stats columns — they have mixed numpy/list types that trip pyarrow,
    # and are only used by the HF dataset viewer, not by training.
    stats_cols = [c for c in ep_meta_df.columns if c.startswith("stats/")]
    ep_write = ep_meta_df.drop(columns=stats_cols)

    # Ensure tasks column is a Python list of strings
    ep_write["tasks"] = ep_write["tasks"].apply(
        lambda x: x if isinstance(x, list) else [str(x)]
    )

    ep_write.to_parquet(os.path.join(ep_dir, "file-000.parquet"), index=False)
    print(f"  wrote episodes file-000.parquet ({len(ep_meta_df)} rows)")

    # ── Update stats.json ────────────────────────────────────────────────────
    stats_path = os.path.join(work_path, "meta", "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

        # Recalculate task_index stats from augmented data
        arr = aug_data["task_index"].values.astype(float)
        stats["task_index"] = {
            "min":   [float(arr.min())],
            "max":   [float(arr.max())],
            "mean":  [float(arr.mean())],
            "std":   [float(arr.std())],
            "count": [int(len(arr))],
            "q01":   [float(np.percentile(arr,  1))],
            "q10":   [float(np.percentile(arr, 10))],
            "q50":   [float(np.percentile(arr, 50))],
            "q90":   [float(np.percentile(arr, 90))],
            "q99":   [float(np.percentile(arr, 99))],
        }

        # Fix count for all other keys (data duplicated 4x)
        for key in stats:
            if key == "task_index":
                continue
            if "count" in stats[key]:
                stats[key]["count"] = [int(stats[key]["count"][0]) * 5]

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print("  updated stats.json")
    else:
        print("  stats.json not found, skipping")

    # ── Update info.json ─────────────────────────────────────────────────────
    info_path = os.path.join(work_path, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    info["total_tasks"]    = n_prompts
    info["total_episodes"] = total_episodes
    info["total_frames"]   = total_frames
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  updated info.json: {total_episodes} eps, {total_frames} frames, {n_prompts} tasks")

    # ── Upload ───────────────────────────────────────────────────────────────
    aug_repo_id = repo_id + "-aug5p"
    api = HfApi()
    api.create_repo(repo_id=aug_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=work_path, repo_id=aug_repo_id, repo_type="dataset"
    )
    print(f"Uploaded → https://huggingface.co/datasets/{aug_repo_id}")
    return aug_repo_id


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("PREVIEWING PROMPT ASSIGNMENTS...")
    preview_all_configs()

    confirm = input("\nLooks good? Type 'yes' to run augmentation: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        exit()

    aug_repos = []
    for repo_id, layout in CONFIGS.items():
        aug_repo = augment_dataset(repo_id, layout)
        aug_repos.append(aug_repo)

    print(f"\n{'='*60}")
    print(f"All done! Uploaded repos:")
    for r in aug_repos:
        print(f"  https://huggingface.co/datasets/{r}")