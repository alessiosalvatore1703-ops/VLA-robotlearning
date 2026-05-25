"""
Modify prompts in ETHrobotlearning/tv-task1-final-clean
to use 4 rotating prompt variants per color (green/red/blue).
Pushes the modified dataset back to HuggingFace.
"""

import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# ── 4 prompt variants per color ──────────────────────────────────────────────
PROMPTS = {
    "green": [
        "Put the banana in the green colored bowl.",
        "Place the banana in the green bowl.",
        "Move the banana to the green colored bowl.",
        "Put the banana in the green bowl.",
    ],
    "red": [
        "Put the banana in the red colored bowl.",
        "Place the banana in the red bowl.",
        "Move the banana to the red colored bowl.",
        "Put the banana in the red bowl.",
    ],
    "blue": [
        "Put the banana in the blue colored bowl.",
        "Place the banana in the blue bowl.",
        "Move the banana to the blue colored bowl.",
        "Put the banana in the blue bowl.",
    ],
}

REPO_ID     = "ETHrobotlearning/tv-task1-clean"
NEW_REPO_ID = "ETHrobotlearning/tv-task1-clean-4prompts"

# ── detect color using word boundaries (avoids "colored" matching "red") ──────
def detect_color(prompt: str) -> str | None:
    p = prompt.lower()
    for color in ["green", "red", "blue"]:
        if re.search(rf"\b{color}\b", p):
            return color
    return None

# ── find all episodes parquet files in the repo ───────────────────────────────
api = HfApi()
all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
ep_files = sorted([f for f in all_files if f.startswith("meta/episodes")])
print(f"Found episodes files: {ep_files}")

# ── process each episodes parquet file ───────────────────────────────────────
color_counters = {"green": 0, "red": 0, "blue": 0}
modified_files = {}  # path -> modified parquet bytes

for ep_file in ep_files:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ep_file,
        repo_type="dataset"
    )

    table = pq.read_table(local_path)
    df = table.to_pandas()

    print(f"\n{ep_file}: {len(df)} episodes")
    print(f"  Sample tasks: {df['tasks'].head(3).tolist()}")

    new_tasks = []
    for i, row in df.iterrows():
        current_prompt = row["tasks"][0] if isinstance(row["tasks"], list) else row["tasks"]
        color = detect_color(str(current_prompt))

        if color is None:
            print(f"  WARNING: ep {i} — could not detect color: '{current_prompt}', keeping original.")
            new_tasks.append(row["tasks"])
            continue

        idx = color_counters[color] % 4
        new_prompt = PROMPTS[color][idx]
        color_counters[color] += 1
        new_tasks.append([new_prompt])

    df["tasks"] = new_tasks

    # write back preserving original schema
    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    buf = pa.BufferOutputStream()
    pq.write_table(new_table, buf)
    modified_files[ep_file] = buf.getvalue().to_pybytes()

# ── summary ───────────────────────────────────────────────────────────────────
print("\n── Assignment summary ──")
for color in ["green", "red", "blue"]:
    for i, p in enumerate(PROMPTS[color]):
        assigned = sum(
            1 for t in [row[0] if isinstance(row, (list, tuple)) else row
                        for ep_bytes in modified_files.values()
                        for row in pq.read_table(pa.BufferReader(ep_bytes))
                               .to_pandas()["tasks"].tolist()]
            if t == p
        )
        print(f"  [{color}] variant {i+1}: {assigned} episodes — '{p}'")
print(f"\n  green: {color_counters['green']} episodes")
print(f"  red:   {color_counters['red']} episodes")
print(f"  blue:  {color_counters['blue']} episodes")

# ── copy full repo to new repo, then upload modified episodes files ───────────
print(f"\nCloning {REPO_ID} -> {NEW_REPO_ID} ...")
api.create_repo(NEW_REPO_ID, repo_type="dataset", exist_ok=True)

ep_files_set = set(ep_files)
other_files = [f for f in all_files if f not in ep_files_set]

for f in other_files:
    local = hf_hub_download(REPO_ID, filename=f, repo_type="dataset")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=f,
        repo_id=NEW_REPO_ID,
        repo_type="dataset",
    )
    print(f"  copied: {f}")

for ep_file, data in modified_files.items():
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo=ep_file,
        repo_id=NEW_REPO_ID,
        repo_type="dataset",
    )
    print(f"  uploaded modified: {ep_file}")

print("\nDone!")