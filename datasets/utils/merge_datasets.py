#!/usr/bin/env python3
"""Merge multiple LeRobot v3 datasets into one.

All input datasets must pass the uniformity checks (same fps, robot_type,
features, video codec, task prompts).  The merge re-numbers episode and
frame indices globally; video files are copied as-is.

Input  — space-separated list of local paths OR HF repo IDs
Output — local path OR bare dataset name (pushed to your HF account)

Usage:
    # HF → HF
    python datasets/utils/merge_datasets.py \\
        --inputs alice/ds1 bob/ds2 carol/ds3 \\
        --output merged_dataset

    # local → local
    python datasets/utils/merge_datasets.py \\
        --inputs /data/ds1 /data/ds2 \\
        --output /data/merged
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------

def _require_hf() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("Error: huggingface_hub is not installed.  Run: pip install huggingface_hub")


def _hf_whoami() -> str:
    from huggingface_hub import whoami
    try:
        return whoami()["name"]
    except Exception as exc:
        sys.exit(f"Error: could not read HF identity ({exc}).  Run: huggingface-cli login")


def _is_hf_repo_id(s: str) -> bool:
    p = Path(s)
    if p.exists():
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts)


def _is_bare_name(s: str) -> bool:
    return "/" not in s and not Path(s).is_absolute() and not s.startswith(".")


def _download_hf_dataset(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {repo_id} …")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))


def _push_to_hub(local_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Creating / updating HF dataset repo: {repo_id} …")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading to {repo_id} …")
    api.upload_folder(folder_path=str(local_path), repo_id=repo_id, repo_type="dataset")
    print(f"Pushed → https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Meta loading
# ---------------------------------------------------------------------------

def _load_info(src: Path) -> Dict:
    with open(src / "meta" / "info.json") as f:
        return json.load(f)


def _load_episodes(src: Path) -> pd.DataFrame:
    ep_files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        sys.exit(f"Error: no episodes parquet files in {src}/meta/episodes/")
    return (pd.concat([pd.read_parquet(f) for f in ep_files])
              .sort_values("episode_index").reset_index(drop=True))


def _load_tasks(src: Path) -> List[str]:
    path = src / "meta" / "tasks.parquet"
    if not path.exists():
        sys.exit(f"Error: {path} not found.  Only LeRobot v3 datasets are supported.")
    df = pd.read_parquet(path)
    if pd.api.types.is_string_dtype(df.index):
        return list(df.index)
    if "task" in df.columns and pd.api.types.is_string_dtype(df["task"]):
        if "task_index" in df.columns:
            df = df.sort_values("task_index")
        return list(df["task"])
    df2 = df.reset_index()
    for col in df2.columns:
        if pd.api.types.is_string_dtype(df2[col]):
            return list(df2[col])
    return []


def _video_keys(info: Dict) -> List[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


# ---------------------------------------------------------------------------
# Uniformity check
# ---------------------------------------------------------------------------

def _feat_sig(feat: Dict) -> Dict:
    sig: Dict = {"dtype": feat.get("dtype"), "shape": feat.get("shape")}
    if feat.get("dtype") == "video":
        vi = feat.get("info") or feat.get("video_info", {})
        sig["codec"]   = vi.get("video.codec")
        sig["pix_fmt"] = vi.get("video.pix_fmt")
    return sig


def validate_uniform(sources: List[Path]) -> Dict:
    """Check all sources are compatible. Returns the reference info.json."""
    ref_info  = _load_info(sources[0])
    errors: List[str] = []

    SCALAR_KEYS = ("fps", "robot_type", "chunks_size")

    for src in sources[1:]:
        info  = _load_info(src)
        name  = src.name

        for k in SCALAR_KEYS:
            # treat 10 and 10.0 as equal
            rv, sv = ref_info.get(k), info.get(k)
            if isinstance(rv, float): rv = int(rv) if rv == int(rv) else rv
            if isinstance(sv, float): sv = int(sv) if sv == int(sv) else sv
            if rv != sv:
                errors.append(f"[{name}] {k}: {info.get(k)!r} ≠ {ref_info.get(k)!r}")

        ref_feats = ref_info.get("features", {})
        src_feats = info.get("features", {})
        if set(ref_feats) != set(src_feats):
            errors.append(f"[{name}] feature keys differ")
        else:
            for fk in ref_feats:
                rs, ss = _feat_sig(ref_feats[fk]), _feat_sig(src_feats[fk])
                if rs != ss:
                    errors.append(f"[{name}] feature '{fk}': {ss} ≠ {rs}")

    if errors:
        print("\nUniformity check FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    print(f"Uniformity check passed ({len(sources)} datasets).")
    return ref_info


def _build_global_tasks(sources: List[Path]) -> Tuple[List[str], List[Dict[int, int]]]:
    """
    Collect all unique task strings across sources (insertion order).
    Returns:
      - global_tasks: ordered list of all unique task strings
      - remaps: per-source dict mapping src task_index → global task_index
    """
    global_tasks: List[str] = []
    seen: Dict[str, int] = {}

    src_task_lists: List[List[str]] = []
    for src in sources:
        tasks = _load_tasks(src)
        src_task_lists.append(tasks)
        for t in tasks:
            if t not in seen:
                seen[t] = len(global_tasks)
                global_tasks.append(t)

    remaps: List[Dict[int, int]] = []
    for tasks in src_task_lists:
        remap = {i: seen[t] for i, t in enumerate(tasks)}
        remaps.append(remap)

    return global_tasks, remaps


# ---------------------------------------------------------------------------
# Episode-row index update helpers
# ---------------------------------------------------------------------------

def _shift_scalar_stat(row: Dict, col: str, delta: int | float) -> None:
    """Add delta to every value in a stats column (handles numpy scalar arrays)."""
    v = row.get(col)
    if v is None:
        return
    if isinstance(v, np.ndarray):
        row[col] = v + delta
    else:
        row[col] = v + delta


def _set_scalar_stat(row: Dict, col_prefix: str, value: float) -> None:
    """Overwrite all stat sub-keys for a constant feature (e.g. episode_index)."""
    for suffix in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        k = f"{col_prefix}/{suffix}"
        if k in row:
            row[k] = np.array([value])
    for suffix in ("std",):
        k = f"{col_prefix}/{suffix}"
        if k in row:
            row[k] = np.array([0.0])


def _update_episode_row(
    row: Dict,
    ep_offset: int,
    frame_offset: int,
    data_file_idx: int,
    vid_file_map: Dict[str, int],   # vk → new file index
    out_ep_file_idx: int,
) -> Dict:
    row = dict(row)

    new_ep_idx = int(row["episode_index"]) + ep_offset
    row["episode_index"]       = new_ep_idx
    row["dataset_from_index"]  = int(row["dataset_from_index"]) + frame_offset
    row["dataset_to_index"]    = int(row["dataset_to_index"])   + frame_offset

    row["data/chunk_index"] = 0
    row["data/file_index"]  = data_file_idx

    for vk, new_fi in vid_file_map.items():
        row[f"videos/{vk}/chunk_index"] = 0
        row[f"videos/{vk}/file_index"]  = new_fi

    row["meta/episodes/chunk_index"] = 0
    row["meta/episodes/file_index"]  = out_ep_file_idx

    # Update per-episode stats that change with global re-indexing
    _set_scalar_stat(row, "stats/episode_index", float(new_ep_idx))
    for suf in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        k = f"stats/index/{suf}"
        _shift_scalar_stat(row, k, frame_offset)

    return row


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _write_readme(dst: Path, repo_id: Optional[str] = None) -> None:
    """Write a minimal LeRobot dataset card.

    Only the YAML front-matter matters for the Hugging Face dataset page to embed
    the LeRobot visualizer: the ``LeRobot`` tag (capital L and R) is what makes HF
    recognise it as a LeRobot dataset, and the ``configs`` block points the viewer
    at the parquet files. This matches the card LeRobot itself generates in
    ``create_lerobot_dataset_card`` — anything else is intentionally left out.
    """
    frontmatter = (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "  - robotics\n"
        "tags:\n"
        "  - LeRobot\n"
        "configs:\n"
        "  - config_name: default\n"
        "    data_files: data/*/*.parquet\n"
        "---\n"
    )

    body = "\nThis dataset was created using [LeRobot](https://github.com/huggingface/lerobot).\n"
    if repo_id:
        body += (
            f'\n<a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path={repo_id}">\n'
            '  <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>\n'
            "</a>\n"
        )

    with open(dst / "README.md", "w") as f:
        f.write(frontmatter + body)


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------

def merge(sources: List[Path], dst: Path, repo_id: Optional[str] = None) -> None:
    ref_info = validate_uniform(sources)
    vkeys    = _video_keys(ref_info)
    global_tasks, task_remaps = _build_global_tasks(sources)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(exist_ok=True)

    ep_offset    = 0   # cumulative episode count
    frame_offset = 0   # cumulative frame count
    out_data_fi  = 0   # sequential data file index across all datasets
    out_vid_fi: Dict[str, int] = {vk: 0 for vk in vkeys}

    all_ep_rows: List[Dict] = []

    for src_i, src in enumerate(tqdm(sources, desc="merging", unit="dataset")):
        info       = _load_info(src)
        episodes   = _load_episodes(src)
        task_remap = task_remaps[src_i]   # src task_index → global task_index
        src_tasks  = _load_tasks(src)     # src task_index → task string

        # ---- data parquets ------------------------------------------------
        data_keys = (episodes[["data/chunk_index", "data/file_index"]]
                     .drop_duplicates()
                     .sort_values(["data/chunk_index", "data/file_index"]))

        # mapping: (src_ci, src_fi) -> output file index
        data_file_map: Dict[Tuple[int,int], int] = {}

        for _, (ci, fi) in data_keys.iterrows():
            ci, fi = int(ci), int(fi)
            src_pq = src  / f"data/chunk-{ci:03d}/file-{fi:03d}.parquet"
            dst_pq = dst  / f"data/chunk-000/file-{out_data_fi:03d}.parquet"
            dst_pq.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_parquet(src_pq)
            df["episode_index"] = df["episode_index"].astype(np.int64) + ep_offset
            df["index"]         = df["index"].astype(np.int64)         + frame_offset
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].map(task_remap).astype(np.int64)
            df.to_parquet(dst_pq, index=False)

            data_file_map[(ci, fi)] = out_data_fi
            out_data_fi += 1

        # ---- video files --------------------------------------------------
        vid_file_map: Dict[str, Dict[Tuple[int,int], int]] = {vk: {} for vk in vkeys}

        for vk in vkeys:
            ci_col = f"videos/{vk}/chunk_index"
            fi_col = f"videos/{vk}/file_index"
            vid_keys = (episodes[[ci_col, fi_col]]
                        .drop_duplicates()
                        .sort_values([ci_col, fi_col]))
            for _, (ci, fi) in vid_keys.iterrows():
                ci, fi = int(ci), int(fi)
                vsrc = src / f"videos/{vk}/chunk-{ci:03d}/file-{fi:03d}.mp4"
                vdst = dst / f"videos/{vk}/chunk-000/file-{out_vid_fi[vk]:03d}.mp4"
                vdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vsrc, vdst)
                vid_file_map[vk][(ci, fi)] = out_vid_fi[vk]
                out_vid_fi[vk] += 1

        # ---- episode metadata rows ----------------------------------------
        for _, ep_row in episodes.iterrows():
            src_d_ci = int(ep_row["data/chunk_index"])
            src_d_fi = int(ep_row["data/file_index"])
            vk_new_fi = {
                vk: vid_file_map[vk][(int(ep_row[f"videos/{vk}/chunk_index"]),
                                      int(ep_row[f"videos/{vk}/file_index"]))]
                for vk in vkeys
            }
            new_row = _update_episode_row(
                ep_row.to_dict(),
                ep_offset    = ep_offset,
                frame_offset = frame_offset,
                data_file_idx = data_file_map[(src_d_ci, src_d_fi)],
                vid_file_map  = vk_new_fi,
                out_ep_file_idx = 0,
            )
            # remap tasks column (array of task strings per episode)
            if "tasks" in new_row:
                raw = new_row["tasks"]
                if isinstance(raw, np.ndarray):
                    new_row["tasks"] = np.array(
                        [global_tasks[task_remap[src_tasks.index(t)]]
                         if t in src_tasks else t for t in raw],
                        dtype=raw.dtype,
                    )
            all_ep_rows.append(new_row)

        ep_offset    += info["total_episodes"]
        frame_offset += info["total_frames"]

    # ---- write merged episodes parquet ------------------------------------
    ep_df = (pd.DataFrame(all_ep_rows)
               .sort_values("episode_index")
               .reset_index(drop=True))

    # Normalize stats columns so pyarrow sees a uniform type per column.
    # Mixed numpy arrays and scalars (or arrays of different dtypes) cause ArrowInvalid.
    def _normalize_stat_val(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list):
            return v
        if v is None:
            return None
        try:
            if np.isnan(float(v)):
                return None
        except (TypeError, ValueError):
            pass
        return [float(v)]

    for col in ep_df.columns:
        if not col.startswith("stats/"):
            continue
        col_vals = ep_df[col]
        # Check if any value is array-like
        has_array = col_vals.apply(
            lambda v: isinstance(v, (np.ndarray, list))
        ).any()
        if has_array:
            ep_df[col] = col_vals.apply(_normalize_stat_val)
        else:
            # All scalars — convert to float for consistency
            ep_df[col] = pd.to_numeric(col_vals, errors="coerce")

    out_ep = dst / "meta/episodes/chunk-000/file-000.parquet"
    out_ep.parent.mkdir(parents=True, exist_ok=True)
    # Write via pyarrow directly so we infer a fresh schema without stale pandas
    # metadata (which can encode fixed_size_list types that conflict with our lists).
    import pyarrow as pa
    import pyarrow.parquet as pq
    arrays, fields = [], []
    for col in ep_df.columns:
        arr = pa.array(ep_df[col].tolist())
        arrays.append(arr)
        fields.append(pa.field(col, arr.type))
    table = pa.table(dict(zip(ep_df.columns, arrays)), schema=pa.schema(fields))
    pq.write_table(table, out_ep)

    # ---- merged tasks.parquet ---------------------------------------------
    tasks_df = pd.DataFrame({
        "task_index": list(range(len(global_tasks))),
        "task": global_tasks,
    })
    tasks_df.to_parquet(dst / "meta" / "tasks.parquet", index=False)

    # ---- info.json --------------------------------------------------------
    new_info = dict(ref_info)
    new_info["total_episodes"] = ep_offset
    new_info["total_frames"]   = frame_offset
    new_info["total_tasks"]    = len(global_tasks)
    new_info["total_videos"]   = sum(out_vid_fi.values())
    new_info["total_chunks"]   = 1
    new_info["splits"]         = {"train": f"0:{ep_offset}"}
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    # ---- README.md --------------------------------------------------------
    _write_readme(dst, repo_id)

    print(
        f"\nMerged {len(sources)} datasets → "
        f"{ep_offset} episodes, {frame_offset} frames."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge multiple LeRobot v3 datasets into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--inputs",  required=True, nargs="+", metavar="SRC",
                    help="Source datasets: local paths or HF repo IDs.")
    ap.add_argument("--output",  required=True, metavar="DST",
                    help="Output: local path, or bare name to push to your HF account.")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("Provide at least two --inputs datasets.")

    any_hf_input  = any(_is_hf_repo_id(s) for s in args.inputs)
    output_is_hf  = _is_bare_name(args.output) or _is_hf_repo_id(args.output)

    if any_hf_input or output_is_hf:
        _require_hf()

    hf_output_repo: Optional[str] = None
    if output_is_hf:
        hf_output_repo = (args.output if _is_hf_repo_id(args.output)
                          else f"{_hf_whoami()}/{args.output}")

    # Resolve all inputs to local directories
    tmp_dirs: List[tempfile.TemporaryDirectory] = []
    local_sources: List[Path] = []

    print("Loading source datasets …")
    for raw in args.inputs:
        if _is_hf_repo_id(raw):
            td = tempfile.TemporaryDirectory(prefix="lerobot_merge_src_")
            tmp_dirs.append(td)
            src = Path(td.name)
            _download_hf_dataset(raw, src)
        else:
            src = Path(raw)
            if not src.is_dir():
                sys.exit(f"Error: source path does not exist: {src}")
        local_sources.append(src)

    tmp_output_dir: Optional[tempfile.TemporaryDirectory] = None
    if output_is_hf:
        tmp_output_dir = tempfile.TemporaryDirectory(prefix="lerobot_merge_dst_")
        dst = Path(tmp_output_dir.name) / "dataset"
    else:
        dst = Path(args.output)
        if dst.exists():
            sys.exit(f"Error: destination already exists: {dst}")

    try:
        merge(local_sources, dst, hf_output_repo)
        if output_is_hf:
            _push_to_hub(dst, hf_output_repo)
        else:
            print(f"Done  →  {dst}")
    finally:
        for td in tmp_dirs:
            td.cleanup()
        if tmp_output_dir is not None:
            tmp_output_dir.cleanup()


if __name__ == "__main__":
    main()
