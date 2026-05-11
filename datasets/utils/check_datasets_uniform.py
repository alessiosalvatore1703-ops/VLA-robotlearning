#!/usr/bin/env python3
"""Check that a set of LeRobot datasets share the same key parameters.

Loads meta/info.json from each dataset and compares:
  - fps, robot_type, chunks_size
  - feature keys, dtypes, shapes
  - video codec, pixel format, and in-video fps for each camera stream

Usage:
    # Local paths
    python datasets/utils/check_datasets_uniform.py /path/to/ds1 /path/to/ds2

    # Hugging Face repo IDs
    python datasets/utils/check_datasets_uniform.py user/ds1 user/ds2

    # Mixed
    python datasets/utils/check_datasets_uniform.py /path/to/ds1 user/ds2

Exits 0 if all checked fields match, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _tasks_from_parquet(path: str) -> List[Dict]:
    import pandas.api.types as ptypes
    df = pd.read_parquet(path)
    # v3: task strings stored as the DataFrame index
    if ptypes.is_string_dtype(df.index):
        return [{"task": str(s)} for s in df.index]
    # fallback: find a string column
    if "task" in df.columns and ptypes.is_string_dtype(df["task"]):
        return [{"task": str(s)} for s in df["task"]]
    df2 = df.reset_index()
    for col in df2.columns:
        if ptypes.is_string_dtype(df2[col]):
            return [{"task": str(s)} for s in df2[col]]
    return []


def _load_local(path: Path) -> Dict:
    meta_dir = path / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    tasks: List[Dict] = []
    if (meta_dir / "tasks.jsonl").exists():
        with open(meta_dir / "tasks.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    elif (meta_dir / "tasks.parquet").exists():
        tasks = _tasks_from_parquet(str(meta_dir / "tasks.parquet"))
    return {"info": info, "tasks": tasks}


def _load_hf(repo_id: str) -> Dict:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        print(
            "Error: huggingface_hub is not installed. "
            "Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    info_path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        info = json.load(f)

    tasks: List[Dict] = []
    try:
        tasks_path = hf_hub_download(repo_id=repo_id, filename="meta/tasks.jsonl", repo_type="dataset")
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    except EntryNotFoundError:
        tasks_path = hf_hub_download(repo_id=repo_id, filename="meta/tasks.parquet", repo_type="dataset")
        tasks = _tasks_from_parquet(tasks_path)

    return {"info": info, "tasks": tasks}


def load_meta(dataset: str) -> Dict:
    p = Path(dataset)
    if p.exists() and p.is_dir():
        return _load_local(p)
    # Treat as HF repo ID
    return _load_hf(dataset)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _short_name(dataset: str) -> str:
    p = Path(dataset)
    return p.name if p.exists() else dataset


def extract_fields(meta: Dict) -> Dict[str, Any]:
    info = meta["info"]
    fields: Dict[str, Any] = {
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "chunks_size": info.get("chunks_size"),
        "feature_keys": sorted(info.get("features", {}).keys()),
    }

    for key, feat in sorted(info.get("features", {}).items()):
        fields[f"feature:{key}:dtype"] = feat.get("dtype")
        fields[f"feature:{key}:shape"] = feat.get("shape")
        if feat.get("dtype") == "video":
            vi = feat.get("video_info") or feat.get("info", {})
            fields[f"feature:{key}:video.fps"] = vi.get("video.fps")
            fields[f"feature:{key}:video.codec"] = vi.get("video.codec")
            fields[f"feature:{key}:video.pix_fmt"] = vi.get("video.pix_fmt")

    return fields


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_RED   = "\033[31m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"

def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text


def _normalize(v: Any) -> Any:
    """Coerce whole-number floats to int so 10.0 and 10 compare equal."""
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize(val) for k, val in v.items()}
    return v


def _serialize(v: Any) -> str:
    return json.dumps(_normalize(v), sort_keys=True)


def compare_and_report(datasets: List[str], metas: List[Dict]) -> bool:
    all_fields = [extract_fields(m) for m in metas]

    all_keys: List[str] = sorted({k for f in all_fields for k in f})

    matches: List[Tuple[str, Any]] = []
    mismatches: List[Tuple[str, List[Any]]] = []

    for key in all_keys:
        values = [f.get(key) for f in all_fields]
        if len({_serialize(v) for v in values}) == 1:
            matches.append((key, values[0]))
        else:
            mismatches.append((key, values))

    # Header
    print()
    print(_color(f"Comparing {len(datasets)} datasets:", _BOLD))
    for i, d in enumerate(datasets):
        print(f"  [{i}] {d}")

    # Matching fields
    print()
    print(_color(f"MATCH ({len(matches)} field(s)):", _BOLD))
    for key, val in matches:
        if key == "feature_keys":
            print(_color(f"  {key}", _GREEN) + f": {val}")
        else:
            print(_color(f"  {key}", _GREEN) + f": {val!r}")

    # Mismatched fields
    if mismatches:
        print()
        print(_color(f"MISMATCH ({len(mismatches)} field(s)):", _BOLD))
        for key, values in mismatches:
            print()
            print(_color(f"  {key}:", _RED))
            for i, (d, v) in enumerate(zip(datasets, values)):
                label = f"[{i}] {_short_name(d)}"
                print(f"    {label}: {v!r}")
        print()
        print(_color("Datasets are NOT uniform.", _RED))
        return False

    print()
    print(_color("All fields match — datasets are uniform.", _GREEN))
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Check that a set of LeRobot datasets share the same parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "datasets",
        nargs="+",
        metavar="DATASET",
        help="Local directory paths or Hugging Face repo IDs (at least two).",
    )
    args = p.parse_args()

    if len(args.datasets) < 2:
        p.error("Provide at least two datasets to compare.")

    print("Loading metadata...")
    metas: List[Dict] = []
    for d in args.datasets:
        print(f"  loading {d} ...", end=" ", flush=True)
        try:
            meta = load_meta(d)
            metas.append(meta)
            n_ep = meta["info"].get("total_episodes", "?")
            print(f"OK  ({n_ep} episodes)")
        except Exception as exc:
            print(f"FAILED\n  Error: {exc}", file=sys.stderr)
            sys.exit(1)

    ok = compare_and_report(args.datasets, metas)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
