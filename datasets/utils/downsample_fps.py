#!/usr/bin/env python3
"""Downsample a LeRobot v2 dataset to a lower FPS.

For each episode the script:
  1. Reads the parquet and keeps only the frames nearest to each target timestamp.
  2. Re-encodes every video stream at the target FPS using those frames.
  3. Rewrites the parquet with updated frame_index, index, and timestamp columns.
  4. Updates meta/info.json (fps, total_frames, video_info fps).

Input  — local path OR Hugging Face repo ID  (e.g. username/my_dataset)
Output — local path OR bare dataset name     (e.g. my_dataset_10fps → pushed to your HF account)

Usage:
    # local → local
    python datasets/utils/downsample_fps.py \\
        --input  /data/dataset_30fps  --output /data/dataset_10fps  --fps 10

    # HF → HF  (downloads input, pushes output to your account)
    python datasets/utils/downsample_fps.py \\
        --input  alice/my_robot_dataset  --output my_robot_dataset_10fps  --fps 10

    # HF → local
    python datasets/utils/downsample_fps.py \\
        --input  alice/my_robot_dataset  --output /data/dataset_10fps  --fps 10

    # local → HF
    python datasets/utils/downsample_fps.py \\
        --input  /data/dataset_30fps  --output my_robot_dataset_10fps  --fps 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import subprocess

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
    """True when s looks like 'user/repo' and is NOT an existing local path."""
    p = Path(s)
    if p.exists():
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts)


def _is_bare_name(s: str) -> bool:
    """True when s is a plain dataset name with no path separators → push to HF."""
    return "/" not in s and not Path(s).is_absolute() and not s.startswith(".")


def _download_hf_dataset(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo_id} from Hugging Face Hub …")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print("Download complete.")


def _push_to_hub(local_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Creating / updating HF dataset repo: {repo_id} …")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading to {repo_id} …")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed → https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Path helpers  (mirrors LeRobot v2 layout)
# ---------------------------------------------------------------------------

def _chunk(ep: int, chunk_size: int) -> int:
    return ep // chunk_size


def _parquet_path(root: Path, ep: int, chunk_size: int) -> Path:
    return root / f"data/chunk-{_chunk(ep, chunk_size):03d}/episode_{ep:06d}.parquet"


def _video_path(root: Path, ep: int, vkey: str, chunk_size: int) -> Path:
    return root / f"videos/chunk-{_chunk(ep, chunk_size):03d}/{vkey}/episode_{ep:06d}.mp4"


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_indices(n_frames: int, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Return sorted unique source-frame indices nearest to each target timestamp."""
    if n_frames == 0:
        return np.array([], dtype=int)
    last_t = (n_frames - 1) / src_fps
    target_times = np.arange(0, last_t + 1.0 / (2.0 * tgt_fps), 1.0 / tgt_fps)
    indices = np.clip(np.round(target_times * src_fps).astype(int), 0, n_frames - 1)
    return np.unique(indices)


# ---------------------------------------------------------------------------
# ffmpeg video I/O (no torchvision dependency)
# ---------------------------------------------------------------------------

def _vid_dims(path: str) -> Tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    w, h = map(int, out.split(","))
    return w, h


def _read_video(path: str) -> np.ndarray:
    """Read all frames as uint8 NHWC numpy array via ffmpeg pipe."""
    w, h = _vid_dims(path)
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "quiet", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    frame_size = h * w * 3
    n = len(raw) // frame_size
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w, 3).copy()


def _write_video(frames: np.ndarray, path: str, fps: float) -> None:
    """Write NHWC uint8 numpy array to MP4 via ffmpeg."""
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-v", "quiet", path,
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{proc.stderr.decode()}")


# ---------------------------------------------------------------------------
# Meta I/O
# ---------------------------------------------------------------------------

def _load_meta(path: Path) -> Dict:
    with open(path / "meta" / "info.json") as f:
        info = json.load(f)
    episodes: List[Dict] = []
    with open(path / "meta" / "episodes.jsonl") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    tasks: List[Dict] = []
    with open(path / "meta" / "tasks.jsonl") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return {"info": info, "episodes": episodes, "tasks": tasks}


# ---------------------------------------------------------------------------
# v3.0 meta helpers
# ---------------------------------------------------------------------------

def _detect_version(src: Path) -> str:
    """Return '2' for LeRobot v2.x (uses .jsonl), '3' for v3.x (uses parquets)."""
    return '2' if (src / "meta" / "episodes.jsonl").exists() else '3'


def _load_episodes_v3(src: Path) -> "pd.DataFrame":
    ep_files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        sys.exit(f"Error: no episodes parquet files found in {src}/meta/episodes/")
    return (pd.concat([pd.read_parquet(f) for f in ep_files])
              .sort_values("episode_index").reset_index(drop=True))


# ---------------------------------------------------------------------------
# Core downsampling
# ---------------------------------------------------------------------------

def downsample_v3(src: Path, dst: Path, tgt_fps: float) -> None:
    info = json.load(open(src / "meta" / "info.json"))
    src_fps: float = float(info["fps"])

    if tgt_fps >= src_fps:
        sys.exit(f"Error: target FPS ({tgt_fps}) must be less than source FPS ({src_fps}).")

    episodes_df = _load_episodes_v3(src)
    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print(f"FPS: {src_fps} → {tgt_fps}  |  episodes: {info['total_episodes']}  |  video streams: {vkeys or 'none'}")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(exist_ok=True)

    # ---- Step 1: downsample data parquets -----------------------------------
    global_idx = 0
    # ep_index → dict of updated fields to merge back into the episodes parquet
    new_ep_info: Dict[int, Dict] = {}

    data_file_keys = (episodes_df[["data/chunk_index", "data/file_index"]]
                      .drop_duplicates()
                      .sort_values(["data/chunk_index", "data/file_index"]))

    for _, (chunk_idx, file_idx) in tqdm(
            list(data_file_keys.iterrows()), desc="data parquets", unit="file"):
        ci, fi = int(chunk_idx), int(file_idx)
        src_pq = src / f"data/chunk-{ci:03d}/file-{fi:03d}.parquet"
        dst_pq = dst / f"data/chunk-{ci:03d}/file-{fi:03d}.parquet"
        dst_pq.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_pq)
        new_dfs: List["pd.DataFrame"] = []

        mask = ((episodes_df["data/chunk_index"] == chunk_idx) &
                (episodes_df["data/file_index"] == file_idx))
        for _, ep_row in episodes_df[mask].sort_values("episode_index").iterrows():
            ep_idx = int(ep_row["episode_index"])
            ep_df = df[df["episode_index"] == ep_idx].copy()
            keep = select_indices(len(ep_df), src_fps, tgt_fps)
            ep_df = ep_df.iloc[keep].copy().reset_index(drop=True)
            n_new = len(ep_df)

            if "frame_index" in ep_df.columns:
                ep_df["frame_index"] = np.arange(n_new, dtype=np.int64)
            if "index" in ep_df.columns:
                ep_df["index"] = np.arange(global_idx, global_idx + n_new, dtype=np.int64)
            if "timestamp" in ep_df.columns:
                ep_df["timestamp"] = (np.arange(n_new, dtype=np.float32) / tgt_fps)

            new_ep_info[ep_idx] = {
                "length": n_new,
                "dataset_from_index": global_idx,
                "dataset_to_index": global_idx + n_new,
            }
            global_idx += n_new
            new_dfs.append(ep_df)

        pd.concat(new_dfs, ignore_index=True).to_parquet(dst_pq, index=False)

    # ---- Step 2: downsample videos ------------------------------------------
    for vk in vkeys:
        ci_col = f"videos/{vk}/chunk_index"
        fi_col = f"videos/{vk}/file_index"
        fts_col = f"videos/{vk}/from_timestamp"

        vid_keys = (episodes_df[[ci_col, fi_col]]
                    .drop_duplicates()
                    .sort_values([ci_col, fi_col]))

        for _, (chunk_idx, file_idx) in tqdm(
                list(vid_keys.iterrows()), desc=f"video {vk}", unit="file"):
            ci, fi = int(chunk_idx), int(file_idx)
            vsrc = src / f"videos/{vk}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            vdst = dst / f"videos/{vk}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            vdst.parent.mkdir(parents=True, exist_ok=True)

            all_frames = _read_video(str(vsrc))
            n_vid = len(all_frames)

            mask = ((episodes_df[ci_col] == chunk_idx) & (episodes_df[fi_col] == file_idx))
            collected: List[np.ndarray] = []
            running_ts = 0.0

            for _, ep_row in episodes_df[mask].sort_values("episode_index").iterrows():
                ep_idx = int(ep_row["episode_index"])
                from_ts = float(ep_row[fts_col])
                src_ep_len = int(ep_row["length"])
                start_f = round(from_ts * src_fps)
                end_f = min(start_f + src_ep_len, n_vid)
                ep_frames = all_frames[start_f:end_f]

                keep = select_indices(len(ep_frames), src_fps, tgt_fps)
                new_frames = ep_frames[keep]
                collected.append(new_frames)

                n_new = new_ep_info[ep_idx]["length"]
                new_ep_info[ep_idx][fts_col] = running_ts
                new_ep_info[ep_idx][f"videos/{vk}/to_timestamp"] = running_ts + (n_new - 1) / tgt_fps
                new_ep_info[ep_idx][ci_col] = ci
                new_ep_info[ep_idx][fi_col] = fi
                running_ts += n_new / tgt_fps

            _write_video(np.concatenate(collected, axis=0), str(vdst), tgt_fps)

    # ---- Step 3: write updated episodes parquet -----------------------------
    new_ep_rows = []
    for _, ep_row in episodes_df.sort_values("episode_index").iterrows():
        ep_idx = int(ep_row["episode_index"])
        new_row = ep_row.to_dict()
        new_row.update(new_ep_info[ep_idx])
        new_ep_rows.append(new_row)

    new_ep_df = pd.DataFrame(new_ep_rows)
    for (chunk_idx, file_idx), grp in new_ep_df.groupby(
            ["meta/episodes/chunk_index", "meta/episodes/file_index"]):
        out = dst / f"meta/episodes/chunk-{int(chunk_idx):03d}/file-{int(file_idx):03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        grp.to_parquet(out, index=False)

    # ---- Step 4: copy tasks + update info.json ------------------------------
    shutil.copy2(src / "meta/tasks.parquet", dst / "meta/tasks.parquet")

    for name in ("stats.json", "stats.safetensors"):
        s = src / "meta" / name
        if s.exists():
            shutil.copy2(s, dst / "meta" / name)

    new_info = dict(info)
    new_info["fps"] = tgt_fps
    new_info["total_frames"] = global_idx
    for vk in vkeys:
        feat = new_info["features"][vk]
        for ik in ("info", "video_info"):
            if ik in feat and "video.fps" in feat[ik]:
                feat[ik]["video.fps"] = tgt_fps

    with open(dst / "meta/info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    print(f"Total frames: {global_idx}  (was {info['total_frames']})")


def downsample_v2(src: Path, dst: Path, tgt_fps: float) -> None:
    meta = _load_meta(src)
    info = meta["info"]
    src_fps: float = float(info["fps"])
    chunk_size: int = info["chunks_size"]

    if tgt_fps >= src_fps:
        sys.exit(f"Error: target FPS ({tgt_fps}) must be less than source FPS ({src_fps}).")

    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print(f"FPS: {src_fps} → {tgt_fps}  |  episodes: {info['total_episodes']}  |  video streams: {vkeys or 'none'}")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(exist_ok=True)

    global_idx = 0
    new_ep_metas: List[Dict] = []

    for ep_meta in tqdm(meta["episodes"], desc="downsampling", unit="ep"):
        ep: int = ep_meta["episode_index"]

        # --- parquet ---
        df = pd.read_parquet(_parquet_path(src, ep, chunk_size))
        keep = select_indices(len(df), src_fps, tgt_fps)
        df = df.iloc[keep].copy().reset_index(drop=True)
        n = len(df)

        if "frame_index" in df.columns:
            df["frame_index"] = np.arange(n)
        if "index" in df.columns:
            df["index"] = np.arange(global_idx, global_idx + n)
        if "timestamp" in df.columns:
            df["timestamp"] = np.arange(n, dtype=float) / tgt_fps

        out_parquet = _parquet_path(dst, ep, chunk_size)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)

        # --- videos ---
        for vk in vkeys:
            vsrc = _video_path(src, ep, vk, chunk_size)
            vdst = _video_path(dst, ep, vk, chunk_size)
            vdst.parent.mkdir(parents=True, exist_ok=True)

            frames = _read_video(str(vsrc))
            valid_keep = keep[keep < len(frames)]
            _write_video(frames[valid_keep], str(vdst), tgt_fps)

        new_ep_metas.append({**ep_meta, "length": n})
        global_idx += n

    # --- meta/info.json ---
    new_info = dict(info)
    new_info["fps"] = tgt_fps
    new_info["total_frames"] = global_idx
    n_ep = len(new_ep_metas)
    new_info["total_episodes"] = n_ep
    new_info["total_chunks"] = max(1, (n_ep - 1) // chunk_size + 1)
    new_info["splits"] = {"train": f"0:{n_ep}"}
    for vk in vkeys:
        vi = new_info["features"][vk].get("video_info", {})
        if vi:
            vi["video.fps"] = tgt_fps

    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)
    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for ep in new_ep_metas:
            f.write(json.dumps(ep) + "\n")
    with open(dst / "meta" / "tasks.jsonl", "w") as f:
        for t in meta["tasks"]:
            f.write(json.dumps(t) + "\n")

    # Copy stats if present (remain approximately valid after FPS change)
    for name in ("stats.json", "stats.safetensors"):
        s = src / "meta" / name
        if s.exists():
            shutil.copy2(s, dst / "meta" / name)

    print(f"Total frames: {global_idx}  (was {info['total_frames']})")


def downsample(src: Path, dst: Path, tgt_fps: float) -> None:
    if _detect_version(src) == '3':
        downsample_v3(src, dst, tgt_fps)
    else:
        downsample_v2(src, dst, tgt_fps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Downsample a LeRobot v2 dataset to a lower FPS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",  required=True, metavar="SRC",
                    help="Source dataset: local path or HF repo ID (user/dataset).")
    ap.add_argument("--output", required=True, metavar="DST",
                    help="Output: local path, or bare name to push to your HF account.")
    ap.add_argument("--fps",    required=True, type=float,
                    help="Target FPS (must be less than source FPS).")
    args = ap.parse_args()

    input_is_hf  = _is_hf_repo_id(args.input)
    output_is_hf = _is_bare_name(args.output) or _is_hf_repo_id(args.output)

    if input_is_hf or output_is_hf:
        _require_hf()

    # Determine full HF repo ID for output
    hf_output_repo: Optional[str] = None
    if output_is_hf:
        if _is_hf_repo_id(args.output):
            hf_output_repo = args.output
        else:
            username = _hf_whoami()
            hf_output_repo = f"{username}/{args.output}"

    # Resolve source to a local directory
    tmp_input_dir: Optional[tempfile.TemporaryDirectory] = None
    if input_is_hf:
        tmp_input_dir = tempfile.TemporaryDirectory(prefix="lerobot_src_")
        src = Path(tmp_input_dir.name)
        _download_hf_dataset(args.input, src)
    else:
        src = Path(args.input)
        if not src.is_dir():
            sys.exit(f"Error: source path does not exist: {src}")

    # Resolve destination to a local directory
    tmp_output_dir: Optional[tempfile.TemporaryDirectory] = None
    if output_is_hf:
        tmp_output_dir = tempfile.TemporaryDirectory(prefix="lerobot_dst_")
        dst = Path(tmp_output_dir.name)
    else:
        dst = Path(args.output)
        if dst.exists() and any(dst.iterdir()):
            sys.exit(f"Error: destination already exists and is not empty: {dst}")

    try:
        downsample(src, dst, args.fps)

        if output_is_hf:
            _push_to_hub(dst, hf_output_repo)
        else:
            print(f"\nDone  →  {dst}")
    finally:
        if tmp_input_dir is not None:
            tmp_input_dir.cleanup()
        if tmp_output_dir is not None:
            tmp_output_dir.cleanup()


if __name__ == "__main__":
    main()
