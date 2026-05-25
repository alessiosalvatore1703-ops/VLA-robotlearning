#!/usr/bin/env python3
"""Create a 30% full / 70% post-grasp LeRobot v3 dataset.

This script is designed for LeRobot v3 datasets on the Hugging Face Hub.  It
downloads metadata/data from the source dataset, proposes gripper-based
post-grasp trim points, optionally lets you review/correct them visually, then
writes a new LeRobot-compatible dataset locally.  If requested, it pushes the
new dataset to the Hub.

Example:
    python create_trimmed_lerobot_dataset.py \\
      --repo-id ETHrobotlearning/colours-task2 \\
      --output-repo-id ETHrobotlearning/colours-task2-trimmed-30full-70postgrasp \\
      --full-ratio 0.3 \\
      --manual-trim \\
      --push-to-hub \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm


DEFAULT_REPO_ID = "ETHrobotlearning/colours-task2"
DEFAULT_OUTPUT_REPO_ID = "ETHrobotlearning/colours-task2-trimmed-30full-70postgrasp"
DEFAULT_TRIM_INDICES_PATH = "trim_indices.json"

STAT_SUFFIXES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


@dataclass
class EpisodePlan:
    source_episode_index: int
    output_episode_index: int
    keep_full: bool
    trim_start: int
    original_length: int
    new_length: int
    task_index: int


def _repo_basename(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def default_output_dir(output_repo_id: str) -> Path:
    return Path("outputs") / "datasets" / _repo_basename(output_repo_id)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_info(root: Path) -> Dict[str, Any]:
    path = root / "meta" / "info.json"
    require_path(path, "meta/info.json")
    with open(path) as f:
        return json.load(f)


def load_tasks(root: Path) -> pd.DataFrame:
    path = root / "meta" / "tasks.parquet"
    require_path(path, "meta/tasks.parquet")
    return pd.read_parquet(path)


def load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files in {root / 'meta' / 'episodes'}")
    return (
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


def load_data(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files in {root / 'data'}")
    return (
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        .sort_values("index")
        .reset_index(drop=True)
    )


def video_keys(info: Dict[str, Any]) -> List[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def find_gripper_index(info: Dict[str, Any], column: str) -> int:
    feature = info.get("features", {}).get(column, {})
    names = feature.get("names") or []
    for idx, name in enumerate(names):
        if "gripper" in str(name).lower():
            return idx

    shape = feature.get("shape") or []
    if shape and int(shape[0]) > 0:
        fallback = int(shape[0]) - 1
        print(
            f"Warning: could not find a gripper name in {column}; "
            f"falling back to last dimension index {fallback}."
        )
        return fallback

    raise ValueError(f"Could not infer gripper dimension for column {column!r}")


def stack_series(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy())


def contiguous_true_start(mask: np.ndarray, run_length: int) -> Optional[int]:
    if run_length <= 1:
        indices = np.flatnonzero(mask)
        return int(indices[0]) if len(indices) else None

    count = 0
    for idx, value in enumerate(mask):
        count = count + 1 if bool(value) else 0
        if count >= run_length:
            return idx - run_length + 1
    return None


def propose_trim_start(
    episode_df: pd.DataFrame,
    info: Dict[str, Any],
    stable_closed_frames: int = 5,
    post_close_wait_frames: int = 8,
    min_remaining_frames: int = 20,
) -> Dict[str, Any]:
    """Propose the first post-grasp frame using a gripper-closure heuristic.

    The SO101 datasets inspected here expose `gripper.pos` as the last element
    of both `observation.state` and `action`.  The heuristic prefers the state
    signal because it reflects the measured robot configuration, and falls back
    to action if needed.
    """
    n = len(episode_df)
    if n <= min_remaining_frames:
        return {
            "trim_start": 0,
            "source": "too_short",
            "reason": f"episode length {n} <= min_remaining_frames {min_remaining_frames}",
        }

    candidates = []
    for column in ("observation.state", "action"):
        if column not in episode_df.columns:
            continue

        gripper_idx = find_gripper_index(info, column)
        signal = stack_series(episode_df[column])[:, gripper_idx].astype(float)
        early = float(np.median(signal[: max(3, min(10, n // 10))]))
        low = float(np.nanmin(signal))
        high = float(np.nanmax(signal))

        # In the inspected SO101 dataset, a closed gripper is much larger than
        # the open value.  This branch also supports robots where closure moves
        # the scalar in the opposite direction.
        closes_up = (high - early) >= (early - low)
        if closes_up:
            threshold = early + 0.45 * max(high - early, 1e-6)
            closed = signal >= threshold
            direction = "up"
        else:
            threshold = early - 0.45 * max(early - low, 1e-6)
            closed = signal <= threshold
            direction = "down"

        start = contiguous_true_start(closed, stable_closed_frames)
        if start is None:
            continue

        trim_start = int(start + post_close_wait_frames)
        trim_start = max(0, min(trim_start, n - min_remaining_frames))
        candidates.append(
            {
                "trim_start": trim_start,
                "closure_start": int(start),
                "source": column,
                "gripper_index": int(gripper_idx),
                "threshold": float(threshold),
                "direction": direction,
                "low": low,
                "high": high,
                "early_median": early,
            }
        )

    if candidates:
        # Prefer the measured state, then action.
        return candidates[0]

    fallback = max(0, min(int(round(n * 0.35)), n - min_remaining_frames))
    return {
        "trim_start": fallback,
        "source": "fallback_fraction",
        "reason": "no stable gripper closure detected",
    }


def read_trim_index_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"episodes": {}}
    with open(path) as f:
        data = json.load(f)
    data.setdefault("episodes", {})
    return data


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def choose_full_episodes(episode_indices: Iterable[int], full_ratio: float, seed: int) -> set[int]:
    episode_indices = sorted(int(value) for value in episode_indices)
    n_full = int(round(len(episode_indices) * full_ratio))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(episode_indices, size=n_full, replace=False)
    return set(int(value) for value in chosen)


def source_video_path(repo_id: str, video_key: str, chunk_idx: int, file_idx: int, token: Optional[str]) -> Path:
    filename = f"videos/{video_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token))


def decode_selected_frames(video_path: Path, frame_indices: List[int]) -> Dict[int, Any]:
    wanted = sorted(set(idx for idx in frame_indices if idx >= 0))
    if not wanted:
        return {}

    frames = {}
    wanted_set = set(wanted)
    max_idx = wanted[-1]

    with av.open(str(video_path)) as container:
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx in wanted_set:
                frames[frame_idx] = frame.to_image()
            if frame_idx >= max_idx:
                break
    return frames


def make_review_contact_sheet(
    video_path: Path,
    output_path: Path,
    source_video_frame: int,
    source_episode_frame: int,
    offsets: Tuple[int, ...] = (-20, -10, -5, 0, 5, 10, 20),
    center_label: str = "CENTER",
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Manual review requires Pillow: pip install pillow") from exc

    frame_indices = [max(0, source_video_frame + offset) for offset in offsets]
    frames = decode_selected_frames(video_path, frame_indices)
    if not frames:
        raise RuntimeError(f"Could not decode review frames from {video_path}")

    thumbs = []
    for offset, frame_idx in zip(offsets, frame_indices):
        image = frames.get(frame_idx)
        if image is None:
            continue
        image = image.convert("RGB")
        image.thumbnail((320, 240))
        canvas = Image.new("RGB", (340, 285), "white")
        canvas.paste(image, ((340 - image.width) // 2, 30))
        draw = ImageDraw.Draw(canvas)
        label = f"ep frame {source_episode_frame + offset} | video frame {frame_idx}"
        if offset == 0:
            label = f"{center_label}: " + label
            draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline="red", width=5)
        draw.text((10, 8), label, fill="black")
        thumbs.append(canvas)

    if not thumbs:
        raise RuntimeError(f"No review thumbnails could be created from {video_path}")

    sheet_width = sum(img.width for img in thumbs)
    sheet_height = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    x = 0
    for img in thumbs:
        sheet.paste(img, (x, 0))
        x += img.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def show_review_image(path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        plt.figure(figsize=(18, 5))
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception:
        print(f"Review contact sheet saved to: {path}")


def manual_review_episode(
    repo_id: str,
    token: Optional[str],
    episode_row: pd.Series,
    proposed_trim_start: int,
    episode_length: int,
    video_key: str,
    fps: float,
    review_dir: Path,
) -> int:
    ep_idx = int(episode_row["episode_index"])
    chunk_idx = int(episode_row[f"videos/{video_key}/chunk_index"])
    file_idx = int(episode_row[f"videos/{video_key}/file_index"])
    from_ts = float(episode_row[f"videos/{video_key}/from_timestamp"])

    video_path = source_video_path(repo_id, video_key, chunk_idx, file_idx, token)
    center = proposed_trim_start
    window_step = 40
    contact_sheet = review_dir / "current_review.jpg"

    def render_window(center_frame: int) -> Path:
        center_frame = max(0, min(int(center_frame), episode_length - 1))
        source_video_frame = int(round(from_ts * fps)) + center_frame
        make_review_contact_sheet(
            video_path=video_path,
            output_path=contact_sheet,
            source_video_frame=source_video_frame,
            source_episode_frame=center_frame,
            center_label="PROPOSED" if center_frame == proposed_trim_start else "CURRENT",
        )
        show_review_image(contact_sheet)
        return contact_sheet

    contact_sheet = render_window(center)

    while True:
        print("")
        print(f"Episode {ep_idx}: length={episode_length}, proposed trim_start={proposed_trim_start}")
        print(f"Current review center: {center}")
        print(f"Contact sheet: {contact_sheet}")
        print("Commands:")
        print("  Enter      accept the current review center as trim_start")
        print("  <integer>  accept that exact frame index")
        print("  n          show the next frame window")
        print("  p          show the previous frame window")
        print("  s <frame>  show a window centered on <frame>")
        print("  q          abort")
        raw = input("trim_start> ").strip()
        if raw == "":
            return center
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt("manual trim review aborted")
        if raw.lower() in {"n", "next"}:
            center = min(center + window_step, episode_length - 1)
            contact_sheet = render_window(center)
            continue
        if raw.lower() in {"p", "prev", "previous"}:
            center = max(center - window_step, 0)
            contact_sheet = render_window(center)
            continue
        if raw.lower().startswith("s "):
            try:
                center = int(raw.split(maxsplit=1)[1])
            except ValueError:
                print("Usage: s <frame>, for example: s 120")
                continue
            if not 0 <= center < episode_length:
                print(f"Frame must be in [0, {episode_length - 1}].")
                continue
            contact_sheet = render_window(center)
            continue
        try:
            value = int(raw)
        except ValueError:
            print("Unknown command. Type an integer, Enter, n, p, s <frame>, or q.")
            continue
        if 0 <= value < episode_length:
            return value
        print(f"Frame must be in [0, {episode_length - 1}].")


def build_trim_plan(
    repo_id: str,
    token: Optional[str],
    info: Dict[str, Any],
    episodes: pd.DataFrame,
    data: pd.DataFrame,
    full_ratio: float,
    seed: int,
    trim_indices_path: Path,
    manual_trim: bool,
    review_dir: Path,
    stable_closed_frames: int,
    post_close_wait_frames: int,
    min_remaining_frames: int,
    trim_only_validated: bool,
    only_validated_trimmed: bool,
) -> Tuple[List[EpisodePlan], Dict[str, Any]]:
    full_episodes = choose_full_episodes(episodes["episode_index"], full_ratio, seed)
    trim_data = read_trim_index_file(trim_indices_path)
    trim_data.update(
        {
            "repo_id": repo_id,
            "full_ratio": full_ratio,
            "seed": seed,
            "stable_closed_frames": stable_closed_frames,
            "post_close_wait_frames": post_close_wait_frames,
            "min_remaining_frames": min_remaining_frames,
        }
    )
    trim_data.setdefault("episodes", {})

    vkeys = video_keys(info)
    review_video_key = vkeys[0] if vkeys else ""
    fps = float(info["fps"])

    plans: List[EpisodePlan] = []
    output_ep_idx = 0

    for _, episode_row in tqdm(
        episodes.iterrows(),
        total=len(episodes),
        desc="planning trims",
        unit="episode",
    ):
        source_ep_idx = int(episode_row["episode_index"])
        ep_df = data[data["episode_index"].astype(int) == source_ep_idx].copy().reset_index(drop=True)
        if ep_df.empty:
            raise ValueError(f"Episode {source_ep_idx} has no frame rows in data parquet files")

        original_length = len(ep_df)
        task_index = int(ep_df["task_index"].iloc[0]) if "task_index" in ep_df.columns else -1
        key = str(source_ep_idx)
        keep_full = source_ep_idx in full_episodes

        existing = trim_data["episodes"].get(key)
        if only_validated_trimmed:
            if existing is None or not bool(existing.get("validated", False)):
                continue
            keep_full = False
            trim_start = int(existing.get("trim_start", 0))
        elif trim_only_validated:
            if existing is not None and bool(existing.get("validated", False)):
                keep_full = False
                trim_start = int(existing.get("trim_start", 0))
            else:
                keep_full = True
                trim_start = 0
        elif existing is not None:
            keep_full = bool(existing.get("keep_full", keep_full))
            trim_start = int(existing.get("trim_start", 0 if keep_full else existing.get("auto_trim_start", 0)))
        else:
            proposal = propose_trim_start(
                ep_df,
                info,
                stable_closed_frames=stable_closed_frames,
                post_close_wait_frames=post_close_wait_frames,
                min_remaining_frames=min_remaining_frames,
            )
            auto_trim_start = 0 if keep_full else int(proposal["trim_start"])
            trim_start = 0 if keep_full else auto_trim_start

            if manual_trim and not keep_full:
                trim_start = manual_review_episode(
                    repo_id=repo_id,
                    token=token,
                    episode_row=episode_row,
                    proposed_trim_start=auto_trim_start,
                    episode_length=original_length,
                    video_key=review_video_key,
                    fps=fps,
                    review_dir=review_dir,
                )

            trim_data["episodes"][key] = {
                "keep_full": keep_full,
                "trim_start": int(trim_start),
                "auto_trim_start": int(auto_trim_start),
                "original_length": int(original_length),
                "proposal": proposal,
                "validated": bool(manual_trim and not keep_full),
            }
            atomic_write_json(trim_indices_path, trim_data)

        trim_start = max(0, min(trim_start, original_length - 1))
        if original_length - trim_start < 1:
            trim_start = 0

        plans.append(
            EpisodePlan(
                source_episode_index=source_ep_idx,
                output_episode_index=output_ep_idx,
                keep_full=keep_full,
                trim_start=trim_start,
                original_length=original_length,
                new_length=original_length - trim_start,
                task_index=task_index,
            )
        )
        output_ep_idx += 1

    # Save once more in case the file existed and only metadata changed.
    atomic_write_json(trim_indices_path, trim_data)
    return plans, trim_data


def numeric_stats(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([arr.shape[0]] * arr.shape[1]),
        "q01": np.quantile(arr, 0.01, axis=0),
        "q10": np.quantile(arr, 0.10, axis=0),
        "q50": np.quantile(arr, 0.50, axis=0),
        "q90": np.quantile(arr, 0.90, axis=0),
        "q99": np.quantile(arr, 0.99, axis=0),
    }


def set_episode_stats(row: Dict[str, Any], ep_df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in ep_df.columns:
            continue
        try:
            if isinstance(ep_df[column].iloc[0], np.ndarray):
                values = np.stack(ep_df[column].to_numpy())
            else:
                values = ep_df[column].to_numpy()
            stats = numeric_stats(values)
        except Exception:
            continue

        for suffix in STAT_SUFFIXES:
            key = f"stats/{column}/{suffix}"
            if key in row and suffix in stats:
                row[key] = stats[suffix]


def encode_video_segment(
    src_video: Path,
    dst_video: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    requested_codec: str,
    crf: int,
    preset: int,
) -> Tuple[int, str]:
    """Write frames [start_frame, end_frame) to a new MP4 and return count/codec."""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    if end_frame <= start_frame:
        raise ValueError(f"Invalid video segment {start_frame}:{end_frame} for {src_video}")

    if requested_codec == "auto":
        codec_name = "libsvtav1" if "libsvtav1" in av.codecs_available else "libx264"
    else:
        codec_name = requested_codec

    if codec_name not in av.codecs_available:
        raise RuntimeError(f"Requested video codec {codec_name!r} is not available in PyAV/FFmpeg")

    with av.open(str(src_video)) as src:
        in_stream = src.streams.video[0]
        width, height = in_stream.width, in_stream.height

    out_count = 0
    with av.open(str(dst_video), mode="w") as dst:
        out_stream = dst.add_stream(codec_name, rate=int(round(fps)))
        if codec_name == "libsvtav1":
            out_stream.options = {"crf": str(crf), "preset": str(preset)}
            output_codec = "av1"
        elif codec_name == "libx264":
            out_stream.options = {"crf": "18", "preset": "fast", "bf": "0"}
            output_codec = "h264"
        else:
            output_codec = codec_name.replace("lib", "")

        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"

        with av.open(str(src_video)) as src:
            for frame_idx, frame in enumerate(src.decode(video=0)):
                if frame_idx < start_frame:
                    continue
                if frame_idx >= end_frame:
                    break
                arr = frame.to_ndarray(format="yuv420p")
                out_frame = av.VideoFrame.from_ndarray(arr, format="yuv420p")
                out_frame.pts = out_count
                out_count += 1
                for packet in out_stream.encode(out_frame):
                    dst.mux(packet)

        for packet in out_stream.encode():
            dst.mux(packet)

    if out_count == 0:
        raise RuntimeError(f"No frames written for {src_video} segment {start_frame}:{end_frame}")
    return out_count, output_codec


def copy_non_episode_meta(src_root: Path, dst_root: Path) -> None:
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    for path in (src_root / "meta").iterdir():
        if path.name in {"info.json", "episodes"}:
            continue
        target = dst_meta / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)

    gitattributes = src_root / ".gitattributes"
    if gitattributes.exists():
        shutil.copy2(gitattributes, dst_root / ".gitattributes")


def write_dataset(
    repo_id: str,
    token: Optional[str],
    src_root: Path,
    output_dir: Path,
    info: Dict[str, Any],
    episodes: pd.DataFrame,
    data: pd.DataFrame,
    plans: List[EpisodePlan],
    video_codec: str,
    video_crf: int,
    video_preset: int,
) -> Dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    copy_non_episode_meta(src_root, output_dir)

    fps = float(info["fps"])
    vkeys = video_keys(info)
    data_dir = output_dir / "data" / "chunk-000"
    episodes_dir = output_dir / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    ep_by_index = {int(row["episode_index"]): row for _, row in episodes.iterrows()}
    global_frame_index = 0
    new_episode_rows: List[Dict[str, Any]] = []
    output_codec: Optional[str] = None

    for plan in tqdm(plans, desc="writing dataset", unit="episode"):
        src_ep_df = data[data["episode_index"].astype(int) == plan.source_episode_index].copy().reset_index(drop=True)
        new_df = src_ep_df.iloc[plan.trim_start :].copy().reset_index(drop=True)
        if new_df.empty:
            raise ValueError(f"Output episode {plan.output_episode_index} would be empty")

        n = len(new_df)
        new_df["episode_index"] = np.int64(plan.output_episode_index)
        new_df["frame_index"] = np.arange(n, dtype=np.int64)
        new_df["timestamp"] = (np.arange(n, dtype=np.float32) / np.float32(fps)).astype(np.float32)
        new_df["index"] = np.arange(global_frame_index, global_frame_index + n, dtype=np.int64)

        data_path = data_dir / f"file-{plan.output_episode_index:03d}.parquet"
        pq.write_table(pa.Table.from_pandas(new_df, preserve_index=False), data_path)

        src_row = ep_by_index[plan.source_episode_index]
        new_row = src_row.to_dict()
        new_row["episode_index"] = plan.output_episode_index
        new_row["length"] = n
        new_row["dataset_from_index"] = global_frame_index
        new_row["dataset_to_index"] = global_frame_index + n
        new_row["data/chunk_index"] = 0
        new_row["data/file_index"] = plan.output_episode_index
        new_row["meta/episodes/chunk_index"] = 0
        new_row["meta/episodes/file_index"] = 0

        set_episode_stats(
            new_row,
            new_df,
            columns=[
                "action",
                "observation.state",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ],
        )

        for vkey in vkeys:
            src_chunk = int(src_row[f"videos/{vkey}/chunk_index"])
            src_file = int(src_row[f"videos/{vkey}/file_index"])
            src_from_ts = float(src_row[f"videos/{vkey}/from_timestamp"])
            src_start_frame = int(round(src_from_ts * fps)) + plan.trim_start
            src_end_frame = src_start_frame + n

            src_video = source_video_path(repo_id, vkey, src_chunk, src_file, token)
            dst_video = output_dir / "videos" / vkey / "chunk-000" / f"file-{plan.output_episode_index:03d}.mp4"
            written, codec = encode_video_segment(
                src_video=src_video,
                dst_video=dst_video,
                start_frame=src_start_frame,
                end_frame=src_end_frame,
                fps=fps,
                requested_codec=video_codec,
                crf=video_crf,
                preset=video_preset,
            )
            if written != n:
                raise RuntimeError(
                    f"Video frame count mismatch for episode {plan.source_episode_index}: "
                    f"wrote {written}, expected {n}"
                )
            output_codec = codec

            new_row[f"videos/{vkey}/chunk_index"] = 0
            new_row[f"videos/{vkey}/file_index"] = plan.output_episode_index
            new_row[f"videos/{vkey}/from_timestamp"] = 0.0
            new_row[f"videos/{vkey}/to_timestamp"] = round(n / fps, 6)

        new_episode_rows.append(new_row)
        global_frame_index += n

    episodes_out = pd.DataFrame(new_episode_rows).sort_values("episode_index").reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(episodes_out, preserve_index=False),
        episodes_dir / "file-000.parquet",
    )

    new_info = dict(info)
    new_info["total_episodes"] = len(plans)
    new_info["total_frames"] = global_frame_index
    new_info["total_videos"] = len(plans) * len(vkeys)
    new_info["total_chunks"] = 1
    new_info["splits"] = {"train": f"0:{len(plans)}"}

    if output_codec is not None:
        for vkey in vkeys:
            feature = new_info["features"][vkey]
            video_info = feature.get("info") or feature.get("video_info") or {}
            video_info["video.codec"] = output_codec
            video_info["video.fps"] = int(round(fps))
            video_info["video.pix_fmt"] = "yuv420p"
            feature["info"] = video_info

    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    return new_info


def validate_output_dataset(output_dir: Path, required_columns: List[str]) -> None:
    print("\nValidating output dataset...")
    info = load_info(output_dir)
    _ = load_tasks(output_dir)
    episodes = load_episodes(output_dir)
    data = load_data(output_dir)

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required data columns: {missing_columns}")

    expected_eps = list(range(int(info["total_episodes"])))
    actual_eps = sorted(int(value) for value in episodes["episode_index"].unique())
    if actual_eps != expected_eps:
        raise ValueError("Episode metadata indices are not consecutive from 0")

    data_eps = sorted(int(value) for value in data["episode_index"].unique())
    if data_eps != expected_eps:
        raise ValueError("Data episode_index values are not consecutive from 0")

    for ep_idx, ep_df in data.groupby("episode_index", sort=True):
        if ep_df.empty:
            raise ValueError(f"Episode {ep_idx} is empty")
        frame_indices = ep_df["frame_index"].astype(int).to_numpy()
        expected_frame_indices = np.arange(len(ep_df), dtype=int)
        if not np.array_equal(frame_indices, expected_frame_indices):
            raise ValueError(f"frame_index is not consecutive from 0 in episode {ep_idx}")

    if len(data) != int(info["total_frames"]):
        raise ValueError(f"total_frames mismatch: info={info['total_frames']} data_rows={len(data)}")

    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        if int(row["length"]) <= 0:
            raise ValueError(f"Episode {ep_idx} has non-positive length")
        data_path = output_dir / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
        if not data_path.exists():
            raise ValueError(f"Episode {ep_idx} references missing data file: {data_path}")
        for vkey in video_keys(info):
            video_path = output_dir / (
                f"videos/{vkey}/chunk-{int(row[f'videos/{vkey}/chunk_index']):03d}/"
                f"file-{int(row[f'videos/{vkey}/file_index']):03d}.mp4"
            )
            if not video_path.exists():
                raise ValueError(f"Episode {ep_idx} references missing video file: {video_path}")

    try:
        import lerobot  # noqa: F401
        print("  LeRobot import check: OK")
    except Exception as exc:
        print(f"  LeRobot import check skipped/unavailable: {exc}")

    print("  Structural checks passed.")


def print_stats(plans: List[EpisodePlan]) -> None:
    original_lengths = np.array([plan.original_length for plan in plans], dtype=float)
    new_lengths = np.array([plan.new_length for plan in plans], dtype=float)
    n_full = sum(1 for plan in plans if plan.keep_full)
    n_trimmed = len(plans) - n_full
    before = int(original_lengths.sum())
    after = int(new_lengths.sum())
    kept = 100.0 * after / before if before else math.nan

    print("")
    print("Dataset trimming statistics")
    print("===========================")
    print(f"Original episodes:              {len(plans)}")
    print(f"Full episodes kept:             {n_full}")
    print(f"Trimmed episodes:               {n_trimmed}")
    print(f"Average original episode length:{original_lengths.mean():9.2f} frames")
    print(f"Average new episode length:     {new_lengths.mean():9.2f} frames")
    print(f"Total frames before trimming:   {before}")
    print(f"Total frames after trimming:    {after}")
    print(f"Percentage of data kept:        {kept:8.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a LeRobot dataset with 30% full episodes and 70% post-grasp trimmed episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Source HF dataset repo ID.")
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID, help="Destination HF dataset repo ID.")
    parser.add_argument("--output-dir", default=None, help="Local output directory. Defaults to outputs/datasets/<output repo name>.")
    parser.add_argument("--full-ratio", type=float, default=0.3, help="Fraction of episodes kept as full trajectories.")
    parser.add_argument("--manual-trim", action="store_true", help="Interactively review proposed trim frames.")
    parser.add_argument(
        "--trim-only-validated",
        action="store_true",
        help="Only trim episodes with validated entries in trim_indices.json; keep every other episode full.",
    )
    parser.add_argument(
        "--only-validated-trimmed",
        action="store_true",
        help="Output only manually validated trimmed episodes from trim_indices.json; drop all other episodes.",
    )
    parser.add_argument("--trim-indices-path", default=DEFAULT_TRIM_INDICES_PATH, help="JSON cache of accepted trim indices.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the saved local dataset to --output-repo-id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to select full episodes.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token. Defaults to $HF_TOKEN or cached login.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output-dir if it already exists.")
    parser.add_argument("--stable-closed-frames", type=int, default=5, help="Consecutive closed-gripper frames required.")
    parser.add_argument("--post-close-wait-frames", type=int, default=8, help="Frames to wait after closure before trimming.")
    parser.add_argument("--min-remaining-frames", type=int, default=20, help="Minimum frames left after trimming.")
    parser.add_argument("--review-dir", default="trim_review_frames", help="Directory for manual review contact sheets.")
    parser.add_argument("--video-codec", default="auto", help="Output encoder: auto, libsvtav1, or libx264.")
    parser.add_argument("--video-crf", type=int, default=35, help="SVT-AV1 CRF if using libsvtav1.")
    parser.add_argument("--video-preset", type=int, default=8, help="SVT-AV1 preset if using libsvtav1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.full_ratio <= 1.0:
        raise ValueError("--full-ratio must be in [0, 1]")
    if args.trim_only_validated and args.only_validated_trimmed:
        raise ValueError("Use only one of --trim-only-validated or --only-validated-trimmed.")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.output_repo_id)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")

    print(f"Source dataset:      {args.repo_id}")
    print(f"Local output:        {output_dir}")
    print(f"Optional Hub output: {args.output_repo_id}")
    print(f"Trim index cache:    {args.trim_indices_path}")

    print("\n[1/6] Downloading metadata and data parquets...")
    src_root = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            ignore_patterns=["videos/*"],
            token=args.token,
        )
    )
    print(f"  Cached at: {src_root}")

    print("\n[2/6] Loading and inspecting dataset...")
    info = load_info(src_root)
    episodes = load_episodes(src_root)
    data = load_data(src_root)
    tasks = load_tasks(src_root)
    vkeys = video_keys(info)
    if not vkeys:
        raise ValueError("This script expects at least one video feature in the LeRobot dataset.")

    print(f"  LeRobot codebase_version: {info.get('codebase_version')}")
    print(f"  fps: {info.get('fps')}")
    print(f"  episodes: {len(episodes)}")
    print(f"  frames: {len(data)}")
    print(f"  tasks: {len(tasks)}")
    print(f"  video keys: {vkeys}")
    print(f"  data columns: {list(data.columns)}")
    print(f"  gripper index in observation.state: {find_gripper_index(info, 'observation.state')}")
    print(f"  gripper index in action: {find_gripper_index(info, 'action')}")

    print("\n[3/6] Planning full/trimmed episodes and trim starts...")
    plans, _trim_cache = build_trim_plan(
        repo_id=args.repo_id,
        token=args.token,
        info=info,
        episodes=episodes,
        data=data,
        full_ratio=args.full_ratio,
        seed=args.seed,
        trim_indices_path=Path(args.trim_indices_path),
        manual_trim=args.manual_trim,
        review_dir=Path(args.review_dir),
        stable_closed_frames=args.stable_closed_frames,
        post_close_wait_frames=args.post_close_wait_frames,
        min_remaining_frames=args.min_remaining_frames,
        trim_only_validated=args.trim_only_validated,
        only_validated_trimmed=args.only_validated_trimmed,
    )
    print_stats(plans)

    print("\n[4/6] Writing local LeRobot dataset...")
    new_info = write_dataset(
        repo_id=args.repo_id,
        token=args.token,
        src_root=src_root,
        output_dir=output_dir,
        info=info,
        episodes=episodes,
        data=data,
        plans=plans,
        video_codec=args.video_codec,
        video_crf=args.video_crf,
        video_preset=args.video_preset,
    )
    print(f"  Wrote {new_info['total_episodes']} episodes and {new_info['total_frames']} frames.")

    print("\n[5/6] Validating saved dataset...")
    validate_output_dataset(output_dir, required_columns=list(data.columns))

    if args.push_to_hub:
        print("\n[6/6] Pushing to Hugging Face Hub...")
        api = HfApi(token=args.token)
        api.create_repo(repo_id=args.output_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(output_dir), repo_id=args.output_repo_id, repo_type="dataset")
        print(f"  Pushed -> https://huggingface.co/datasets/{args.output_repo_id}")
    else:
        print("\n[6/6] Push skipped. Local dataset is ready.")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
