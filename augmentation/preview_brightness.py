#!/usr/bin/env python3
"""
Preview brightness levels on a single video from a LeRobot v3 HF Hub dataset.

Downloads one video file from --src and re-encodes it at every brightness
multiplier between --lo and --hi (step --step), so you can eyeball the range
before running the full augmentation.

Usage:
    python preview_brightness.py --src ETHrobotlearning/config3-green-blue-red
    python preview_brightness.py --src user/ds --lo 0.5 --hi 1.7 --step 0.1
    python preview_brightness.py --src user/ds --video-key observation.images.front --file 0
"""

import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
from huggingface_hub import hf_hub_download

VIDEO_CRF = 30
VIDEO_PRESET = 8


def discover_video_keys(info: dict) -> list[str]:
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def reencode_with_brightness(src_video: Path, out_video: Path, brightness: float, fps: float):
    out_video.parent.mkdir(parents=True, exist_ok=True)
    codec_name = "libsvtav1" if "libsvtav1" in av.codecs_available else "libx264"
    rate = int(round(fps))

    with av.open(str(src_video)) as src:
        in_stream = src.streams.video[0]
        w, h = in_stream.width, in_stream.height

    with av.open(str(out_video), mode="w") as dst:
        if codec_name == "libsvtav1":
            out_stream = dst.add_stream(codec_name, rate=rate)
            out_stream.options = {"crf": str(VIDEO_CRF), "preset": str(VIDEO_PRESET)}
        else:
            out_stream = dst.add_stream(codec_name, rate=rate)
            out_stream.options = {"crf": "18", "preset": "fast", "bf": "0"}
        out_stream.width = w
        out_stream.height = h
        out_stream.pix_fmt = "yuv420p"

        out_pts = 0
        with av.open(str(src_video)) as src:
            for frame in src.decode(video=0):
                arr = frame.to_ndarray(format="rgb24").astype(np.float32)
                arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
                out_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                out_frame = out_frame.reformat(format="yuv420p")
                out_frame.pts = out_pts
                out_pts += 1
                for pkt in out_stream.encode(out_frame):
                    dst.mux(pkt)
        for pkt in out_stream.encode():
            dst.mux(pkt)
    return out_pts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source HF repo (e.g. user/my_dataset)")
    parser.add_argument("--out-dir", default="brightness_preview",
                        help="Local output directory (default: brightness_preview)")
    parser.add_argument("--video-key", default=None,
                        help="Video feature key (default: first one in info.json)")
    parser.add_argument("--file", type=int, default=0,
                        help="Video file index to download (default: 0)")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Video chunk index (default: 0)")
    parser.add_argument("--lo", type=float, default=0.5, help="Min brightness multiplier (default: 0.5)")
    parser.add_argument("--hi", type=float, default=1.7, help="Max brightness multiplier (default: 1.7)")
    parser.add_argument("--step", type=float, default=0.1, help="Step between multipliers (default: 0.1)")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Source: {args.src}")

    # ----- Fetch info.json to learn video keys and fps -----
    info_path = Path(
        hf_hub_download(
            repo_id=args.src,
            repo_type="dataset",
            filename="meta/info.json",
            token=token,
        )
    )
    with open(info_path) as f:
        info = json.load(f)
    fps = float(info["fps"])
    video_keys = discover_video_keys(info)
    print(f"  fps={fps}  available video keys: {video_keys}")

    vk = args.video_key or video_keys[0]
    if vk not in video_keys:
        raise SystemExit(f"--video-key {vk!r} not found. Available: {video_keys}")

    # ----- Download the chosen video file -----
    repo_path = f"videos/{vk}/chunk-{args.chunk:03d}/file-{args.file:03d}.mp4"
    print(f"\nDownloading {repo_path}...")
    src_video = Path(
        hf_hub_download(
            repo_id=args.src,
            repo_type="dataset",
            filename=repo_path,
            token=token,
        )
    )
    print(f"  cached at: {src_video}  ({src_video.stat().st_size/1e6:.1f} MB)")

    # ----- Build brightness levels -----
    levels = []
    b = args.lo
    while b <= args.hi + 1e-9:
        levels.append(round(b, 3))
        b += args.step
    print(f"\nBrightness levels ({len(levels)}): {levels}")

    # ----- Re-encode each level -----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy original for reference (brightness=1.0 is not exactly the same as a copy
    # because it goes through a decode+encode roundtrip, so include both).
    print(f"\nWriting outputs to {out_dir.resolve()}/")
    for b in levels:
        tag = f"{b:.2f}".replace(".", "p")
        out_path = out_dir / f"brightness_{tag}.mp4"
        n = reencode_with_brightness(src_video, out_path, b, fps)
        print(f"  brightness={b:.2f}  -> {out_path.name}  ({n} frames)")

    import shutil
    ref_path = out_dir / "original_source.mp4"
    shutil.copy2(src_video, ref_path)
    print(f"  original (untouched copy) -> {ref_path.name}")

    print(f"\nDone. Open the files in {out_dir.resolve()} and pick a usable range.")


if __name__ == "__main__":
    main()
