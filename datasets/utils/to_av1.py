#!/usr/bin/env python3
"""Re-encode a LeRobot dataset's videos from H.264 to AV1 (SVT-AV1).

All data parquets and meta files are copied unchanged.
Only the .mp4 files under videos/ are re-encoded.
meta/info.json is updated to reflect the new codec.

Input  — local path OR Hugging Face repo ID  (e.g. username/my_dataset)
Output — local path OR bare dataset name     (e.g. my_dataset_av1 → pushed to your HF account)

Usage:
    # local → local
    python datasets/utils/to_av1.py --input /data/dataset_h264 --output /data/dataset_av1

    # HF → HF  (downloads input, pushes output to your account)
    python datasets/utils/to_av1.py --input alice/my_dataset --output my_dataset_av1

    # HF → local
    python datasets/utils/to_av1.py --input alice/my_dataset --output /data/dataset_av1

    # local → HF
    python datasets/utils/to_av1.py --input /data/dataset_h264 --output my_dataset_av1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm


# ---------------------------------------------------------------------------
# HF helpers  (same pattern as downsample_fps.py)
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
    print(f"Downloading {repo_id} from Hugging Face Hub …")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    print("Download complete.")


def _push_to_hub(local_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Creating / updating HF dataset repo: {repo_id} …")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading to {repo_id} …")
    api.upload_folder(folder_path=str(local_path), repo_id=repo_id, repo_type="dataset")
    print(f"Pushed → https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# AV1 encoding
# ---------------------------------------------------------------------------

def _check_encoder() -> str:
    out = subprocess.run(
        ["ffmpeg", "-encoders", "-v", "quiet"],
        capture_output=True, text=True,
    ).stdout
    for enc in ("libsvtav1", "libaom-av1"):
        if enc in out:
            return enc
    sys.exit("Error: no AV1 encoder found in ffmpeg (need libsvtav1 or libaom-av1).")


def _reencode(src: Path, dst: Path, encoder: str, crf: int, preset: int) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c:v", encoder, "-pix_fmt", "yuv420p"]
    if encoder == "libsvtav1":
        cmd += ["-crf", str(crf), "-preset", str(preset)]
    else:  # libaom-av1
        cmd += ["-crf", str(crf), "-b:v", "0", "-cpu-used", str(preset)]
    cmd += ["-v", "quiet", str(dst)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg AV1 encode failed for {src}:\n{proc.stderr.decode()}")


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, crf: int, preset: int) -> None:
    encoder = _check_encoder()
    print(f"AV1 encoder : {encoder}  (CRF {crf}, preset {preset})")

    # Copy entire dataset tree to dst, then re-encode videos in-place
    print("Copying dataset …")
    if dst.exists():
        sys.exit(f"Error: destination already exists: {dst}")
    shutil.copytree(src, dst)

    video_files = sorted((dst / "videos").rglob("*.mp4"))
    if not video_files:
        sys.exit("Error: no .mp4 files found under videos/")

    print(f"Re-encoding {len(video_files)} video file(s) to AV1 …")
    for vf in tqdm(video_files, unit="file"):
        tmp = vf.with_suffix(".tmp.mp4")
        _reencode(vf, tmp, encoder, crf, preset)
        vf.unlink()
        tmp.rename(vf)

    # Update codec in meta/info.json
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for feat in info.get("features", {}).values():
        if feat.get("dtype") == "video":
            for key in ("info", "video_info"):
                if key in feat and "video.codec" in feat[key]:
                    feat[key]["video.codec"] = "av1"
    info_path.write_text(json.dumps(info, indent=2))

    print(f"Done.  {len(video_files)} file(s) converted to AV1.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-encode a LeRobot dataset's videos from H.264 to AV1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",  required=True, metavar="SRC",
                    help="Source dataset: local path or HF repo ID (user/dataset).")
    ap.add_argument("--output", required=True, metavar="DST",
                    help="Output: local path, or bare name to push to your HF account.")
    ap.add_argument("--crf",    type=int, default=30,
                    help="CRF quality (lower = better quality, larger file). Default: 30.")
    ap.add_argument("--preset", type=int, default=8,
                    help="Encoder speed preset (SVT-AV1: 0=slowest…13=fastest). Default: 8.")
    args = ap.parse_args()

    input_is_hf  = _is_hf_repo_id(args.input)
    output_is_hf = _is_bare_name(args.output) or _is_hf_repo_id(args.output)

    if input_is_hf or output_is_hf:
        _require_hf()

    hf_output_repo: Optional[str] = None
    if output_is_hf:
        if _is_hf_repo_id(args.output):
            hf_output_repo = args.output
        else:
            hf_output_repo = f"{_hf_whoami()}/{args.output}"

    tmp_input_dir: Optional[tempfile.TemporaryDirectory] = None
    if input_is_hf:
        tmp_input_dir = tempfile.TemporaryDirectory(prefix="lerobot_av1_src_")
        src = Path(tmp_input_dir.name)
        _download_hf_dataset(args.input, src)
    else:
        src = Path(args.input)
        if not src.is_dir():
            sys.exit(f"Error: source path does not exist: {src}")

    tmp_output_dir: Optional[tempfile.TemporaryDirectory] = None
    if output_is_hf:
        tmp_output_dir = tempfile.TemporaryDirectory(prefix="lerobot_av1_dst_")
        dst = Path(tmp_output_dir.name) / "dataset"
    else:
        dst = Path(args.output)

    try:
        convert(src, dst, args.crf, args.preset)
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
