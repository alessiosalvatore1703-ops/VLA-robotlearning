"""
Celebrity-swap augmentation pipeline — CLI entry point.

Usage:
    # Prepare celeb bank (one-time):
    python -m augmentation.celebrity_swap.run_pipeline prep-bank

    # Run the swap pipeline:
    python -m augmentation.celebrity_swap.run_pipeline run \\
        --src ETHrobotlearning/task3-TOY-clean \\
        --dst ./output/task3-celebswap \\
        --n-aug 3 \\
        --workers 4

    # Visualise detection on random episodes (before any swapping):
    python -m augmentation.celebrity_swap.run_pipeline viz \\
        --src ETHrobotlearning/task3-TOY-clean \\
        --n-episodes 20

    # Push finished dataset to Hub:
    python -m augmentation.celebrity_swap.run_pipeline push \\
        --dst ./output/task3-celebswap \\
        --hub-repo ETHrobotlearning/task3-TOY-celebswap
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_prep_bank(args) -> None:
    from augmentation.celebrity_swap.prepare_celeb_bank import build_bank
    build_bank(Path(args.bank_dir), args.hf_repo)


def cmd_run(args) -> None:
    from augmentation.celebrity_swap.build_lerobot import build_dataset, _load_tasks, _load_episodes
    from augmentation.celebrity_swap.episode_worker import (
        process_episode, OOD_CELEBS, _extract_target_celeb
    )

    src_path = _resolve_src(args.src, args.cache_dir)
    dst_path = Path(args.dst)
    bank_dir = Path(args.bank_dir)

    if not (bank_dir / "textures").exists():
        logger.error("Bank not found at %s — run 'prep-bank' first.", bank_dir)
        sys.exit(1)

    with open(src_path / "meta" / "info.json") as f:
        original_info = json.load(f)

    fps = float(original_info.get("fps", 10))
    src_tasks    = _load_tasks(src_path)
    src_episodes = _load_episodes(src_path)
    n_episodes   = len(src_episodes)

    logger.info("Source: %s  (%d episodes)", src_path, n_episodes)
    logger.info("Will generate %d augmentations per episode → %d new episodes",
                args.n_aug, n_episodes * args.n_aug)

    # Build a balanced target-celeb queue so all OOD celebs appear roughly equally
    rng = random.Random(args.seed)
    target_queue = _balanced_target_queue(n_episodes * args.n_aug, OOD_CELEBS, rng)

    # Build work items for the pool
    tmp_dir = dst_path / "_tmp_videos"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    work_items = []
    for ep_info in src_episodes:
        ep_idx   = ep_info["episode_index"]
        task_str = ep_info.get("task_str", "")
        # Resolve task string from source tasks parquet
        vid_path = src_path / f"videos/{_VIDEO_KEY}/chunk-000/file-{ep_idx:03d}.mp4"
        work_items.append({
            "src_video_path": vid_path,
            "dst_dir":        tmp_dir / f"ep{ep_idx:06d}",
            "src_task_str":   task_str,
            "bank_dir":       bank_dir,
            "n_aug":          args.n_aug,
            "seed":           args.seed + ep_idx,
            "fps":            fps,
            "target_celebs":  [target_queue.pop(0) for _ in range(args.n_aug)
                                if target_queue],
        })

    logger.info("Processing with %d workers …", args.workers)
    all_aug_results: list[dict] = []

    if args.workers == 1:
        for item in work_items:
            results = _worker_fn(item)
            all_aug_results.extend(_to_records(item, results))
    else:
        with mp.Pool(args.workers) as pool:
            for results, item in zip(pool.imap(_worker_fn, work_items), work_items):
                all_aug_results.extend(_to_records(item, results))

    logger.info("Compositing done. Building LeRobot v3 dataset …")
    build_dataset(
        src_path=src_path,
        dst_path=dst_path,
        aug_results=all_aug_results,
        original_info=original_info,
        include_originals=not args.no_originals,
    )

    # Clean up tmp videos (already moved by build_dataset)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if args.push_hub:
        _push_to_hub(dst_path, args.push_hub)


def cmd_viz(args) -> None:
    from augmentation.celebrity_swap.viz_qa import visualise_detection
    src_path = _resolve_src(args.src, args.cache_dir)
    visualise_detection(src_path, n_episodes=args.n_episodes, out_dir=Path(args.out_dir))


def cmd_push(args) -> None:
    _push_to_hub(Path(args.dst), args.hub_repo)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VIDEO_KEY = "observation.images.front"


def _resolve_src(src: str, cache_dir: str) -> Path:
    p = Path(src)
    if p.exists():
        return p
    logger.info("Downloading %s from HuggingFace …", src)
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id=src,
        repo_type="dataset",
        local_dir=Path(cache_dir) / src.replace("/", "--"),
    )
    return Path(local)


def _balanced_target_queue(n_total: int, pool: list[str], rng: random.Random) -> list[str]:
    """Return a list of length n_total where each celeb in pool appears as evenly as possible."""
    full_rounds = n_total // len(pool)
    remainder   = n_total % len(pool)
    queue = pool * full_rounds
    queue += rng.sample(pool, remainder)
    rng.shuffle(queue)
    return queue


def _worker_fn(item: dict):
    """Unpickle-able top-level function for multiprocessing."""
    from augmentation.celebrity_swap.episode_worker import process_episode
    rng = random.Random(item["seed"])
    return process_episode(
        src_video_path=Path(item["src_video_path"]),
        dst_dir=Path(item["dst_dir"]),
        src_task_str=item["src_task_str"],
        bank_dir=Path(item["bank_dir"]),
        n_aug=item["n_aug"],
        rng=rng,
        target_celeb_queue=list(item.get("target_celebs", [])),
        fps=item["fps"],
    )


def _to_records(item: dict, results) -> list[dict]:
    src_ep_idx = int(Path(item["dst_dir"]).name.replace("ep", ""))
    out = []
    for r in results:
        out.append({
            "src_episode_idx": src_ep_idx,
            "new_task_str":    r.new_task_str,
            "new_video_path":  str(r.new_video_path),
        })
    return out


def _push_to_hub(dst: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(dst), repo_id=repo_id, repo_type="dataset")
    logger.info("Pushed to https://huggingface.co/datasets/%s", repo_id)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="celebrity_swap")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prep-bank
    pb = sub.add_parser("prep-bank", help="Download celeb30 and build A5 textures.")
    pb.add_argument("--bank-dir",  default="./bank")
    pb.add_argument("--hf-repo",   default="ielminawi/celeb30")

    # run
    rp = sub.add_parser("run", help="Run the full swap pipeline.")
    rp.add_argument("--src",          default="ETHrobotlearning/task3-TOY-clean",
                    help="Local path or HF repo ID of the source dataset.")
    rp.add_argument("--dst",          required=True,
                    help="Output directory for the augmented dataset.")
    rp.add_argument("--bank-dir",     default="./bank")
    rp.add_argument("--cache-dir",    default="./hf_cache")
    rp.add_argument("--n-aug",        type=int, default=3,
                    help="Augmentations per source episode.")
    rp.add_argument("--workers",      type=int, default=4)
    rp.add_argument("--seed",         type=int, default=42)
    rp.add_argument("--no-originals", action="store_true",
                    help="Omit original (unswapped) episodes from output.")
    rp.add_argument("--push-hub",     default="",
                    help="If set, push the finished dataset to this HF repo ID.")

    # viz
    vp = sub.add_parser("viz", help="Visualise portrait detection on random frames.")
    vp.add_argument("--src",         default="ETHrobotlearning/task3-TOY-clean")
    vp.add_argument("--cache-dir",   default="./hf_cache")
    vp.add_argument("--n-episodes",  type=int, default=20)
    vp.add_argument("--out-dir",     default="./viz_output")

    # push
    pp = sub.add_parser("push", help="Push a local dataset to HF Hub.")
    pp.add_argument("--dst",      required=True)
    pp.add_argument("--hub-repo", required=True)

    args = parser.parse_args()
    {"prep-bank": cmd_prep_bank,
     "run":       cmd_run,
     "viz":       cmd_viz,
     "push":      cmd_push}[args.cmd](args)


if __name__ == "__main__":
    main()
