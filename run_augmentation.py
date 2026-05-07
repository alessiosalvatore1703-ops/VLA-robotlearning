"""CLI entry point for the LeRobot Eval 1 augmentation pipeline.

Usage examples:

    # Basic run (local in, local out)
    python run_augmentation.py \\
        --input  /path/to/eval1_raw \\
        --output /path/to/eval1_augmented

    # Increase variant counts
    python run_augmentation.py \\
        --input  /path/to/eval1_raw \\
        --output /path/to/eval1_augmented \\
        --text-variants 8 --visual-variants 4

    # Push to Hugging Face Hub after augmentation
    python run_augmentation.py \\
        --input  /path/to/eval1_raw \\
        --output /path/to/eval1_augmented \\
        --push-to-hub --hub-repo-id your-username/eval1-augmented
"""

import argparse
import logging
import sys
from pathlib import Path

from augmentation.config import AugmentationConfig
from augmentation.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment a LeRobot v2 Eval 1 dataset.")
    p.add_argument("--input", required=True, type=Path, help="Path to source LeRobot dataset.")
    p.add_argument("--output", required=True, type=Path, help="Path to write augmented dataset.")

    # Variant counts
    p.add_argument("--text-variants", type=int, default=5,
                   help="Number of instruction variants per episode (default: 5).")
    p.add_argument("--visual-variants", type=int, default=3,
                   help="Number of visual variants per text variant (default: 3).")

    # Visual augmentation bounds (optional overrides)
    p.add_argument("--brightness-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="Brightness factor range (default: 0.75 1.25).")
    p.add_argument("--contrast-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--saturation-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--hue-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Hue shift in degrees, e.g. -8 8 (default: -8 8).")
    p.add_argument("--noise-std-max", type=float, default=None,
                   help="Max Gaussian noise std on [0,255] scale (default: 6.0).")

    # Other
    p.add_argument("--no-original", action="store_true",
                   help="Do not include the original episodes in the output dataset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunks-size", type=int, default=1000,
                   help="Episodes per chunk folder (default: 1000).")

    # Hub
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo-id", type=str, default="",
                   help="Hugging Face repo id, e.g. your-username/dataset-name.")
    p.add_argument("--hub-public", action="store_true",
                   help="Make the Hub repo public (default: private).")

    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = AugmentationConfig(
        n_text_variants=args.text_variants,
        n_visual_variants=args.visual_variants,
        include_original=not args.no_original,
        seed=args.seed,
        chunks_size=args.chunks_size,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_private=not args.hub_public,
    )

    if args.brightness_range is not None:
        cfg.brightness_range = tuple(args.brightness_range)
    if args.contrast_range is not None:
        cfg.contrast_range = tuple(args.contrast_range)
    if args.saturation_range is not None:
        cfg.saturation_range = tuple(args.saturation_range)
    if args.hue_range is not None:
        cfg.hue_range = tuple(args.hue_range)
    if args.noise_std_max is not None:
        cfg.noise_std_max = args.noise_std_max

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists() and any(output_path.iterdir()):
        print(
            f"Warning: output path already exists and is not empty: {output_path}\n"
            "Existing files may be overwritten. Continue? [y/N] ",
            end="",
            file=sys.stderr,
        )
        if input("").strip().lower() != "y":
            sys.exit(0)

    output_path.mkdir(parents=True, exist_ok=True)

    run_pipeline(input_path, output_path, cfg)


if __name__ == "__main__":
    main()
