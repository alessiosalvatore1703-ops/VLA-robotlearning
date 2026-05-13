import argparse

import yaml

from benchmarks.celebrity_recognition.data.loader import load_dataset
from benchmarks.celebrity_recognition.eval.metrics import compute_accuracy
from benchmarks.celebrity_recognition.eval.runner import run
from benchmarks.celebrity_recognition.models import REGISTRY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(REGISTRY))
    p.add_argument("--dataset", required=True, help="Path to dataset.csv")
    p.add_argument("--prompt", default="default", help="Prompt key in configs/prompts.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (for quick runs)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_dir = "benchmarks/celebrity_recognition/configs"
    with open(f"{cfg_dir}/models.yaml") as f:
        model_cfg = yaml.safe_load(f)[args.model]
    with open(f"{cfg_dir}/prompts.yaml") as f:
        prompt_template = yaml.safe_load(f)[args.prompt]

    model = REGISTRY[args.model](**model_cfg)
    print(f"Loading {args.model}...")
    model.load()

    samples = load_dataset(args.dataset)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Running on {len(samples)} samples...")

    output_csv = args.output or f"benchmarks/celebrity_recognition/results/{args.model}.csv"
    results = run(model, samples, prompt_template, output_csv)

    stats = compute_accuracy(results)
    print(f"\n=== {args.model} ===")
    print(f"Accuracy : {stats['accuracy']:.1%}  ({stats['correct']}/{stats['total']})")
    print(f"Unparseable responses: {stats['unparseable']}")
    print(f"Per-position breakdown:")
    for pos, s in stats["per_position"].items():
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {pos:15s}  {acc:.1%}  ({s['correct']}/{s['total']})")
    print(f"\nResults → {output_csv}")


if __name__ == "__main__":
    main()
