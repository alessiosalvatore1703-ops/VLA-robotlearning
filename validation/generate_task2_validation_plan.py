#!/usr/bin/env python3
"""Generate real-robot validation sheets for task2 prompt/config evaluation.

The generated CSV is meant to be used while running policy rollouts on the
real setup. Each row tells the operator which bowl configuration to set, which
prompt to send to the policy, and what the expected target bowl is.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


COLORS = ("red", "green", "blue")
POSITIONS = ("left", "center", "right")


@dataclass(frozen=True)
class BowlConfig:
    name: str
    left: str
    center: str
    right: str

    @property
    def order(self) -> tuple[str, str, str]:
        return (self.left, self.center, self.right)


CONFIGS = (
    BowlConfig("config1-green-red-blue", "green", "red", "blue"),
    BowlConfig("config2-red-green-blue", "red", "green", "blue"),
    BowlConfig("config3-red-blue-green", "red", "blue", "green"),
    BowlConfig("config4-blue-red-green", "blue", "red", "green"),
    BowlConfig("config5-green-blue-red", "green", "blue", "red"),
    BowlConfig("config6-blue-green-red", "blue", "green", "red"),
)


DIRECT_COLOR_TEMPLATES = (
    "Put the banana in the {color} bowl.",
    "Put the banana in the {color} colored bowl.",
    "Place the banana in the {color} bowl.",
    "Move the banana to the {color} colored bowl.",
)


NEGATION_PROMPTS = {
    "red": "Put the banana into the bowl that is not green and not blue.",
    "green": "Put the banana into the bowl that is not red and not blue.",
    "blue": "Put the banana into the bowl that is not red and not green.",
}


ORDINAL_PROMPTS = {
    "left": (
        "Put the banana into the 1st bowl from the left from the robot perspective.",
        "Put the banana into the left bowl from the robot perspective.",
        "Put the banana into the leftmost bowl from the robot perspective.",
    ),
    "center": (
        "Put the banana into the 2nd bowl from the left from the robot perspective.",
        "Put the banana into the center bowl from the robot perspective.",
        "Put the banana into the middle bowl from the robot perspective.",
    ),
    "right": (
        "Put the banana into the 3rd bowl from the left from the robot perspective.",
        "Put the banana into the right bowl from the robot perspective.",
        "Put the banana into the rightmost bowl from the robot perspective.",
    ),
}


BETWEEN_PROMPTS = {
    frozenset(("blue", "green")): "Put the banana into the bowl between the blue bowl and the green bowl.",
    frozenset(("blue", "red")): "Put the banana into the bowl between the blue bowl and the red bowl.",
    frozenset(("red", "green")): "Put the banana into the bowl between the red bowl and the green bowl.",
}


BASE_COLUMNS = (
    "trial_id",
    "repeat_id",
    "config_name",
    "left_bowl",
    "center_bowl",
    "right_bowl",
    "target_color",
    "target_position",
    "prompt_family",
    "prompt_variant",
    "prompt",
)

SCORING_COLUMNS = (
    "policy_repo_id",
    "checkpoint",
    "date",
    "operator",
    "grasp_success",
    "correct_bowl_motion",
    "release_success",
    "task_success",
    "actual_bowl_color",
    "actual_bowl_position",
    "failure_mode",
    "notes",
)


def base_row(
    cfg: BowlConfig,
    target_idx: int,
    prompt_family: str,
    prompt_variant: str,
    prompt: str,
) -> dict[str, str]:
    return {
        "config_name": cfg.name,
        "left_bowl": cfg.left,
        "center_bowl": cfg.center,
        "right_bowl": cfg.right,
        "target_color": cfg.order[target_idx],
        "target_position": POSITIONS[target_idx],
        "prompt_family": prompt_family,
        "prompt_variant": prompt_variant,
        "prompt": prompt,
    }


def relative_rows(cfg: BowlConfig, target_idx: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    order = cfg.order

    if target_idx > 0:
        ref_color = order[target_idx - 1]
        rows.append(
            base_row(
                cfg,
                target_idx,
                "relative",
                f"right_of_{ref_color}",
                f"Put the banana into the bowl on the right of the {ref_color} bowl from the robot perspective.",
            )
        )

    if target_idx < 2:
        ref_color = order[target_idx + 1]
        rows.append(
            base_row(
                cfg,
                target_idx,
                "relative",
                f"left_of_{ref_color}",
                f"Put the banana into the bowl on the left of the {ref_color} bowl from the robot perspective.",
            )
        )

    return rows


def between_row(cfg: BowlConfig) -> dict[str, str]:
    edge_colors = frozenset((cfg.left, cfg.right))
    return base_row(cfg, 1, "between", "between_edge_bowls", BETWEEN_PROMPTS[edge_colors])


def full_profile_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for cfg in CONFIGS:
        for target_idx, target_color in enumerate(cfg.order):
            target_position = POSITIONS[target_idx]

            for variant_idx, template in enumerate(DIRECT_COLOR_TEMPLATES, start=1):
                rows.append(
                    base_row(
                        cfg,
                        target_idx,
                        "color",
                        f"color_template_{variant_idx}",
                        template.format(color=target_color),
                    )
                )

            rows.append(
                base_row(
                    cfg,
                    target_idx,
                    "negation",
                    "not_other_two_colors",
                    NEGATION_PROMPTS[target_color],
                )
            )

            for variant_idx, prompt in enumerate(ORDINAL_PROMPTS[target_position], start=1):
                rows.append(
                    base_row(
                        cfg,
                        target_idx,
                        "ordinal",
                        f"ordinal_template_{variant_idx}",
                        prompt,
                    )
                )

            rows.extend(relative_rows(cfg, target_idx))

            if target_idx == 1:
                rows.append(between_row(cfg))

    return rows


def balanced_profile_rows() -> list[dict[str, str]]:
    """Build a smaller balanced plan: 5 prompt tests per target/config.

    For every configuration and target bowl, this covers:
    - direct color prompt
    - one color paraphrase
    - negation prompt
    - one ordinal/position prompt
    - one relational prompt

    Middle-bowl relational prompts rotate between "between", "right of left",
    and "left of right" across configurations so all relation types are covered.
    """

    rows: list[dict[str, str]] = []

    for cfg_idx, cfg in enumerate(CONFIGS):
        for target_idx, target_color in enumerate(cfg.order):
            target_position = POSITIONS[target_idx]

            rows.append(
                base_row(
                    cfg,
                    target_idx,
                    "color",
                    "canonical",
                    DIRECT_COLOR_TEMPLATES[0].format(color=target_color),
                )
            )

            paraphrase_template = DIRECT_COLOR_TEMPLATES[1 + ((cfg_idx + target_idx) % 3)]
            rows.append(
                base_row(
                    cfg,
                    target_idx,
                    "color",
                    "paraphrase",
                    paraphrase_template.format(color=target_color),
                )
            )

            rows.append(
                base_row(
                    cfg,
                    target_idx,
                    "negation",
                    "not_other_two_colors",
                    NEGATION_PROMPTS[target_color],
                )
            )

            ordinal_prompt = ORDINAL_PROMPTS[target_position][(cfg_idx + target_idx) % 3]
            rows.append(
                base_row(
                    cfg,
                    target_idx,
                    "ordinal",
                    "position_paraphrase",
                    ordinal_prompt,
                )
            )

            rel_rows = relative_rows(cfg, target_idx)
            if target_idx == 1:
                middle_options = [between_row(cfg), *rel_rows]
                rows.append(middle_options[cfg_idx % len(middle_options)])
            else:
                rows.append(rel_rows[0])

    return rows


def compact30_profile_rows() -> list[dict[str, str]]:
    """Build a 30-rollout plan for quick real-robot validation.

    The plan has 5 rollouts per bowl configuration. Across all 30 rollouts it
    is exactly balanced by target color, and each prompt slot is balanced by
    target color too:

    - 6 canonical color prompts
    - 6 color paraphrases
    - 6 negation prompts
    - 6 ordinal/position prompts
    - 6 spatial relation prompts
    """

    target_plan = (
        (0, 1, 2, 2, 1),
        (0, 1, 0, 1, 2),
        (2, 0, 0, 1, 2),
        (0, 0, 2, 1, 2),
        (1, 1, 0, 2, 2),
        (2, 1, 0, 1, 0),
    )

    rows: list[dict[str, str]] = []
    for cfg_idx, (cfg, targets) in enumerate(zip(CONFIGS, target_plan)):
        canonical_idx, paraphrase_idx, negation_idx, ordinal_idx, relation_idx = targets

        canonical_color = cfg.order[canonical_idx]
        rows.append(
            base_row(
                cfg,
                canonical_idx,
                "color",
                "canonical",
                DIRECT_COLOR_TEMPLATES[0].format(color=canonical_color),
            )
        )

        paraphrase_color = cfg.order[paraphrase_idx]
        paraphrase_template = DIRECT_COLOR_TEMPLATES[1 + (cfg_idx % 3)]
        rows.append(
            base_row(
                cfg,
                paraphrase_idx,
                "color",
                "paraphrase",
                paraphrase_template.format(color=paraphrase_color),
            )
        )

        negation_color = cfg.order[negation_idx]
        rows.append(
            base_row(
                cfg,
                negation_idx,
                "negation",
                "not_other_two_colors",
                NEGATION_PROMPTS[negation_color],
            )
        )

        ordinal_position = POSITIONS[ordinal_idx]
        rows.append(
            base_row(
                cfg,
                ordinal_idx,
                "ordinal",
                "position_paraphrase",
                ORDINAL_PROMPTS[ordinal_position][cfg_idx % 3],
            )
        )

        if relation_idx == 1:
            rows.append(between_row(cfg))
        else:
            rows.append(relative_rows(cfg, relation_idx)[0])

    return rows


def expand_repeats(
    rows: list[dict[str, str]],
    repeats: int,
    policy_repo_id: str,
    checkpoint: str,
) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    trial_id = 1
    for row in rows:
        for repeat_id in range(1, repeats + 1):
            out = {"trial_id": str(trial_id), "repeat_id": str(repeat_id), **row}
            for column in SCORING_COLUMNS:
                out[column] = ""
            out["policy_repo_id"] = policy_repo_id
            out["checkpoint"] = checkpoint
            expanded.append(out)
            trial_id += 1
    return expanded


def print_summary(rows: list[dict[str, str]]) -> None:
    print(f"rows: {len(rows)}")
    print("by configuration:")
    for name, count in sorted(Counter(row["config_name"] for row in rows).items()):
        print(f"  {name}: {count}")
    print("by prompt family:")
    for name, count in sorted(Counter(row["prompt_family"] for row in rows).items()):
        print(f"  {name}: {count}")
    print("by target color:")
    for name, count in sorted(Counter(row["target_color"] for row in rows).items()):
        print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("compact30", "balanced", "full"), default="balanced")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--policy-repo-id", default="")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repeats = args.repeats
    if repeats is None:
        repeats = 1 if args.profile == "compact30" else 3
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")

    if args.profile == "compact30":
        rows = compact30_profile_rows()
    elif args.profile == "balanced":
        rows = balanced_profile_rows()
    else:
        rows = full_profile_rows()
    rows = expand_repeats(rows, repeats, args.policy_repo_id, args.checkpoint)

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)
        for idx, row in enumerate(rows, start=1):
            row["trial_id"] = str(idx)

    output = args.output or Path("validation") / f"task2_{args.profile}_validation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(*BASE_COLUMNS, *SCORING_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
