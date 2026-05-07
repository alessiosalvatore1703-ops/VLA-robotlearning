"""Template-based instruction augmentation for Eval 1 pick-and-place tasks."""

from __future__ import annotations

import re
import random
from typing import List, Optional

# ---------------------------------------------------------------------------
# Template library — all semantically equivalent to "place {obj} in {color} bowl"
# ---------------------------------------------------------------------------

_TEMPLATES: List[str] = [
    # simple direct
    "place the {obj} into the {color} bowl",
    "put the {obj} in the {color} bowl",
    "move the {obj} to the {color} bowl",
    "drop the {obj} into the {color} bowl",
    "set the {obj} down in the {color} bowl",
    # two-step pick-then-place
    "pick up the {obj} and place it in the {color} bowl",
    "pick up the {obj} and put it in the {color} bowl",
    "grab the {obj} and place it in the {color} bowl",
    "take the {obj} and put it in the {color} bowl",
    "grasp the {obj} and drop it in the {color} bowl",
    "lift the {obj} and place it in the {color} bowl",
    # polite / conversational
    "please place the {obj} into the {color} bowl",
    "please put the {obj} in the {color} bowl",
    "could you place the {obj} in the {color} bowl",
    # with filler words
    "go ahead and place the {obj} into the {color} bowl",
    "now place the {obj} in the {color} bowl",
    "carefully place the {obj} into the {color} bowl",
    # slightly varied phrasing
    "place the {obj} in the {color}-colored bowl",
    "transfer the {obj} to the {color} bowl",
    "deposit the {obj} into the {color} bowl",
]

_KNOWN_COLORS = {
    "red", "blue", "green",
}

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_color(instruction: str) -> Optional[str]:
    instr = instruction.lower()
    for color in _KNOWN_COLORS:
        if re.search(rf"\b{color}\b", instr):
            return color
    return None


def extract_object(instruction: str) -> Optional[str]:
    """Find the first noun after 'the' that isn't a color or 'bowl'."""
    instr = instruction.lower()
    skip = _KNOWN_COLORS | {"bowl", "it"}
    for match in re.finditer(r"\bthe\s+(\w+)", instr):
        word = match.group(1)
        if word not in skip:
            return word
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_variants(
    instruction: str,
    n: int,
    rng: random.Random,
    include_original: bool = False,
) -> List[str]:
    """Return up to n unique instruction variants.

    If the instruction doesn't match the expected pattern, returns the
    original instruction repeated as a single-element list.
    """
    color = extract_color(instruction)
    obj = extract_object(instruction)

    if color is None or obj is None:
        return [instruction] * n

    candidates = [t.format(obj=obj, color=color) for t in _TEMPLATES]

    if include_original and instruction.lower() not in [c.lower() for c in candidates]:
        candidates.insert(0, instruction)

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if len(unique) <= n:
        return unique

    rng.shuffle(unique)
    return unique[:n]
