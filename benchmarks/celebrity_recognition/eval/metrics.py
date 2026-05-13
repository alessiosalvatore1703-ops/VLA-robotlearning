POSITIONS = ["top_left", "top_right", "bottom_left", "bottom_right"]
POSITION_ALIASES = {
    "top left": "top_left", "top-left": "top_left",
    "top right": "top_right", "top-right": "top_right",
    "bottom left": "bottom_left", "bottom-left": "bottom_left",
    "bottom right": "bottom_right", "bottom-right": "bottom_right",
}


def normalize_position(text: str) -> str | None:
    t = text.lower().strip()
    if t in POSITIONS:
        return t
    if t in POSITION_ALIASES:
        return POSITION_ALIASES[t]
    for pos in POSITIONS:
        if pos in t or pos.replace("_", " ") in t or pos.replace("_", "-") in t:
            return pos
    return None


def compute_accuracy(results: list[dict]) -> dict:
    correct = 0
    unparseable = 0
    per_position: dict[str, dict] = {p: {"correct": 0, "total": 0} for p in POSITIONS}

    for r in results:
        pred = normalize_position(r["prediction"])
        label = r["target_position"]
        per_position[label]["total"] += 1

        if pred is None:
            unparseable += 1
        elif pred == label:
            correct += 1
            per_position[label]["correct"] += 1

    total = len(results)
    return {
        "total": total,
        "correct": correct,
        "unparseable": unparseable,
        "accuracy": correct / total if total else 0.0,
        "per_position": per_position,
    }
