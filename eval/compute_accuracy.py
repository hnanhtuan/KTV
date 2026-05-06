import argparse
import json
import re
import string
from collections import defaultdict


OPTION_LETTERS = string.ascii_uppercase


def load_jsonl(path):
    """Load one JSON object per line from an inference output file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return records


def normalize_text(text):
    """Normalize free-form answer text for simple candidate matching."""
    text = text.lower().strip()
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = re.sub(r"^[a-f][\).:\-\s]+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def prediction_to_index(prediction, candidates):
    """Map a model prediction string to a candidate index."""
    if not isinstance(prediction, str):
        return None

    prediction = prediction.strip()

    # Prefer explicit multiple-choice markers such as "C)", "C:", "(C)", or "Answer: C".
    letter_match = re.search(
        r"(?:^|\b|[\(\[])([A-Fa-f])(?:[\)\].:\-]|\b)",
        prediction,
    )
    if letter_match:
        index = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= index < len(candidates):
            return index

    # Fall back to matching normalized candidate text inside the prediction.
    normalized_prediction = normalize_text(prediction)
    normalized_candidates = [normalize_text(candidate) for candidate in candidates]
    for index, candidate in enumerate(normalized_candidates):
        if candidate and candidate in normalized_prediction:
            return index

    return None


def compute_accuracy(records):
    """Compute overall and per-task multiple-choice accuracy."""
    totals = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})

    for record in records:
        task_name = record.get("task_name", "unknown")
        candidates = record.get("candidates", [])
        answer_index = record.get("answer_number")
        predicted_index = prediction_to_index(record.get("pred"), candidates)

        totals[task_name]["total"] += 1
        totals["overall"]["total"] += 1

        if predicted_index is None:
            totals[task_name]["unparsed"] += 1
            totals["overall"]["unparsed"] += 1
            continue

        if predicted_index == answer_index:
            totals[task_name]["correct"] += 1
            totals["overall"]["correct"] += 1

    return totals


def print_accuracy(totals):
    """Print accuracy summary in a compact, readable format."""
    for task_name in sorted(totals):
        stats = totals[task_name]
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total else 0.0
        print(f"{task_name}:")
        print(f"  correct: {correct}")
        print(f"  total: {total}")
        print(f"  unparsed: {stats['unparsed']}")
        print(f"  accuracy: {accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred_path",
        help="Path to an inference JSONL file, for example outputs/nextqa_test_cpu.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_jsonl(args.pred_path)
    totals = compute_accuracy(records)
    print_accuracy(totals)


if __name__ == "__main__":
    main()
