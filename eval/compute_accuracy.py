import argparse
import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ktv.core.tracking import (
    default_shell_artifact_paths,
    log_accuracy_metrics,
    track_run,
    write_summary_json,
)


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

    letter_match = re.search(
        r"(?:^|\b|[\(\[])([A-Fa-f])(?:[\)\].:\-]|\b)",
        prediction,
    )
    if letter_match:
        index = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= index < len(candidates):
            return index

    normalized_prediction = normalize_text(prediction)
    normalized_candidates = [normalize_text(candidate) for candidate in candidates]
    for index, candidate in enumerate(normalized_candidates):
        if candidate and candidate in normalized_prediction:
            return index

    return None


def build_accuracy_summary(records):
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

    overall = totals["overall"]
    tasks = {}
    for task_name, stats in totals.items():
        if task_name == "overall":
            continue
        total = stats["total"]
        tasks[task_name] = {
            "correct": stats["correct"],
            "total": total,
            "unparsed": stats["unparsed"],
            "accuracy": stats["correct"] / total if total else 0.0,
        }

    total = overall["total"]
    return {
        "correct": overall["correct"],
        "total": total,
        "unparsed": overall["unparsed"],
        "accuracy": overall["correct"] / total if total else 0.0,
        "tasks": tasks,
    }


def compute_accuracy(records):
    """Backward-compatible accuracy entrypoint used by other scripts."""
    return build_accuracy_summary(records)


def load_accuracy_summary(pred_path):
    """Return accuracy metrics for one prediction file, or None if absent."""
    if not os.path.exists(pred_path):
        return None
    records = load_jsonl(pred_path)
    return build_accuracy_summary(records)


def print_accuracy(summary):
    """Print accuracy summary in a compact, readable format."""
    ordered_sections = [("overall", {
        "correct": summary["correct"],
        "total": summary["total"],
        "unparsed": summary["unparsed"],
        "accuracy": summary["accuracy"],
    })]
    ordered_sections.extend(sorted(summary["tasks"].items()))

    for task_name, stats in ordered_sections:
        print(f"{task_name}:")
        print(f"  correct: {stats['correct']}")
        print(f"  total: {stats['total']}")
        print(f"  unparsed: {stats['unparsed']}")
        print(f"  accuracy: {stats['accuracy']:.4f}")


def default_json_output_path(pred_path):
    pred_path = Path(pred_path)
    return pred_path.with_name(f"{pred_path.stem}_accuracy.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred_path",
        help="Path to an inference JSONL file, for example outputs/nextqa_test_cpu.json.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON path for structured accuracy metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_jsonl(args.pred_path)
    summary = build_accuracy_summary(records)
    print_accuracy(summary)

    json_output = args.json_output or default_json_output_path(args.pred_path)
    json_output = write_summary_json(json_output, summary)

    output_dir = Path(args.pred_path).resolve().parent
    with track_run(
        stage="qa_evaluation",
        script_path=__file__,
        output_dir=output_dir,
        extra_tags={"pred_path": str(Path(args.pred_path).resolve())},
    ) as tracker:
        tracker.log_params(
            {
                "pred_path": str(Path(args.pred_path).resolve()),
                "json_output": json_output,
            }
        )
        log_accuracy_metrics(tracker, summary)
        tracker.log_artifact(args.pred_path, artifact_path="predictions")
        tracker.log_artifact(json_output, artifact_path="evaluation")
        tracker.log_artifacts(default_shell_artifact_paths(), artifact_path="logs")


if __name__ == "__main__":
    main()
