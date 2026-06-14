"""Summarize temporal-chain first-frame strategy sweep results."""

import argparse
import csv
import os
import re
from pathlib import Path

from ktv.core.tracking import (
    default_shell_artifact_paths,
    track_run,
    write_summary_json,
)


OVERALL_RE = re.compile(r"^overall:\s*$")
ACCURACY_RE = re.compile(r"^\s*accuracy:\s*([0-9.]+)\s*$")
CORRECT_RE = re.compile(r"^\s*correct:\s*(\d+)\s*$")
TOTAL_RE = re.compile(r"^\s*total:\s*(\d+)\s*$")
UNPARSED_RE = re.compile(r"^\s*unparsed:\s*(\d+)\s*$")


def parse_overall_accuracy(path):
    current_section = None
    stats = {"accuracy": None, "correct": None, "total": None, "unparsed": None}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if OVERALL_RE.match(line):
                current_section = "overall"
                continue
            if line and not line.startswith(" ") and line.rstrip().endswith(":"):
                current_section = line.strip()[:-1]
                continue
            if current_section != "overall":
                continue

            accuracy_match = ACCURACY_RE.match(line)
            correct_match = CORRECT_RE.match(line)
            total_match = TOTAL_RE.match(line)
            unparsed_match = UNPARSED_RE.match(line)
            if accuracy_match:
                stats["accuracy"] = float(accuracy_match.group(1))
            elif correct_match:
                stats["correct"] = int(correct_match.group(1))
            elif total_match:
                stats["total"] = int(total_match.group(1))
            elif unparsed_match:
                stats["unparsed"] = int(unparsed_match.group(1))

    if stats["accuracy"] is None:
        raise ValueError(f"No overall accuracy found in {path}")
    return stats


def discover_accuracy_files(root, output_name):
    pattern = f"{output_name}_accuracy.txt"
    return sorted(Path(root).glob(f"**/{pattern}"))


def row_from_accuracy_path(path):
    parts = path.parts
    strategy = ""
    dataset = ""
    keyframes = ""
    for part in parts:
        if part.startswith("strategy_"):
            strategy = part.removeprefix("strategy_")
        elif part.startswith("keyframes_"):
            keyframes = part.removeprefix("keyframes_")
    if len(parts) >= 3:
        for index, part in enumerate(parts[:-1]):
            if part == "outputs" and index + 1 < len(parts):
                dataset = parts[index + 1]
                break
    if not dataset:
        dataset = path.parent.name
    return dataset, strategy, keyframes


def collect_rows(root, output_name):
    rows = []
    for path in discover_accuracy_files(root, output_name):
        dataset, strategy, keyframes = row_from_accuracy_path(path)
        if not strategy:
            continue
        stats = parse_overall_accuracy(path)
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "keyframes": keyframes,
                "accuracy": stats["accuracy"],
                "correct": stats["correct"],
                "total": stats["total"],
                "unparsed": stats["unparsed"],
                "path": str(path),
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["keyframes"], -row["accuracy"], row["strategy"]))
    return rows


def write_csv(rows, output_csv):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "strategy",
                "keyframes",
                "accuracy",
                "correct",
                "total",
                "unparsed",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows):
    if not rows:
        print("No accuracy files found.")
        return

    print("dataset\tkeyframes\tstrategy\taccuracy\tcorrect\ttotal\tunparsed")
    for row in rows:
        print(
            f"{row['dataset']}\t{row['keyframes']}\t{row['strategy']}\t"
            f"{row['accuracy']:.4f}\t{row['correct']}\t{row['total']}\t{row['unparsed']}"
        )


def build_report_summary(rows, output_csv):
    accuracies = [row["accuracy"] for row in rows]
    datasets = sorted({row["dataset"] for row in rows})
    return {
        "row_count": len(rows),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "best_accuracy": max(accuracies) if accuracies else 0.0,
        "average_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else 0.0,
        "output_csv": str(Path(output_csv).resolve()) if output_csv else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root folder to scan for first-frame strategy sweep accuracy files.",
    )
    parser.add_argument(
        "--output-name",
        default="predictions",
        help="Inference output stem used by the sweep script.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for the summary table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.root, args.output_name)
    print_rows(rows)
    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"Saved summary CSV to {args.output_csv}")

    summary_path = None
    if args.output_csv:
        summary_path = Path(args.output_csv).with_name(
            f"{Path(args.output_csv).stem}_summary.json"
        )
        summary = build_report_summary(rows, args.output_csv)
        write_summary_json(summary_path, summary)
    else:
        summary = build_report_summary(rows, "")

    with track_run(
        stage="temporal_chain_first_frame_report",
        script_path=__file__,
        output_dir=args.root,
        extra_tags={"root": str(Path(args.root).resolve())},
    ) as tracker:
        tracker.log_params(
            {
                "root": str(Path(args.root).resolve()),
                "output_name": args.output_name,
                "output_csv": str(Path(args.output_csv).resolve()) if args.output_csv else "",
            }
        )
        tracker.log_metrics(
            {
                "row_count": summary["row_count"],
                "dataset_count": summary["dataset_count"],
                "best_accuracy": summary["best_accuracy"],
                "average_accuracy": summary["average_accuracy"],
            }
        )
        artifacts = [summary_path] if summary_path else []
        if args.output_csv:
            artifacts.append(args.output_csv)
        tracker.log_artifacts(artifacts, artifact_path="reports")
        tracker.log_artifacts(default_shell_artifact_paths(), artifact_path="logs")


if __name__ == "__main__":
    main()
