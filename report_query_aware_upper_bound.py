import argparse
import json
import os
import re
import string
from collections import defaultdict
from pathlib import Path

from ktv.core.tracking import (
    default_shell_artifact_paths,
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


def compute_accuracy(records):
    """Compute overall multiple-choice accuracy."""
    correct = 0
    total = 0
    unparsed = 0
    task_counts = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})

    for record in records:
        task_name = record.get("task_name", "unknown")
        candidates = record.get("candidates", [])
        answer_index = record.get("answer_number")
        predicted_index = prediction_to_index(record.get("pred"), candidates)

        total += 1
        task_counts[task_name]["total"] += 1

        if predicted_index is None:
            unparsed += 1
            task_counts[task_name]["unparsed"] += 1
            continue

        if predicted_index == answer_index:
            correct += 1
            task_counts[task_name]["correct"] += 1

    accuracy = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "unparsed": unparsed,
        "accuracy": accuracy,
        "tasks": task_counts,
    }


def load_accuracy_summary(pred_path):
    """Return accuracy metrics for one prediction file, or None if absent."""
    if not os.path.exists(pred_path):
        return None
    records = load_jsonl(pred_path)
    return compute_accuracy(records)


def build_prediction_path(dataset, variant_dir, query_mode, prune_mode, tokens_num):
    """Build the expected prediction file path for one experiment run."""
    if query_mode is None:
        variant_root = os.path.join("outputs", dataset, variant_dir)
    else:
        variant_root = os.path.join("outputs", dataset, variant_dir, query_mode, prune_mode)
    return os.path.join(variant_root, f"predictions_tokens{tokens_num}.json")


def collect_rows(datasets, query_modes, prune_modes, token_list, dense_pool_sizes):
    """Collect comparable rows for all requested datasets and variants."""
    rows_by_dataset = defaultdict(list)

    for dataset in datasets:
        for prune_mode in prune_modes:
            reference_variant = f"ktv_full_{prune_mode}"
            for tokens_num in token_list:
                reference_path = build_prediction_path(
                    dataset,
                    reference_variant,
                    None,
                    prune_mode,
                    tokens_num,
                )
                reference_metrics = load_accuracy_summary(reference_path)
                reference_accuracy = None if reference_metrics is None else reference_metrics["accuracy"]

                variants = [("query_aware_12_candidate", None)]
                variants.extend(
                    (f"query_aware_dense_uniform_f{pool_size}", pool_size)
                    for pool_size in dense_pool_sizes
                )
                for variant_dir, pool_size in variants:
                    for query_mode in query_modes:
                        pred_path = build_prediction_path(
                            dataset,
                            variant_dir,
                            query_mode,
                            prune_mode,
                            tokens_num,
                        )
                        metrics = load_accuracy_summary(pred_path)
                        if metrics is None:
                            continue
                        delta = (
                            None
                            if reference_accuracy is None
                            else metrics["accuracy"] - reference_accuracy
                        )
                        rows_by_dataset[dataset].append(
                            {
                                "variant": variant_dir,
                                "dense_candidate_pool_size": pool_size,
                                "query_mode": query_mode,
                                "prune_mode": prune_mode,
                                "tokens_num": tokens_num,
                                "correct": metrics["correct"],
                                "total": metrics["total"],
                                "unparsed": metrics["unparsed"],
                                "accuracy": metrics["accuracy"],
                                "delta_vs_full_ktv": delta,
                                "pred_path": pred_path,
                                "reference_variant": reference_variant,
                                "reference_accuracy": reference_accuracy,
                            }
                        )
    return rows_by_dataset


def render_markdown(rows_by_dataset):
    """Render one markdown table per dataset."""
    lines = ["# Query-Aware Proxy Upper-Bound Report", ""]
    for dataset in sorted(rows_by_dataset):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(
            "| variant | query_mode | prune_mode | tokens | accuracy | delta_vs_full_ktv | correct/total |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")

        sorted_rows = sorted(
            rows_by_dataset[dataset],
            key=lambda row: (
                row["prune_mode"],
                row["tokens_num"],
                row["variant"],
                row["query_mode"],
            ),
        )
        for row in sorted_rows:
            accuracy = f'{row["accuracy"]:.4f}'
            delta = (
                "n/a"
                if row["delta_vs_full_ktv"] is None
                else f'{row["delta_vs_full_ktv"]:+.4f}'
            )
            lines.append(
                "| "
                f'{row["variant"]} | {row["query_mode"]} | {row["prune_mode"]} | '
                f'{row["tokens_num"]} | {accuracy} | {delta} | '
                f'{row["correct"]}/{row["total"]} |'
            )
        lines.append("")
    return "\n".join(lines)


def build_report_summary(rows_by_dataset, output_path, summary_path):
    rows = [row for dataset_rows in rows_by_dataset.values() for row in dataset_rows]
    accuracies = [row["accuracy"] for row in rows]
    deltas = [row["delta_vs_full_ktv"] for row in rows if row["delta_vs_full_ktv"] is not None]
    return {
        "output_path": str(Path(output_path).resolve()),
        "summary_path": str(Path(summary_path).resolve()),
        "dataset_count": len(rows_by_dataset),
        "row_count": len(rows),
        "best_accuracy": max(accuracies) if accuracies else 0.0,
        "average_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else 0.0,
        "best_delta_vs_full_ktv": max(deltas) if deltas else 0.0,
        "worst_delta_vs_full_ktv": min(deltas) if deltas else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nextqa", "videomme", "intentqa", "egoschema"],
    )
    parser.add_argument(
        "--query-modes",
        nargs="+",
        default=["question_only", "question_plus_options"],
    )
    parser.add_argument(
        "--prune-modes",
        nargs="+",
        default=["cls_new_token_sim", "uniform_token"],
    )
    parser.add_argument(
        "--token-list",
        nargs="+",
        type=int,
        default=[504, 936, 1872],
    )
    parser.add_argument(
        "--dense-pool-sizes",
        nargs="+",
        type=int,
        default=[12, 16, 24, 32, 48],
    )
    parser.add_argument(
        "--output-path",
        default="outputs/query_aware_upper_bound_report.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows_by_dataset = collect_rows(
        args.datasets,
        args.query_modes,
        args.prune_modes,
        args.token_list,
        args.dense_pool_sizes,
    )
    markdown = render_markdown(rows_by_dataset)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(markdown)

    summary_path = Path(args.output_path).with_name(
        f"{Path(args.output_path).stem}_summary.json"
    )
    summary = build_report_summary(rows_by_dataset, args.output_path, summary_path)
    write_summary_json(summary_path, summary)

    with track_run(
        stage="query_aware_upper_bound_report",
        script_path=__file__,
        output_dir=output_dir or ".",
        extra_tags={"report_path": str(Path(args.output_path).resolve())},
    ) as tracker:
        tracker.log_params(
            {
                "datasets": json.dumps(args.datasets),
                "query_modes": json.dumps(args.query_modes),
                "prune_modes": json.dumps(args.prune_modes),
                "token_list": json.dumps(args.token_list),
                "dense_pool_sizes": json.dumps(args.dense_pool_sizes),
                "output_path": str(Path(args.output_path).resolve()),
            }
        )
        tracker.log_metrics(
            {
                "dataset_count": summary["dataset_count"],
                "row_count": summary["row_count"],
                "best_accuracy": summary["best_accuracy"],
                "average_accuracy": summary["average_accuracy"],
                "best_delta_vs_full_ktv": summary["best_delta_vs_full_ktv"],
                "worst_delta_vs_full_ktv": summary["worst_delta_vs_full_ktv"],
            }
        )
        tracker.log_artifacts(
            [args.output_path, summary_path],
            artifact_path="reports",
        )
        tracker.log_artifacts(default_shell_artifact_paths(), artifact_path="logs")


if __name__ == "__main__":
    main()
