#!/usr/bin/env python3
"""Backfill historical KTV output artifacts into MLflow runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from eval.compute_accuracy import load_accuracy_summary
from ktv.core.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    configure_mlflow_environment,
    flatten_params,
    repo_tracking_uri,
    sanitize_metric_key,
)

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None  # type: ignore[assignment]


BACKFILL_TAG = "ktv_backfill"
BACKFILL_SOURCE_SHA_TAG = "ktv_backfill_source_sha1"
BACKFILL_SOURCE_PATH_TAG = "ktv_backfill_source_path"
BACKFILL_SOURCE_TYPE_TAG = "ktv_backfill_source_type"
BACKFILL_IMPORTED_AT_TAG = "ktv_backfill_imported_at"
RUN_NAME_TAG = "mlflow.runName"
TIMESTAMP_RE = re.compile(r"(20\d{6}T\d{6}Z)")
ACCURACY_SUFFIX_RE = re.compile(r"_accuracy(?:\.[^.]+)?$")
SUMMARY_SUFFIX_RE = re.compile(r"(?:_summary|_accuracy)\.json$")
VALID_TYPES = ("directory", "prediction", "keyframes", "report")


@dataclass
class CandidateRun:
    source_path: Path
    source_type: str
    stage: str
    workflow: str
    dataset: str | None
    run_name: str
    output_dir: Path
    script_tag: str
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    file_artifacts: list[tuple[Path, str]] = field(default_factory=list)
    dir_artifacts: list[tuple[Path, str]] = field(default_factory=list)
    inline_artifacts: list[tuple[str, str, str]] = field(default_factory=list)
    start_time_ms: int = 0
    end_time_ms: int = 0

    @property
    def fingerprint(self) -> str:
        return hashlib.sha1(str(self.source_path.resolve()).encode("utf-8")).hexdigest()

    @property
    def source_relpath(self) -> str:
        try:
            return str(self.source_path.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(self.source_path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical KTV outputs into MLflow runs.",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root outputs folder to scan for historical runs.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI") or repo_tracking_uri(REPO_ROOT),
        help="MLflow tracking URI. Defaults to the repo-local mlruns file store.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name for imported runs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset filter. May be passed multiple times.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=VALID_TYPES,
        default=list(VALID_TYPES),
        help="Candidate types to include in the backfill.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of discovered runs to process after filtering.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Import even if a run with the same historical source fingerprint already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover historical runs and print what would be imported without creating MLflow runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per discovered or imported run.",
    )
    return parser.parse_args()


def should_skip_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts) or ".ipynb_checkpoints" in path.parts


def parse_shell_value(raw_value: str) -> str:
    stripped = raw_value.strip()
    if not stripped:
        return ""
    try:
        parts = shlex.split(stripped, posix=True)
    except ValueError:
        return stripped.strip("'\"")
    if not parts:
        return ""
    return " ".join(parts)


def parse_env_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = parse_shell_value(value)
    return parsed


def load_resolved_configs(hydra_dir: Path) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    if not hydra_dir.exists() or not hydra_dir.is_dir():
        return configs
    for path in sorted(hydra_dir.glob("*.yaml")):
        if OmegaConf is not None:
            loaded = OmegaConf.load(path)
            configs[path.stem] = OmegaConf.to_container(loaded, resolve=True)
        else:  # pragma: no cover
            configs[path.stem] = path.read_text(encoding="utf-8")
    return configs


def relative_dataset(path: Path, outputs_root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(outputs_root.resolve())
    except ValueError:
        return None
    if not relative.parts:
        return None
    return relative.parts[0]


def infer_workflow_from_stem(stem: str, *, source_type: str | None = None) -> str:
    lower_stem = stem.lower()
    if "temporal_chain" in lower_stem:
        return "temporal_chain"
    if "upper_bound" in lower_stem or "query_aware" in lower_stem:
        return "query_aware_upper_bound"
    if "baseline" in lower_stem:
        return "baseline"
    if source_type == "keyframes" or "keyframe" in lower_stem:
        return "keyframe_ranking"
    if "ktv" in lower_stem:
        return "ktv"
    if source_type == "prediction":
        return "qa_inference"
    if source_type == "report":
        return "report"
    return "legacy"


def infer_workflow_from_path(path: Path, outputs_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(outputs_root.resolve())
    except ValueError:
        relative = path

    if path.is_dir() and len(relative.parts) >= 2 and relative.parts[1] and not relative.parts[1].startswith("20"):
        return relative.parts[1]
    if path.is_file() and len(relative.parts) >= 3 and relative.parts[1] and not relative.parts[1].startswith("20"):
        return relative.parts[1]
    stem = path.stem if path.is_file() else path.name
    return infer_workflow_from_stem(stem)


def infer_script_tag(stage: str, workflow: str, stem: str = "") -> str:
    lower_stem = stem.lower()
    if stage == "report":
        if "upper_bound" in lower_stem or "query_aware" in lower_stem:
            return "report_query_aware_upper_bound.py"
        if "temporal_chain" in workflow or "sweep" in lower_stem:
            return "report_temporal_chain_first_frame_sweep.py"
        return "historical_report"
    if stage == "keyframe_ranking":
        if "temporal_chain" in workflow or "temporal_chain" in lower_stem:
            return "temporal_chain_rank_keyframes.py"
        if "query_aware" in workflow or "upper_bound" in workflow:
            return "query_aware_select_keyframes.py"
        return "cluster_and_rank_keyframes.py"
    if stage == "workflow":
        return "historical_workflow"
    return "run_inference_multiple_choice_qa.py"


def looks_like_prediction_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _ in range(5):
                line = handle.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    return False
                keys = set(payload.keys())
                return {"pred", "candidates"}.issubset(keys) and (
                    "answer_number" in keys or "answer" in keys
                )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return False


def looks_like_keyframe_file(path: Path) -> bool:
    if "keyframe" not in path.stem.lower():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            prefix = handle.read(256)
    except OSError:
        return False
    return prefix.lstrip().startswith("{")


def load_accuracy_txt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = float(line)
        except ValueError:
            return None
        accuracy = value / 100.0 if value > 1.0 else value
        return {
            "accuracy": accuracy,
            "accuracy_percent": value if value > 1.0 else value * 100.0,
            "tasks": {},
        }
    return None


def accuracy_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    value = summary.get("accuracy")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        metrics["overall.accuracy"] = float(value)
    accuracy_percent = summary.get("accuracy_percent")
    if isinstance(accuracy_percent, (int, float)) and math.isfinite(float(accuracy_percent)):
        metrics["overall.accuracy_percent"] = float(accuracy_percent)
    return metrics


def metrics_from_csv_report(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            rows.append({str(key): value for key, value in row.items() if key is not None})

    summary: dict[str, Any] = {
        "row_count": len(rows),
        "columns": sorted(rows[0].keys()) if rows else [],
    }
    metrics: dict[str, float] = {"report.row_count": float(len(rows))}

    numeric_by_column: dict[str, list[float]] = {}
    for row in rows:
        for key, raw_value in row.items():
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            numeric_by_column.setdefault(key, []).append(value)

    for key, values in numeric_by_column.items():
        if not values:
            continue
        safe_key = sanitize_metric_key(key)
        if "accuracy" in key.lower():
            metrics[f"report.{safe_key}.best"] = max(values)
            metrics[f"report.{safe_key}.average"] = sum(values) / len(values)
            summary[f"best_{key}"] = max(values)
            summary[f"average_{key}"] = sum(values) / len(values)

    return metrics, summary


def resolve_repo_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def metric_candidate(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for child in sorted(path.rglob("*")):
        if child.is_file() and not should_skip_path(child):
            yield child


def iso_to_epoch_ms(raw_timestamp: str) -> int | None:
    if not raw_timestamp:
        return None
    normalized = raw_timestamp.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except ValueError:
        return None


def path_timestamp_ms(path: Path) -> int | None:
    match = TIMESTAMP_RE.search(str(path))
    if not match:
        return None
    return iso_to_epoch_ms(match.group(1))


def compute_time_range(primary_path: Path, artifacts: Iterable[Path], env_params: dict[str, str] | None = None) -> tuple[int, int]:
    env_params = env_params or {}
    candidates: list[int] = []
    env_timestamp = iso_to_epoch_ms(env_params.get("RUN_TIMESTAMP_UTC", ""))
    if env_timestamp is not None:
        candidates.append(env_timestamp)
    path_timestamp = path_timestamp_ms(primary_path)
    if path_timestamp is not None:
        candidates.append(path_timestamp)

    file_times: list[int] = []
    seen: set[Path] = set()
    for artifact in artifacts:
        for file_path in iter_files(artifact):
            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                file_times.append(int(file_path.stat().st_mtime * 1000))
            except OSError:
                continue

    if file_times:
        start_time = min(candidates + file_times) if candidates else min(file_times)
        end_time = max(candidates + file_times) if candidates else max(file_times)
        return start_time, max(start_time, end_time)

    fallback = int(primary_path.stat().st_mtime * 1000)
    if candidates:
        fallback = min(candidates + [fallback])
        return fallback, max(candidates + [fallback])
    return fallback, fallback


def collect_directory_roots(outputs_root: Path) -> list[Path]:
    roots: set[Path] = set()
    for current_root, dirnames, filenames in os.walk(outputs_root):
        current_path = Path(current_root).resolve()
        if should_skip_path(current_path):
            dirnames[:] = []
            continue

        if "chosen_parameters.env" in filenames or "hydra_resolved_configs" in dirnames:
            roots.add(current_path)
            dirnames[:] = []
            continue

        pruned_dirnames: list[str] = []
        for dirname in dirnames:
            child_dir = (current_path / dirname).resolve()
            if should_skip_path(child_dir):
                continue
            pruned_dirnames.append(dirname)
        dirnames[:] = pruned_dirnames
    return sorted(roots)


def pair_accuracy_sidecars(prediction_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for sibling in sorted(prediction_path.parent.iterdir()):
        if not sibling.is_file():
            continue
        if sibling.name == prediction_path.name:
            continue
        if sibling.stem.startswith(prediction_path.stem) and (
            "accuracy" in sibling.name or sibling.name.endswith("summary.json")
        ):
            candidates.append(sibling)
    return candidates


def add_file_artifact(target: list[tuple[Path, str]], path: Path, artifact_path: str) -> None:
    if path.exists() and path.is_file():
        target.append((path.resolve(), artifact_path))


def add_dir_artifact(target: list[tuple[Path, str]], path: Path, artifact_path: str) -> None:
    if path.exists() and path.is_dir():
        target.append((path.resolve(), artifact_path))


def build_manifest(candidate: CandidateRun) -> dict[str, Any]:
    return {
        "source_path": str(candidate.source_path.resolve()),
        "source_relpath": candidate.source_relpath,
        "source_type": candidate.source_type,
        "fingerprint": candidate.fingerprint,
        "dataset": candidate.dataset,
        "workflow": candidate.workflow,
        "stage": candidate.stage,
        "run_name": candidate.run_name,
        "output_dir": str(candidate.output_dir.resolve()),
        "script": candidate.script_tag,
        "start_time_ms": candidate.start_time_ms,
        "end_time_ms": candidate.end_time_ms,
        "params_count": len(candidate.params),
        "metrics_count": len(candidate.metrics),
        "file_artifacts": [
            {"path": str(path), "artifact_path": artifact_path}
            for path, artifact_path in candidate.file_artifacts
        ],
        "dir_artifacts": [
            {"path": str(path), "artifact_path": artifact_path}
            for path, artifact_path in candidate.dir_artifacts
        ],
    }


def build_directory_candidate(run_dir: Path, outputs_root: Path) -> CandidateRun | None:
    chosen_env = run_dir / "chosen_parameters.env"
    hydra_dir = run_dir / "hydra_resolved_configs"
    logs_dir = run_dir / "logs"
    env_params = parse_env_file(chosen_env)
    resolved_configs = load_resolved_configs(hydra_dir)

    prediction_files = sorted(
        path
        for path in run_dir.glob("predictions*.json*")
        if path.is_file() and "accuracy" not in path.name and not path.name.endswith("summary.json")
    )
    keyframe_files = sorted(
        path for path in run_dir.glob("*keyframe*.json") if path.is_file()
    )
    report_files = sorted(
        path for path in run_dir.iterdir() if path.is_file() and path.suffix in {".csv", ".md"}
    ) if run_dir.exists() else []

    stage = "workflow"
    if not prediction_files and keyframe_files:
        stage = "keyframe_ranking"
    elif not prediction_files and report_files:
        stage = "report"

    dataset = env_params.get("DATASET") or env_params.get("INFERENCE_DATASET") or relative_dataset(run_dir, outputs_root)
    workflow = infer_workflow_from_path(run_dir, outputs_root)
    script_tag = infer_script_tag(stage, workflow, run_dir.name)
    run_name = env_params.get("UNIQUE_EXP_ID") or run_dir.name
    if dataset and "/" not in run_name:
        run_name = f"{dataset}/{run_name}"

    output_dir = resolve_repo_path(env_params.get("OUTPUT_DIR")) or run_dir.resolve()
    candidate = CandidateRun(
        source_path=run_dir.resolve(),
        source_type="directory",
        stage=stage,
        workflow=workflow,
        dataset=dataset,
        run_name=run_name,
        output_dir=output_dir,
        script_tag=script_tag,
    )

    if env_params:
        for key, value in env_params.items():
            candidate.params[f"shell.{key.lower()}"] = value

    for config_name, config_payload in resolved_configs.items():
        candidate.params.update(
            flatten_params({config_name: config_payload}, exclude_top_level=())
        )

    add_file_artifact(candidate.file_artifacts, chosen_env, "config")
    add_file_artifact(candidate.file_artifacts, run_dir / "latest_log_path.txt", "logs")
    add_dir_artifact(candidate.dir_artifacts, hydra_dir, "config")
    add_dir_artifact(candidate.dir_artifacts, logs_dir, "logs")

    for path in prediction_files:
        add_file_artifact(candidate.file_artifacts, path, "predictions")
        for sidecar in pair_accuracy_sidecars(path):
            artifact_group = "evaluation" if "accuracy" in sidecar.name or sidecar.name.endswith("summary.json") else "predictions"
            add_file_artifact(candidate.file_artifacts, sidecar, artifact_group)

    for path in keyframe_files:
        add_file_artifact(candidate.file_artifacts, path, "keyframes")

    for path in report_files:
        add_file_artifact(candidate.file_artifacts, path, "reports")

    summary_payload: dict[str, Any] | None = None
    if prediction_files:
        primary_prediction = prediction_files[0]
        accuracy_json = primary_prediction.with_name(f"{primary_prediction.stem}_accuracy.json")
        accuracy_txt = primary_prediction.with_name(f"{primary_prediction.stem}_accuracy.txt")
        summary_payload = None
        if accuracy_json.exists():
            summary_payload = json.loads(accuracy_json.read_text(encoding="utf-8"))
        if summary_payload is None:
            summary_payload = load_accuracy_txt(accuracy_txt)
        if summary_payload is None:
            summary_payload = load_accuracy_summary(str(primary_prediction))
        if summary_payload:
            candidate.metrics.update(accuracy_metrics(summary_payload))
            candidate.inline_artifacts.append(
                (
                    f"{primary_prediction.stem}_backfilled_accuracy.json",
                    json.dumps(summary_payload, ensure_ascii=False, indent=2),
                    "evaluation",
                )
            )
    elif stage == "report" and report_files:
        metrics, report_summary = metrics_from_csv_report(report_files[0]) if report_files[0].suffix == ".csv" else ({}, {"path": str(report_files[0])})
        candidate.metrics.update(metrics)
        candidate.inline_artifacts.append(
            (
                f"{report_files[0].stem}_backfilled_summary.json",
                json.dumps(report_summary, ensure_ascii=False, indent=2),
                "reports",
            )
        )

    artifact_roots = [
        path for path, _ in candidate.file_artifacts
    ] + [path for path, _ in candidate.dir_artifacts]
    candidate.start_time_ms, candidate.end_time_ms = compute_time_range(
        run_dir,
        artifact_roots or [run_dir],
        env_params,
    )
    candidate.tags = {
        "dataset": dataset or "unknown",
        "workflow": workflow,
        "stage": stage,
        "script": script_tag,
        "output_dir": str(output_dir.resolve()),
    }
    candidate.inline_artifacts.append(
        (
            "backfill_manifest.json",
            json.dumps(build_manifest(candidate), ensure_ascii=False, indent=2),
            "metadata",
        )
    )
    return candidate


def build_prediction_candidate(prediction_path: Path, outputs_root: Path) -> CandidateRun | None:
    dataset = relative_dataset(prediction_path, outputs_root)
    workflow = infer_workflow_from_path(prediction_path, outputs_root)
    if workflow in {prediction_path.name, prediction_path.stem} or workflow == "legacy":
        workflow = infer_workflow_from_stem(prediction_path.stem, source_type="prediction")
    run_name = f"{dataset}/{prediction_path.stem}" if dataset else prediction_path.stem
    candidate = CandidateRun(
        source_path=prediction_path.resolve(),
        source_type="prediction",
        stage="qa_inference",
        workflow=workflow,
        dataset=dataset,
        run_name=run_name,
        output_dir=prediction_path.parent.resolve(),
        script_tag=infer_script_tag("qa_inference", workflow, prediction_path.stem),
    )

    add_file_artifact(candidate.file_artifacts, prediction_path, "predictions")
    for sidecar in pair_accuracy_sidecars(prediction_path):
        artifact_group = "evaluation" if "accuracy" in sidecar.name or sidecar.name.endswith("summary.json") else "predictions"
        add_file_artifact(candidate.file_artifacts, sidecar, artifact_group)

    accuracy_json = prediction_path.with_name(f"{prediction_path.stem}_accuracy.json")
    accuracy_txt = prediction_path.with_name(f"{prediction_path.stem}_accuracy.txt")
    summary_payload = None
    if accuracy_json.exists():
        summary_payload = json.loads(accuracy_json.read_text(encoding="utf-8"))
    if summary_payload is None:
        summary_payload = load_accuracy_txt(accuracy_txt)
    if summary_payload is None:
        summary_payload = load_accuracy_summary(str(prediction_path))
    if summary_payload:
        candidate.metrics.update(accuracy_metrics(summary_payload))
        candidate.inline_artifacts.append(
            (
                f"{prediction_path.stem}_backfilled_accuracy.json",
                json.dumps(summary_payload, ensure_ascii=False, indent=2),
                "evaluation",
            )
        )

    candidate.start_time_ms, candidate.end_time_ms = compute_time_range(
        prediction_path,
        [path for path, _ in candidate.file_artifacts] or [prediction_path],
    )
    candidate.tags = {
        "dataset": dataset or "unknown",
        "workflow": workflow,
        "stage": candidate.stage,
        "script": candidate.script_tag,
        "output_dir": str(candidate.output_dir.resolve()),
    }
    candidate.inline_artifacts.append(
        (
            "backfill_manifest.json",
            json.dumps(build_manifest(candidate), ensure_ascii=False, indent=2),
            "metadata",
        )
    )
    return candidate


def build_keyframe_candidate(keyframe_path: Path, outputs_root: Path) -> CandidateRun:
    dataset = relative_dataset(keyframe_path, outputs_root)
    workflow = infer_workflow_from_path(keyframe_path, outputs_root)
    if workflow in {keyframe_path.name, keyframe_path.stem} or workflow == "legacy":
        workflow = infer_workflow_from_stem(keyframe_path.stem, source_type="keyframes")
    run_name = f"{dataset}/{keyframe_path.stem}" if dataset else keyframe_path.stem
    candidate = CandidateRun(
        source_path=keyframe_path.resolve(),
        source_type="keyframes",
        stage="keyframe_ranking",
        workflow=workflow,
        dataset=dataset,
        run_name=run_name,
        output_dir=keyframe_path.parent.resolve(),
        script_tag=infer_script_tag("keyframe_ranking", workflow, keyframe_path.stem),
    )
    add_file_artifact(candidate.file_artifacts, keyframe_path, "keyframes")
    candidate.start_time_ms, candidate.end_time_ms = compute_time_range(
        keyframe_path,
        [keyframe_path],
    )
    candidate.tags = {
        "dataset": dataset or "unknown",
        "workflow": workflow,
        "stage": candidate.stage,
        "script": candidate.script_tag,
        "output_dir": str(candidate.output_dir.resolve()),
    }
    candidate.inline_artifacts.append(
        (
            "backfill_manifest.json",
            json.dumps(build_manifest(candidate), ensure_ascii=False, indent=2),
            "metadata",
        )
    )
    return candidate


def build_report_candidate(report_path: Path, outputs_root: Path) -> CandidateRun:
    dataset = relative_dataset(report_path, outputs_root)
    workflow = infer_workflow_from_path(report_path, outputs_root)
    if workflow in {report_path.name, report_path.stem} or workflow == "legacy":
        workflow = infer_workflow_from_stem(report_path.stem, source_type="report")
    run_name = f"{dataset}/{report_path.stem}" if dataset else report_path.stem
    candidate = CandidateRun(
        source_path=report_path.resolve(),
        source_type="report",
        stage="report",
        workflow=workflow,
        dataset=dataset,
        run_name=run_name,
        output_dir=report_path.parent.resolve(),
        script_tag=infer_script_tag("report", workflow, report_path.stem),
    )
    add_file_artifact(candidate.file_artifacts, report_path, "reports")
    if report_path.suffix == ".csv":
        metrics, report_summary = metrics_from_csv_report(report_path)
        candidate.metrics.update(metrics)
        candidate.inline_artifacts.append(
            (
                f"{report_path.stem}_backfilled_summary.json",
                json.dumps(report_summary, ensure_ascii=False, indent=2),
                "reports",
            )
        )
    candidate.start_time_ms, candidate.end_time_ms = compute_time_range(
        report_path,
        [report_path],
    )
    candidate.tags = {
        "dataset": dataset or "unknown",
        "workflow": workflow,
        "stage": candidate.stage,
        "script": candidate.script_tag,
        "output_dir": str(candidate.output_dir.resolve()),
    }
    candidate.inline_artifacts.append(
        (
            "backfill_manifest.json",
            json.dumps(build_manifest(candidate), ensure_ascii=False, indent=2),
            "metadata",
        )
    )
    return candidate


def discover_flat_candidates(outputs_root: Path, directory_roots: list[Path]) -> list[CandidateRun]:
    directory_root_set = {path.resolve() for path in directory_roots}
    candidates: list[CandidateRun] = []

    for current_root, dirnames, filenames in os.walk(outputs_root):
        current_path = Path(current_root).resolve()
        if current_path in directory_root_set:
            dirnames[:] = []
            continue

        pruned_dirnames: list[str] = []
        for dirname in dirnames:
            child_dir = (current_path / dirname).resolve()
            if should_skip_path(child_dir):
                continue
            if child_dir in directory_root_set:
                continue
            pruned_dirnames.append(dirname)
        dirnames[:] = pruned_dirnames

        for filename in sorted(filenames):
            path = (current_path / filename).resolve()
            if should_skip_path(path):
                continue

            lower_name = path.name.lower()
            if path.suffix in {".json", ".jsonl"}:
                if "accuracy" in lower_name or lower_name.endswith("summary.json"):
                    continue
                if looks_like_prediction_file(path):
                    candidates.append(build_prediction_candidate(path, outputs_root))
                    continue
                if looks_like_keyframe_file(path):
                    candidates.append(build_keyframe_candidate(path, outputs_root))
                    continue
            elif path.suffix in {".csv", ".md"}:
                candidates.append(build_report_candidate(path, outputs_root))
    return candidates


def discover_candidates(outputs_root: Path) -> list[CandidateRun]:
    directory_roots = collect_directory_roots(outputs_root)
    candidates: list[CandidateRun] = []
    for run_dir in directory_roots:
        candidate = build_directory_candidate(run_dir, outputs_root)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(discover_flat_candidates(outputs_root, directory_roots))
    candidates.sort(key=lambda item: (item.start_time_ms, item.dataset or "", item.source_relpath))
    return candidates


def get_or_create_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(experiment_name)


def existing_run_id(client: MlflowClient, experiment_id: str, fingerprint: str) -> str | None:
    matches = client.search_runs(
        [experiment_id],
        filter_string=f"tags.{BACKFILL_SOURCE_SHA_TAG} = '{fingerprint}'",
        max_results=1,
    )
    if not matches:
        return None
    return matches[0].info.run_id


def log_inline_artifact(client: MlflowClient, run_id: str, filename: str, contents: str, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / filename
        temp_path.write_text(contents, encoding="utf-8")
        client.log_artifact(run_id, str(temp_path), artifact_path=artifact_path)


def create_backfill_run(
    client: MlflowClient,
    experiment_id: str,
    candidate: CandidateRun,
) -> str:
    imported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tags = dict(candidate.tags)
    tags.update(
        {
            RUN_NAME_TAG: candidate.run_name,
            BACKFILL_TAG: "true",
            BACKFILL_SOURCE_SHA_TAG: candidate.fingerprint,
            BACKFILL_SOURCE_PATH_TAG: str(candidate.source_path.resolve()),
            BACKFILL_SOURCE_TYPE_TAG: candidate.source_type,
            BACKFILL_IMPORTED_AT_TAG: imported_at,
        }
    )

    run = client.create_run(
        experiment_id=experiment_id,
        start_time=candidate.start_time_ms,
        tags=tags,
        run_name=candidate.run_name,
    )
    run_id = run.info.run_id

    try:
        for key, value in sorted(candidate.params.items()):
            if value is None:
                continue
            client.log_param(run_id, key, value)
        for key, value in sorted(candidate.metrics.items()):
            metric_value = metric_candidate(value)
            if metric_value is None:
                continue
            client.log_metric(run_id, key, metric_value, timestamp=candidate.end_time_ms)
        for path, artifact_path in candidate.file_artifacts:
            client.log_artifact(run_id, str(path), artifact_path=artifact_path)
        for path, artifact_path in candidate.dir_artifacts:
            client.log_artifacts(run_id, str(path), artifact_path=artifact_path)
        for filename, contents, artifact_path in candidate.inline_artifacts:
            log_inline_artifact(client, run_id, filename, contents, artifact_path)
        client.set_terminated(run_id, status="FINISHED", end_time=candidate.end_time_ms)
    except Exception:
        try:
            client.set_terminated(run_id, status="FAILED", end_time=max(candidate.end_time_ms, candidate.start_time_ms))
        except Exception:
            pass
        raise

    return run_id


def filter_candidates(candidates: list[CandidateRun], args: argparse.Namespace) -> list[CandidateRun]:
    allowed_datasets = {entry.lower() for entry in args.dataset if entry}
    allowed_types = set(args.types)
    filtered: list[CandidateRun] = []
    for candidate in candidates:
        if candidate.source_type not in allowed_types:
            continue
        if allowed_datasets and (candidate.dataset or "").lower() not in allowed_datasets:
            continue
        filtered.append(candidate)
    if args.limit is not None:
        filtered = filtered[: args.limit]
    return filtered


def summarize_candidates(candidates: list[CandidateRun]) -> dict[str, int]:
    summary = {candidate_type: 0 for candidate_type in VALID_TYPES}
    for candidate in candidates:
        summary[candidate.source_type] = summary.get(candidate.source_type, 0) + 1
    return summary


def print_candidate(candidate: CandidateRun) -> None:
    print(
        f"[{candidate.source_type}] {candidate.run_name} "
        f"dataset={candidate.dataset or 'unknown'} workflow={candidate.workflow} "
        f"source={candidate.source_relpath}"
    )


def main() -> int:
    args = parse_args()
    outputs_root = resolve_repo_path(args.outputs_root)
    if outputs_root is None or not outputs_root.exists():
        raise SystemExit(f"Outputs root not found: {args.outputs_root}")

    candidates = filter_candidates(discover_candidates(outputs_root), args)
    summary = summarize_candidates(candidates)
    print(
        "Discovered historical candidates: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    )
    print(f"Total candidates after filtering: {len(candidates)}")

    if args.verbose:
        for candidate in candidates:
            print_candidate(candidate)

    if args.dry_run:
        return 0

    configure_mlflow_environment(args.tracking_uri)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment_id = get_or_create_experiment_id(client, args.experiment_name)

    created = 0
    skipped = 0
    failures: list[str] = []
    for candidate in candidates:
        existing = None if args.force else existing_run_id(client, experiment_id, candidate.fingerprint)
        if existing:
            skipped += 1
            if args.verbose:
                print(f"Skipped existing run {existing} for {candidate.source_relpath}")
            continue
        try:
            run_id = create_backfill_run(client, experiment_id, candidate)
        except Exception as exc:  # pragma: no cover - exercised by manual runs
            failures.append(f"{candidate.source_relpath}: {exc}")
            print(f"Failed to backfill {candidate.source_relpath}: {exc}", file=sys.stderr)
            continue
        created += 1
        if args.verbose:
            print(f"Imported {candidate.source_relpath} -> {run_id}")

    print(f"Backfill complete: created={created}, skipped={skipped}, failed={len(failures)}")
    if failures:
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
