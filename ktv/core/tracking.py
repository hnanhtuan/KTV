"""Shared MLflow helpers for KTV experiment tracking."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import mlflow
from mlflow.tracking import MlflowClient

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError:  # pragma: no cover - OmegaConf is available in normal runtime.
    DictConfig = None  # type: ignore[assignment]
    ListConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


DEFAULT_EXPERIMENT_NAME = "ktv"
PARENT_RUN_TAG = "mlflow.parentRunId"
RUN_NAME_TAG = "mlflow.runName"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_bool(value: Any, default: bool = True) -> bool:
    """Parse a flexible boolean value from config or environment strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def repo_tracking_uri(repo_root: Path | None = None) -> str:
    """Return the default repo-local MLflow tracking URI."""
    root = repo_root or REPO_ROOT
    return (root / "mlruns").resolve().as_uri()


def configure_mlflow_environment(tracking_uri: str) -> None:
    """Apply environment defaults required by local-first MLflow tracking."""
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    if tracking_uri.startswith("file:") or "://" not in tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def _looks_like_omegaconf(value: Any) -> bool:
    if OmegaConf is None:
        return False
    supported = tuple(
        entry for entry in (DictConfig, ListConfig) if entry is not None
    )
    return isinstance(value, supported)


def _to_plain_data(value: Any, *, resolve: bool = True) -> Any:
    if _looks_like_omegaconf(value):
        return OmegaConf.to_container(value, resolve=resolve)
    return value


def _normalize_tags(raw_tags: Any) -> dict[str, str]:
    if raw_tags is None:
        return {}
    raw_tags = _to_plain_data(raw_tags)
    if not isinstance(raw_tags, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw_tags.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _read_json_env(env_name: str) -> dict[str, str]:
    raw_value = os.environ.get(env_name)
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return _normalize_tags(parsed)


def flatten_params(
    value: Any,
    *,
    prefix: str = "",
    exclude_top_level: Iterable[str] | None = None,
) -> dict[str, str]:
    """Flatten a config structure into MLflow-friendly param key/value pairs."""
    plain_value = _to_plain_data(value, resolve=True)
    exclude = {entry for entry in (exclude_top_level or ())}
    flattened: dict[str, str] = {}

    def _walk(current: Any, current_prefix: str) -> None:
        if isinstance(current, Mapping):
            for key, nested_value in current.items():
                if not current_prefix and key in exclude:
                    continue
                next_prefix = f"{current_prefix}.{key}" if current_prefix else str(key)
                _walk(nested_value, next_prefix)
            return

        if isinstance(current, (list, tuple)):
            flattened[current_prefix] = json.dumps(current, ensure_ascii=True)
            return

        if current is None:
            flattened[current_prefix] = "null"
            return

        if isinstance(current, bool):
            flattened[current_prefix] = "true" if current else "false"
            return

        flattened[current_prefix] = str(current)

    if plain_value is None:
        return flattened
    if isinstance(plain_value, Mapping):
        _walk(plain_value, prefix)
    else:
        flattened[prefix or "value"] = str(plain_value)
    return flattened


def _metric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def sanitize_metric_key(value: str) -> str:
    """Convert a free-form label into a metric-safe key suffix."""
    safe = []
    for char in value:
        if char.isalnum():
            safe.append(char.lower())
        else:
            safe.append("_")
    normalized = "".join(safe).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "unknown"


@dataclass
class TrackingConfig:
    enabled: bool
    tracking_uri: str
    experiment_name: str
    run_name: str | None
    nested: bool
    parent_run_id: str | None
    tags: dict[str, str] = field(default_factory=dict)


def resolve_tracking_config(
    cfg: Any = None,
    *,
    stage: str | None = None,
    script_path: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    run_name: str | None = None,
    nested: bool | None = None,
    extra_tags: Mapping[str, Any] | None = None,
) -> TrackingConfig:
    """Resolve MLflow config from Hydra config plus environment defaults."""
    cfg_mlflow: Mapping[str, Any] = {}
    if cfg is not None:
        cfg_data = _to_plain_data(cfg, resolve=True)
        if isinstance(cfg_data, Mapping):
            raw_mlflow = cfg_data.get("mlflow")
            if isinstance(raw_mlflow, Mapping):
                cfg_mlflow = raw_mlflow

    enabled = parse_bool(
        cfg_mlflow.get("enabled", os.environ.get("KTV_MLFLOW_ENABLED", "1"))
    )
    tracking_uri = str(
        cfg_mlflow.get("tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI")
        or repo_tracking_uri()
    )
    experiment_name = str(
        cfg_mlflow.get("experiment_name")
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or DEFAULT_EXPERIMENT_NAME
    )

    parent_run_id = os.environ.get("KTV_MLFLOW_PARENT_RUN_ID") or None
    if nested is None:
        nested = parse_bool(cfg_mlflow.get("nested"), default=bool(parent_run_id))

    merged_tags: dict[str, str] = {}
    merged_tags.update(_read_json_env("KTV_MLFLOW_TAGS_JSON"))
    merged_tags.update(_normalize_tags(cfg_mlflow.get("tags")))
    merged_tags.update(_normalize_tags(extra_tags))

    if stage:
        merged_tags.setdefault("stage", stage)
    if script_path:
        merged_tags.setdefault("script", Path(script_path).name)
    if output_dir is not None:
        merged_tags.setdefault("output_dir", str(Path(output_dir).resolve()))
    workflow = os.environ.get("KTV_MLFLOW_WORKFLOW")
    if workflow:
        merged_tags.setdefault("workflow", workflow)

    if run_name is None:
        run_name = cfg_mlflow.get("run_name")
    if run_name is None:
        parent_name = os.environ.get("KTV_MLFLOW_RUN_NAME")
        if parent_name and stage:
            run_name = f"{parent_name}/{stage}"
        elif stage:
            run_name = stage
        elif script_path:
            run_name = Path(script_path).stem

    return TrackingConfig(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=str(run_name) if run_name else None,
        nested=bool(nested),
        parent_run_id=parent_run_id,
        tags=merged_tags,
    )


def _get_or_create_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(experiment_name)


def create_run(config: TrackingConfig) -> str:
    """Create an MLflow run without attaching logging state to the current process."""
    configure_mlflow_environment(config.tracking_uri)
    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient()
    experiment_id = _get_or_create_experiment_id(client, config.experiment_name)

    tags = dict(config.tags)
    if config.run_name:
        tags.setdefault(RUN_NAME_TAG, config.run_name)
    if config.parent_run_id and config.nested:
        tags.setdefault(PARENT_RUN_TAG, config.parent_run_id)

    run = client.create_run(experiment_id=experiment_id, tags=tags)
    return run.info.run_id


class ExperimentTracker(AbstractContextManager["ExperimentTracker"]):
    """Managed MLflow child run with lightweight artifact helpers."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.run_id: str | None = None
        self._active = False

    def __enter__(self) -> "ExperimentTracker":
        if not self.config.enabled:
            return self
        self.run_id = create_run(self.config)
        configure_mlflow_environment(self.config.tracking_uri)
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.start_run(run_id=self.run_id)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self._active:
            status = "FAILED" if exc_type is not None else "FINISHED"
            mlflow.end_run(status=status)
            self._active = False
        return False

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if not self._active:
            return
        for key, value in _normalize_tags(tags).items():
            mlflow.set_tag(key, value)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self._active:
            return
        for key, value in params.items():
            if value is None:
                continue
            mlflow.log_param(str(key), str(value))

    def log_params_from_config(
        self,
        cfg: Any,
        *,
        exclude_top_level: Iterable[str] = ("hydra", "mlflow"),
    ) -> None:
        if not self._active:
            return
        self.log_params(flatten_params(cfg, exclude_top_level=exclude_top_level))

    def log_metrics(self, metrics: Mapping[str, Any]) -> None:
        if not self._active:
            return
        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            metric_value = _metric_value(value)
            if metric_value is None:
                continue
            normalized[str(key)] = metric_value
        if normalized:
            mlflow.log_metrics(normalized)

    def log_artifact(
        self,
        path: str | os.PathLike[str],
        *,
        artifact_path: str | None = None,
    ) -> None:
        if not self._active:
            return
        resolved_path = Path(path)
        if not resolved_path.exists():
            return
        if resolved_path.is_dir():
            mlflow.log_artifacts(str(resolved_path), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(resolved_path), artifact_path=artifact_path)

    def log_artifacts(
        self,
        paths: Iterable[str | os.PathLike[str] | None],
        *,
        artifact_path: str | None = None,
    ) -> None:
        for path in paths:
            if path is None:
                continue
            self.log_artifact(path, artifact_path=artifact_path)

    def log_text_artifact(
        self,
        text: str,
        filename: str,
        *,
        artifact_path: str | None = None,
    ) -> None:
        if not self._active:
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / filename
            output_path.write_text(text, encoding="utf-8")
            self.log_artifact(output_path, artifact_path=artifact_path)

    def log_json_artifact(
        self,
        payload: Any,
        filename: str,
        *,
        artifact_path: str | None = None,
    ) -> None:
        if not self._active:
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / filename
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.log_artifact(output_path, artifact_path=artifact_path)

    def log_resolved_config(
        self,
        cfg: Any,
        *,
        filename: str = "resolved_config.yaml",
        artifact_path: str = "config",
    ) -> None:
        if not self._active:
            return
        if _looks_like_omegaconf(cfg):
            contents = OmegaConf.to_yaml(cfg, resolve=True)
        else:
            contents = json.dumps(_to_plain_data(cfg), ensure_ascii=False, indent=2)
        self.log_text_artifact(contents, filename, artifact_path=artifact_path)


def write_summary_json(
    summary_path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> str:
    """Write a JSON summary file and return its absolute path."""
    resolved_path = Path(summary_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(resolved_path.resolve())


def default_shell_artifact_paths() -> list[str]:
    """Return shell-produced lightweight artifact paths when available."""
    candidates = [
        os.environ.get("KTV_EXPERIMENT_COMBINED_LOG_PATH"),
        os.environ.get("KTV_EXPERIMENT_STDOUT_LOG_PATH"),
        os.environ.get("KTV_EXPERIMENT_STDERR_LOG_PATH"),
    ]
    experiment_dir = os.environ.get("KTV_EXPERIMENT_DIR")
    if experiment_dir:
        candidates.extend(
            [
                os.path.join(experiment_dir, "chosen_parameters.env"),
                os.path.join(experiment_dir, "latest_log_path.txt"),
                os.path.join(experiment_dir, "hydra_resolved_configs"),
            ]
        )
    return [path for path in candidates if path]


def track_run(
    cfg: Any = None,
    *,
    stage: str,
    script_path: str,
    output_dir: str | os.PathLike[str] | None = None,
    run_name: str | None = None,
    extra_tags: Mapping[str, Any] | None = None,
) -> ExperimentTracker:
    """Create a managed tracker for one Python stage or report."""
    config = resolve_tracking_config(
        cfg,
        stage=stage,
        script_path=script_path,
        output_dir=output_dir,
        run_name=run_name,
        extra_tags=extra_tags,
    )
    return ExperimentTracker(config)


def log_accuracy_metrics(
    tracker: ExperimentTracker,
    summary: Mapping[str, Any],
    *,
    overall_prefix: str = "overall",
    task_prefix: str = "task",
) -> None:
    """Log overall accuracy metric from a structured summary."""
    if not tracker.config.enabled:
        return
    tracker.log_metrics(
        {
            f"{overall_prefix}.accuracy": summary.get("accuracy"),
        }
    )


def _parse_tag_args(raw_tags: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_tag in raw_tags:
        if "=" not in raw_tag:
            continue
        key, value = raw_tag.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        parsed[key] = value
    return parsed


def _cli_tracking_config(args: argparse.Namespace) -> TrackingConfig:
    merged_tags = _parse_tag_args(args.tag)
    if args.tags_json:
        try:
            merged_tags.update(_normalize_tags(json.loads(args.tags_json)))
        except json.JSONDecodeError:
            pass
    return resolve_tracking_config(
        None,
        stage=args.stage,
        script_path=args.script_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        nested=args.nested,
        extra_tags=merged_tags,
    )


def _cli_start_run(args: argparse.Namespace) -> int:
    config = _cli_tracking_config(args)
    if not config.enabled:
        return 0
    print(create_run(config))
    return 0


def _cli_finish_run(args: argparse.Namespace) -> int:
    tracking_uri = (
        args.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or repo_tracking_uri()
    )
    configure_mlflow_environment(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.start_run(run_id=args.run_id)
    for path in args.artifact_path:
        resolved_path = Path(path)
        if not resolved_path.exists():
            continue
        if resolved_path.is_dir():
            mlflow.log_artifacts(str(resolved_path))
        else:
            mlflow.log_artifact(str(resolved_path))
    for key, value in _parse_tag_args(args.tag).items():
        mlflow.set_tag(key, value)
    mlflow.end_run(status=args.status)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KTV MLflow helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start-run")
    start_parser.add_argument("--run-name", default=None)
    start_parser.add_argument("--stage", default="workflow")
    start_parser.add_argument("--script-path", default=None)
    start_parser.add_argument("--output-dir", default=None)
    start_parser.add_argument("--nested", action="store_true")
    start_parser.add_argument("--tag", action="append", default=[])
    start_parser.add_argument("--tags-json", default=None)
    start_parser.set_defaults(func=_cli_start_run)

    finish_parser = subparsers.add_parser("finish-run")
    finish_parser.add_argument("--run-id", required=True)
    finish_parser.add_argument("--status", default="FINISHED")
    finish_parser.add_argument("--tracking-uri", default=None)
    finish_parser.add_argument("--artifact-path", action="append", default=[])
    finish_parser.add_argument("--tag", action="append", default=[])
    finish_parser.set_defaults(func=_cli_finish_run)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
