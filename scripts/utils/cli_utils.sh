#!/bin/bash

# Common CLI helpers for strict shell scripts.

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "Error: ${flag} requires a value." >&2
    exit 1
  fi
}

die_unknown_option() {
  local opt="$1"
  echo "Error: Unknown option '${opt}'." >&2
  echo "Run with --help for usage." >&2
  exit 1
}

require_bool_01() {
  local flag="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "Error: ${flag} must be 0 or 1." >&2
    exit 1
  fi
}

slugify_log_label() {
  local raw="${1:-run}"
  raw="${raw// /_}"
  raw="${raw//\//_}"
  raw="${raw//[^A-Za-z0-9._-]/_}"
  raw="${raw##_}"
  raw="${raw%%_}"
  if [[ -z "${raw}" ]]; then
    raw="run"
  fi
  printf '%s' "${raw}"
}

ktv_repo_root() {
  local helper_dir
  helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${helper_dir}/../.." && pwd
}

ktv_mlflow_enabled() {
  local raw="${KTV_MLFLOW_ENABLED:-1}"
  raw="${raw,,}"
  [[ "${raw}" != "0" && "${raw}" != "false" && "${raw}" != "no" && "${raw}" != "off" ]]
}

configure_mlflow_defaults() {
  export KTV_REPO_ROOT="${KTV_REPO_ROOT:-$(ktv_repo_root)}"
  export KTV_MLFLOW_ENABLED="${KTV_MLFLOW_ENABLED:-1}"

  if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
    local tracking_dir
    tracking_dir="$(realpath -m "${KTV_REPO_ROOT}/mlruns")"
    export MLFLOW_TRACKING_URI="file://${tracking_dir}"
  fi
  export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-ktv}"
  export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
}

prepare_experiment_log_files() {
  local experiment_dir="$1"
  local log_label="${2:-run}"
  local timestamp
  local label_slug
  local combined_name
  local stdout_name
  local stderr_name
  local latest_root_path_file

  mkdir -p "${experiment_dir}"

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  label_slug="$(slugify_log_label "${log_label}")"

  export KTV_EXPERIMENT_DIR="${experiment_dir}"
  export KTV_EXPERIMENT_LOG_DIR="${experiment_dir}/logs"
  mkdir -p "${KTV_EXPERIMENT_LOG_DIR}"

  combined_name="${timestamp}_${label_slug}.log"
  stdout_name="${timestamp}_${label_slug}.stdout.log"
  stderr_name="${timestamp}_${label_slug}.stderr.log"

  export KTV_EXPERIMENT_LOG_LABEL="${label_slug}"
  export KTV_EXPERIMENT_LOG_TIMESTAMP="${timestamp}"
  export KTV_EXPERIMENT_COMBINED_LOG_PATH="${KTV_EXPERIMENT_LOG_DIR}/${combined_name}"
  export KTV_EXPERIMENT_STDOUT_LOG_PATH="${KTV_EXPERIMENT_LOG_DIR}/${stdout_name}"
  export KTV_EXPERIMENT_STDERR_LOG_PATH="${KTV_EXPERIMENT_LOG_DIR}/${stderr_name}"

  : > "${KTV_EXPERIMENT_COMBINED_LOG_PATH}"
  : > "${KTV_EXPERIMENT_STDOUT_LOG_PATH}"
  : > "${KTV_EXPERIMENT_STDERR_LOG_PATH}"

  ln -sfn "${combined_name}" "${KTV_EXPERIMENT_LOG_DIR}/latest.log"
  ln -sfn "${stdout_name}" "${KTV_EXPERIMENT_LOG_DIR}/latest.stdout.log"
  ln -sfn "${stderr_name}" "${KTV_EXPERIMENT_LOG_DIR}/latest.stderr.log"

  ln -sfn "logs/${combined_name}" "${experiment_dir}/latest.log"
  ln -sfn "logs/${stdout_name}" "${experiment_dir}/latest.stdout.log"
  ln -sfn "logs/${stderr_name}" "${experiment_dir}/latest.stderr.log"

  latest_root_path_file="${experiment_dir}/latest_log_path.txt"
  printf '%s\n' "${KTV_EXPERIMENT_COMBINED_LOG_PATH}" > "${latest_root_path_file}"
  printf '%s\n' "${KTV_EXPERIMENT_COMBINED_LOG_PATH}" > "${KTV_EXPERIMENT_LOG_DIR}/latest_log_path.txt"
}

print_experiment_log_banner() {
  if [[ "${KTV_EXPERIMENT_LOG_BANNER_PRINTED:-0}" == "1" ]]; then
    return 0
  fi

  export KTV_EXPERIMENT_LOG_BANNER_PRINTED=1
  echo "Persistent logs:"
  echo "  combined: ${KTV_EXPERIMENT_COMBINED_LOG_PATH}"
  echo "  stdout:   ${KTV_EXPERIMENT_STDOUT_LOG_PATH}"
  echo "  stderr:   ${KTV_EXPERIMENT_STDERR_LOG_PATH}"
  echo "  latest:   ${KTV_EXPERIMENT_DIR}/latest.log"
}

start_parent_mlflow_run() {
  local experiment_dir="$1"
  local run_name="${2:-run}"
  local output_dir_abs
  local run_id

  if ! ktv_mlflow_enabled; then
    return 0
  fi
  if [[ -n "${KTV_MLFLOW_PARENT_RUN_ID:-}" ]]; then
    return 0
  fi

  configure_mlflow_defaults
  output_dir_abs="$(realpath -m "${experiment_dir}")"
  run_id="$(uv run python "${KTV_REPO_ROOT}/experiment_tracking.py" start-run \
    --run-name "${run_name}" \
    --stage workflow \
    --script-path "${0##*/}" \
    --output-dir "${output_dir_abs}" \
    --tag "workflow=${run_name}" \
    --tag "output_dir=${output_dir_abs}" 2>/dev/null)"

  if [[ -n "${run_id}" ]]; then
    export KTV_MLFLOW_PARENT_RUN_ID="${run_id}"
    export KTV_MLFLOW_RUN_NAME="${run_name}"
    export KTV_MLFLOW_WORKFLOW="${run_name}"
    export KTV_MLFLOW_TAGS_JSON="{\"workflow\":\"${run_name}\",\"output_dir\":\"${output_dir_abs}\"}"
  else
    echo "Warning: failed to start MLflow parent run for ${run_name}." >&2
  fi
}

finish_parent_mlflow_run() {
  local exit_status="$1"
  local mlflow_status="FINISHED"
  local cmd=()
  local artifact

  if [[ -z "${KTV_MLFLOW_PARENT_RUN_ID:-}" ]]; then
    return 0
  fi
  if [[ "${exit_status}" -ne 0 ]]; then
    mlflow_status="FAILED"
  fi

  cmd=(
    uv run python "${KTV_REPO_ROOT}/experiment_tracking.py" finish-run
    --run-id "${KTV_MLFLOW_PARENT_RUN_ID}"
    --status "${mlflow_status}"
  )

  for artifact in \
    "${KTV_EXPERIMENT_LOG_DIR:-}" \
    "${KTV_EXPERIMENT_DIR:-}/chosen_parameters.env" \
    "${KTV_EXPERIMENT_DIR:-}/latest_log_path.txt" \
    "${KTV_EXPERIMENT_DIR:-}/hydra_resolved_configs"; do
    if [[ -n "${artifact}" && -e "${artifact}" ]]; then
      cmd+=(--artifact-path "${artifact}")
    fi
  done

  if ! "${cmd[@]}" >/dev/null 2>&1; then
    echo "Warning: failed to finalize MLflow parent run ${KTV_MLFLOW_PARENT_RUN_ID}." >&2
  fi

  unset KTV_MLFLOW_PARENT_RUN_ID
  unset KTV_MLFLOW_RUN_NAME
  unset KTV_MLFLOW_WORKFLOW
  unset KTV_MLFLOW_TAGS_JSON
}

print_experiment_log_exit_summary() {
  local exit_status="$1"
  if [[ -z "${KTV_EXPERIMENT_COMBINED_LOG_PATH:-}" ]]; then
    return 0
  fi

  if [[ "${exit_status}" -eq 0 ]]; then
    echo "Run finished successfully. Latest log: ${KTV_EXPERIMENT_DIR}/latest.log"
  else
    echo "Run failed with exit code ${exit_status}. Latest log: ${KTV_EXPERIMENT_DIR}/latest.log" >&2
  fi
}

handle_experiment_shell_exit() {
  local exit_status="$1"
  finish_parent_mlflow_run "${exit_status}"
  print_experiment_log_exit_summary "${exit_status}"
}

enable_current_shell_logging() {
  if [[ -n "${KTV_EXPERIMENT_LOGGING_ENABLED:-}" ]]; then
    return 0
  fi

  export KTV_EXPERIMENT_LOGGING_ENABLED=1
  exec > >(tee -a "${KTV_EXPERIMENT_STDOUT_LOG_PATH}" | tee -a "${KTV_EXPERIMENT_COMBINED_LOG_PATH}")
  exec 2> >(tee -a "${KTV_EXPERIMENT_STDERR_LOG_PATH}" | tee -a "${KTV_EXPERIMENT_COMBINED_LOG_PATH}" >&2)
  print_experiment_log_banner
  trap 'handle_experiment_shell_exit "$?"' EXIT
}

run_with_experiment_logging() {
  local experiment_dir="$1"
  local log_label="${2:-run}"
  shift 2

  (
    prepare_experiment_log_files "${experiment_dir}" "${log_label}"
    enable_current_shell_logging
    start_parent_mlflow_run "${experiment_dir}" "${log_label}"
    "$@"
  )
}
