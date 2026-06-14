#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/utils/cli_utils.sh
source "${SCRIPT_DIR}/scripts/utils/cli_utils.sh"

print_help() {
  cat <<'USAGE'
Usage:
  run_query_aware_upper_bound_experiments.sh [options]

This script runs exactly one dense-uniform query-aware upper-bound setting.
Parameter sweeps and non-upper-bound variants are not supported.

Strict options (unknown flags fail):
  --dataset <name>                        Dataset to run.
                                          Default: videomme
  --device <name>                         Device override for feature extraction and selection.
                                          Default: auto
  --run-dir <path>                        Explicit metadata/logging run dir. If omitted, one is created.
  --run-root-dir <path>                   Base dir for auto-created run dirs.
                                          Default: outputs/query_aware_upper_bound
  --unique-exp-id <id>                    Run folder name when --run-dir is omitted.
                                          Default: UTC timestamp + pid

  --tokens-num <int>                      Token budget for inference.
                                          Default: 504
  --prune-mode <mode>                     Inference prune mode.
                                          Default: cls_new_token_sim
  --query-mode <mode>                     Query text mode.
                                          Default: question_only
  --dense-candidate-pool-size <int>       Dense-uniform candidate pool size.
                                          Default: 48
  --rate <value>                          Token-pruning rate for inference. Default: 0.2
  --num-frames <int>                      Inference num_frames override. Default: 6
  --output-top-k <int>                    Final selected keyframes per question. Default: 6

  --run-feature-extract <0|1>             Run extract_frame_features.py first. Default: 1
  --run-reference-full-ktv <0|1>          Unsupported in this script. Must be 0. Default: 0
  --run-query-aware-12-candidate <0|1>    Unsupported in this script. Must be 0. Default: 0
  --run-query-aware-dense-uniform <0|1>   Run dense-uniform query-aware variant. Must be 1. Default: 1
  --generate-report <0|1>                 Generate markdown comparison report. Default: 1
  --report-output-path <path>             Report output path.
                                          Default: <run-dir>/query_aware_upper_bound_report.md

Optional repeatable Hydra override flags:
  --feature-override <key=value>          Extra override for extract_frame_features.py
  --reference-keyframe-override <key=value>
                                          Extra override for cluster_and_rank_keyframes.py
  --query-keyframe-override <key=value>   Extra override for query_aware_select_keyframes.py
  --inference-override <key=value>        Extra override for run_inference_multiple_choice_qa.py

Other:
  --debug                                 Enable verbose debugging
  -h, --help                              Show this help
USAGE
}

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

DEFAULT_DATASETS=(videomme)
DEFAULT_TOKEN_LIST=(504)
DEFAULT_PRUNE_MODES=(cls_new_token_sim)
DEFAULT_QUERY_MODES=(question_only)
DEFAULT_DENSE_POOL_SIZES=(48)

DATASETS=()
TOKEN_LIST=()
PRUNE_MODES=()
QUERY_MODES=()
DENSE_CANDIDATE_POOL_SIZES=()
FEATURE_OVERRIDES=()
REFERENCE_KEYFRAME_OVERRIDES=()
QUERY_KEYFRAME_OVERRIDES=()
INFERENCE_OVERRIDES=()

RATE="0.2"
DEVICE="auto"
NUM_FRAMES="6"
OUTPUT_TOP_K="6"
RUN_FEATURE_EXTRACT="1"
RUN_REFERENCE_FULL_KTV="0"
RUN_QUERY_AWARE_12_CANDIDATE="0"
RUN_QUERY_AWARE_DENSE_UNIFORM="1"
GENERATE_REPORT="1"
REPORT_OUTPUT_PATH=""
RUN_DIR=""
RUN_ROOT_DIR="outputs/query_aware_upper_bound"
UNIQUE_EXP_ID=""
DEBUG=0

require_single_value() {
  local flag="$1"
  local count="$2"
  if [[ "${count}" -gt 1 ]]; then
    echo "Error: ${flag} only accepts a single value in this script." >&2
    exit 1
  fi
}

write_logged_parameter() {
  local key="$1"
  local value="$2"
  printf '%s=%q\n' "${key}" "${value}"
}

join_by_space() {
  if [[ $# -eq 0 ]]; then
    printf ''
    return
  fi
  printf '%s' "$1"
  shift
  while [[ $# -gt 0 ]]; do
    printf ' %s' "$1"
    shift
  done
}

resolve_inference_experiment() {
  local dataset="$1"
  case "${dataset}" in
    videomme) echo "videomme_6keyframe" ;;
    *) echo "${dataset}" ;;
  esac
}

resolve_keyframe_experiment() {
  local dataset="$1"
  case "${dataset}" in
    videomme_6keyframe) echo "videomme" ;;
    *) echo "${dataset}" ;;
  esac
}

reference_keyframe_json() {
  local dataset="$1"
  echo "outputs/${dataset}/keyframe_selection/keyframe${OUTPUT_TOP_K}_order.json"
}

variant_dir_name() {
  local selection_mode="$1"
  local dense_pool_size="$2"
  if [[ "${selection_mode}" == "uniform_12" || "${selection_mode}" == "clustered_12" ]]; then
    echo "query_aware_12_candidate"
  else
    echo "query_aware_dense_uniform_f${dense_pool_size}"
  fi
}

cfg_dump_path() {
  local stage_key="$1"
  echo "${HYDRA_LOG_DIR}/${stage_key}.yaml"
}

log_stage_config() {
  local stage_key="$1"
  shift
  uv run python "$@" --cfg job --resolve > "$(cfg_dump_path "${stage_key}")"
}

run_inference_and_eval() {
  local dataset="$1"
  local setting_dir="$2"
  local tokens_num="$3"
  local keyframe_json="$4"
  local prune_mode="$5"
  shift 5

  local output_name="predictions_tokens${tokens_num}"
  local pred_path="${setting_dir}/${output_name}.json"
  local acc_path="${setting_dir}/accuracy_tokens${tokens_num}.txt"
  local experiment_name
  experiment_name="$(resolve_inference_experiment "${dataset}")"
  local stage_key="inference_${dataset}_${prune_mode}_tokens${tokens_num}_$(basename "${setting_dir}")"

  mkdir -p "${setting_dir}"

  echo "=================================================="
  echo "Dataset: ${dataset} | Tokens: ${tokens_num} | Dir: ${setting_dir}"
  echo "=================================================="

  log_stage_config "${stage_key}" run_inference_multiple_choice_qa.py \
    experiment="${experiment_name}" \
    device="${DEVICE}" \
    output_dir="${setting_dir}" \
    output_name="${output_name}" \
    key_frame_path="${keyframe_json}" \
    prune_mode="${prune_mode}" \
    rate="${RATE}" \
    tokens_num="${tokens_num}" \
    num_frames="${NUM_FRAMES}" \
    "$@" \
    "${INFERENCE_OVERRIDES[@]}"

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${experiment_name}" \
    device="${DEVICE}" \
    output_dir="${setting_dir}" \
    output_name="${output_name}" \
    key_frame_path="${keyframe_json}" \
    prune_mode="${prune_mode}" \
    rate="${RATE}" \
    tokens_num="${tokens_num}" \
    num_frames="${NUM_FRAMES}" \
    "$@" \
    "${INFERENCE_OVERRIDES[@]}"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" --json-output "${acc_path%.txt}.json" | tee "${acc_path}"
  echo "Saved accuracy report: ${acc_path}"
}

run_reference_full_ktv() {
  local dataset="$1"
  local keyframe_json="$2"

  for prune_mode in "${PRUNE_MODES[@]}"; do
    local setting_dir="outputs/${dataset}/ktv_full_${prune_mode}"
    local tokens_num
    for tokens_num in "${TOKEN_LIST[@]}"; do
      run_inference_and_eval "${dataset}" "${setting_dir}" "${tokens_num}" "${keyframe_json}" "${prune_mode}"
    done
  done
}

run_query_aware_variant() {
  local dataset="$1"
  local selection_mode="$2"
  local query_mode="$3"
  local dense_pool_size="$4"
  local variant_dir
  variant_dir="$(variant_dir_name "${selection_mode}" "${dense_pool_size}")"

  local keyframe_dir="outputs/${dataset}/${variant_dir}/${query_mode}/keyframes"
  local keyframe_json="outputs/${dataset}/${variant_dir}/${query_mode}/keyframe${OUTPUT_TOP_K}_order.json"
  local stage_key="select_${dataset}_${variant_dir}_${query_mode}"

  log_stage_config "${stage_key}" query_aware_select_keyframes.py     experiment="${dataset}"     device="${DEVICE}"     selection_mode="${selection_mode}"     query_mode="${query_mode}"     dense_candidate_pool_size="${dense_pool_size}"     output_top_k="${OUTPUT_TOP_K}"     save_cluster_path="${keyframe_dir}"     combined_output_path="${keyframe_json}"     "${QUERY_KEYFRAME_OVERRIDES[@]}"

  uv run python query_aware_select_keyframes.py     experiment="${dataset}"     device="${DEVICE}"     selection_mode="${selection_mode}"     query_mode="${query_mode}"     dense_candidate_pool_size="${dense_pool_size}"     output_top_k="${OUTPUT_TOP_K}"     save_cluster_path="${keyframe_dir}"     combined_output_path="${keyframe_json}"     "${QUERY_KEYFRAME_OVERRIDES[@]}"

  local setting_dir="outputs/${dataset}/${variant_dir}/${query_mode}/${PRUNE_MODE}"
  run_inference_and_eval     "${dataset}"     "${setting_dir}"     "${TOKENS_NUM}"     "${keyframe_json}"     "${PRUNE_MODE}"
}

query_aware_experiment_dir() {
  local dataset="$1"
  local selection_mode="$2"
  local query_mode="$3"
  local dense_pool_size="$4"
  local variant_dir
  variant_dir="$(variant_dir_name "${selection_mode}" "${dense_pool_size}")"
  echo "outputs/${dataset}/${variant_dir}/${query_mode}/${PRUNE_MODE}"
}

run_query_aware_workflow() {
  local dataset="$1"
  local experiment_name
  experiment_name="$(resolve_keyframe_experiment "${dataset}")"

  if [[ "${RUN_FEATURE_EXTRACT}" == "1" ]]; then
    echo "Preparing frame features for ${dataset}"
    log_stage_config "feature_${dataset}" extract_frame_features.py       experiment="${experiment_name}"       device="${DEVICE}"       "${FEATURE_OVERRIDES[@]}"

    uv run python extract_frame_features.py       experiment="${experiment_name}"       device="${DEVICE}"       "${FEATURE_OVERRIDES[@]}"
  fi

  echo "Running query-aware dense-uniform upper-bound setting for ${dataset}"
  run_query_aware_variant "${dataset}" "dense_uniform" "${QUERY_MODE}" "${DENSE_CANDIDATE_POOL_SIZE}"
}

generate_query_aware_report() {
  log_stage_config "report" report_query_aware_upper_bound.py     --datasets "${DATASET}"     --query-modes "${QUERY_MODE}"     --prune-modes "${PRUNE_MODE}"     --token-list "${TOKENS_NUM}"     --dense-pool-sizes "${DENSE_CANDIDATE_POOL_SIZE}"     --output-path "${REPORT_OUTPUT_PATH}"

  uv run python report_query_aware_upper_bound.py     --datasets "${DATASET}"     --query-modes "${QUERY_MODE}"     --prune-modes "${PRUNE_MODE}"     --token-list "${TOKENS_NUM}"     --dense-pool-sizes "${DENSE_CANDIDATE_POOL_SIZE}"     --output-path "${REPORT_OUTPUT_PATH}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) require_value "$1" "${2:-}"; DATASETS+=("$2"); shift 2 ;;
    --device) require_value "$1" "${2:-}"; DEVICE="$2"; shift 2 ;;
    --run-dir) require_value "$1" "${2:-}"; RUN_DIR="$2"; shift 2 ;;
    --run-root-dir) require_value "$1" "${2:-}"; RUN_ROOT_DIR="$2"; shift 2 ;;
    --unique-exp-id) require_value "$1" "${2:-}"; UNIQUE_EXP_ID="$2"; shift 2 ;;
    --tokens-num) require_value "$1" "${2:-}"; TOKEN_LIST+=("$2"); shift 2 ;;
    --prune-mode) require_value "$1" "${2:-}"; PRUNE_MODES+=("$2"); shift 2 ;;
    --query-mode) require_value "$1" "${2:-}"; QUERY_MODES+=("$2"); shift 2 ;;
    --dense-candidate-pool-size) require_value "$1" "${2:-}"; DENSE_CANDIDATE_POOL_SIZES+=("$2"); shift 2 ;;
    --rate) require_value "$1" "${2:-}"; RATE="$2"; shift 2 ;;
    --num-frames) require_value "$1" "${2:-}"; NUM_FRAMES="$2"; shift 2 ;;
    --output-top-k) require_value "$1" "${2:-}"; OUTPUT_TOP_K="$2"; shift 2 ;;
    --run-feature-extract) require_value "$1" "${2:-}"; RUN_FEATURE_EXTRACT="$2"; shift 2 ;;
    --run-reference-full-ktv) require_value "$1" "${2:-}"; RUN_REFERENCE_FULL_KTV="$2"; shift 2 ;;
    --run-query-aware-12-candidate) require_value "$1" "${2:-}"; RUN_QUERY_AWARE_12_CANDIDATE="$2"; shift 2 ;;
    --run-query-aware-dense-uniform) require_value "$1" "${2:-}"; RUN_QUERY_AWARE_DENSE_UNIFORM="$2"; shift 2 ;;
    --generate-report) require_value "$1" "${2:-}"; GENERATE_REPORT="$2"; shift 2 ;;
    --report-output-path) require_value "$1" "${2:-}"; REPORT_OUTPUT_PATH="$2"; shift 2 ;;
    --feature-override) require_value "$1" "${2:-}"; FEATURE_OVERRIDES+=("$2"); shift 2 ;;
    --reference-keyframe-override) require_value "$1" "${2:-}"; REFERENCE_KEYFRAME_OVERRIDES+=("$2"); shift 2 ;;
    --query-keyframe-override) require_value "$1" "${2:-}"; QUERY_KEYFRAME_OVERRIDES+=("$2"); shift 2 ;;
    --inference-override) require_value "$1" "${2:-}"; INFERENCE_OVERRIDES+=("$2"); shift 2 ;;
    --debug) DEBUG=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) die_unknown_option "$1" ;;
  esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi
if [[ ${#TOKEN_LIST[@]} -eq 0 ]]; then
  TOKEN_LIST=("${DEFAULT_TOKEN_LIST[@]}")
fi
if [[ ${#PRUNE_MODES[@]} -eq 0 ]]; then
  PRUNE_MODES=("${DEFAULT_PRUNE_MODES[@]}")
fi
if [[ ${#QUERY_MODES[@]} -eq 0 ]]; then
  QUERY_MODES=("${DEFAULT_QUERY_MODES[@]}")
fi
if [[ ${#DENSE_CANDIDATE_POOL_SIZES[@]} -eq 0 ]]; then
  DENSE_CANDIDATE_POOL_SIZES=("${DEFAULT_DENSE_POOL_SIZES[@]}")
fi

require_single_value "--dataset" "${#DATASETS[@]}"
require_single_value "--tokens-num" "${#TOKEN_LIST[@]}"
require_single_value "--prune-mode" "${#PRUNE_MODES[@]}"
require_single_value "--query-mode" "${#QUERY_MODES[@]}"
require_single_value "--dense-candidate-pool-size" "${#DENSE_CANDIDATE_POOL_SIZES[@]}"

require_bool_01 "--run-feature-extract" "${RUN_FEATURE_EXTRACT}"
require_bool_01 "--run-reference-full-ktv" "${RUN_REFERENCE_FULL_KTV}"
require_bool_01 "--run-query-aware-12-candidate" "${RUN_QUERY_AWARE_12_CANDIDATE}"
require_bool_01 "--run-query-aware-dense-uniform" "${RUN_QUERY_AWARE_DENSE_UNIFORM}"
require_bool_01 "--generate-report" "${GENERATE_REPORT}"

if [[ "${RUN_REFERENCE_FULL_KTV}" != "0" ]]; then
  echo "Error: --run-reference-full-ktv must be 0; this script only runs the query-aware upper-bound setting." >&2
  exit 1
fi
if [[ "${RUN_QUERY_AWARE_12_CANDIDATE}" != "0" ]]; then
  echo "Error: --run-query-aware-12-candidate must be 0; this script only runs the dense-uniform query-aware upper-bound setting." >&2
  exit 1
fi
if [[ "${RUN_QUERY_AWARE_DENSE_UNIFORM}" != "1" ]]; then
  echo "Error: --run-query-aware-dense-uniform must be 1; this script only runs the dense-uniform query-aware upper-bound setting." >&2
  exit 1
fi

DATASET="${DATASETS[0]}"
TOKENS_NUM="${TOKEN_LIST[0]}"
PRUNE_MODE="${PRUNE_MODES[0]}"
QUERY_MODE="${QUERY_MODES[0]}"
DENSE_CANDIDATE_POOL_SIZE="${DENSE_CANDIDATE_POOL_SIZES[0]}"

UNIQUE_EXP_ID="${UNIQUE_EXP_ID:-$(date -u +%Y%m%dT%H%M%SZ)_pid$$}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${RUN_ROOT_DIR}/${UNIQUE_EXP_ID}"
fi
HYDRA_LOG_DIR="${RUN_DIR}/hydra_resolved_configs"
mkdir -p "${RUN_DIR}" "${HYDRA_LOG_DIR}"

if [[ -z "${REPORT_OUTPUT_PATH}" ]]; then
  REPORT_OUTPUT_PATH="${RUN_DIR}/query_aware_upper_bound_report.md"
fi

PARAM_LOG_PATH="${RUN_DIR}/chosen_parameters.env"
{
  write_logged_parameter "RUN_TIMESTAMP_UTC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_logged_parameter "RUN_DIR" "${RUN_DIR}"
  write_logged_parameter "RUN_ROOT_DIR" "${RUN_ROOT_DIR}"
  write_logged_parameter "UNIQUE_EXP_ID" "${UNIQUE_EXP_ID}"
  write_logged_parameter "DATASET" "${DATASET}"
  write_logged_parameter "DEVICE" "${DEVICE}"
  write_logged_parameter "TOKENS_NUM" "${TOKENS_NUM}"
  write_logged_parameter "PRUNE_MODE" "${PRUNE_MODE}"
  write_logged_parameter "QUERY_MODE" "${QUERY_MODE}"
  write_logged_parameter "DENSE_CANDIDATE_POOL_SIZE" "${DENSE_CANDIDATE_POOL_SIZE}"
  write_logged_parameter "RATE" "${RATE}"
  write_logged_parameter "NUM_FRAMES" "${NUM_FRAMES}"
  write_logged_parameter "OUTPUT_TOP_K" "${OUTPUT_TOP_K}"
  write_logged_parameter "RUN_FEATURE_EXTRACT" "${RUN_FEATURE_EXTRACT}"
  write_logged_parameter "RUN_REFERENCE_FULL_KTV" "${RUN_REFERENCE_FULL_KTV}"
  write_logged_parameter "RUN_QUERY_AWARE_12_CANDIDATE" "${RUN_QUERY_AWARE_12_CANDIDATE}"
  write_logged_parameter "RUN_QUERY_AWARE_DENSE_UNIFORM" "${RUN_QUERY_AWARE_DENSE_UNIFORM}"
  write_logged_parameter "GENERATE_REPORT" "${GENERATE_REPORT}"
  write_logged_parameter "REPORT_OUTPUT_PATH" "${REPORT_OUTPUT_PATH}"
  write_logged_parameter "FEATURE_OVERRIDES" "$(join_by_space "${FEATURE_OVERRIDES[@]}")"
  write_logged_parameter "REFERENCE_KEYFRAME_OVERRIDES" "$(join_by_space "${REFERENCE_KEYFRAME_OVERRIDES[@]}")"
  write_logged_parameter "QUERY_KEYFRAME_OVERRIDES" "$(join_by_space "${QUERY_KEYFRAME_OVERRIDES[@]}")"
  write_logged_parameter "INFERENCE_OVERRIDES" "$(join_by_space "${INFERENCE_OVERRIDES[@]}")"
} > "${PARAM_LOG_PATH}"

if [[ "${DEBUG}" == "1" ]]; then
  export HYDRA_FULL_ERROR=1
  export PYTHONFAULTHANDLER=1
  export PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
  set -x
fi

setting_dir="$(query_aware_experiment_dir "${DATASET}" "dense_uniform" "${QUERY_MODE}" "${DENSE_CANDIDATE_POOL_SIZE}")"
run_with_experiment_logging "${setting_dir}" "query_aware_${DATASET}_${QUERY_MODE}_${PRUNE_MODE}_tokens${TOKENS_NUM}" run_query_aware_workflow "${DATASET}"

if [[ "${GENERATE_REPORT}" == "1" ]]; then
  run_with_experiment_logging "${RUN_DIR}" "query_aware_upper_bound_report" generate_query_aware_report
fi
