#!/bin/bash
set -euo pipefail

# Load shared strict-CLI helpers.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/utils/cli_utils.sh
source "${SCRIPT_DIR}/scripts/utils/cli_utils.sh"

print_help() {
  cat <<'USAGE'
Usage:
  run_temporal_chain_experiments.sh [options]

Strict options (unknown flags fail):
  --dataset <name>                     Dataset name (default: nextqa)
  --device <name>                      Device override for all stages (default: auto)
  --output-dir <path>                  Experiment output dir; if omitted, a unique run folder is created under outputs/<dataset>/temporal_chain/

  --run-keyframe-select <0|1>          Run frame-feature extraction stage (default: 0)
  --feature-dataset <name>             Feature extraction experiment (default: <dataset> without _temporal_chain)
  --keyframe-dataset <name>            Keyframe-ranking experiment (default: <feature-dataset>_temporal_chain)
  --inference-dataset <name>           Inference experiment alias before mapping (default: <feature-dataset>)
  --inference-config <name>            Final inference experiment config (default: mapped from --inference-dataset)

  --dataset-slug <name>                Slug used in auto-generated run folder names
  --run-root-dir <path>                Base dir for auto-created experiment folders (default: outputs/<dataset>/temporal_chain)
  --unique-exp-id <id>                 Experiment folder name for auto-created runs (default: UTC timestamp + pid)

  --num-keyframes <int>                Temporal-chain keyframe count (default: 12)
  --lambda-event <float>               Temporal-chain weight (default: 0.5)
  --alpha-gap <float>                  Temporal-chain weight (default: 0.6)
  --beta-redundancy <float>            Temporal-chain weight (default: 0.8)
  --score-normalizer <name>            Score normalizer for temporal-chain terms (default: minmax)
  --max-frames-to-extract <int>        Temporal-chain frame cap (default: 5400)
  --keyframe-json <path>               Combined keyframe json (default: <output-dir>/keyframe6_order.json)
  --keyframe-impl <single|multiprocess>
                                       Keyframe script variant (default: single)

  --output-name <name>                 Inference output file stem (default: predictions)
  --num-frames <int>                   Inference frames (default: 6)
  --prune-mode <mode>                  Inference prune mode (default: null)
  --rate <value>                       Inference pruning rate (default: null)
  --tokens-num <int>                   Inference token budget (default: 936)

Optional repeatable Hydra override flags:
  --feature-override <key=value>       Extra override for extract_frame_features.py
  --keyframe-override <key=value>      Extra override for temporal_chain_rank_keyframes.py
  --inference-override <key=value>     Extra override for run_inference_multiple_choice_qa.py

Other:
  --debug                              Enable verbose debugging
  -h, --help                           Show this help
USAGE
}

# Defaults for all supported options.
DEBUG=0
DATASET="nextqa"
DEVICE="auto"
OUTPUT_DIR=""

RUN_KEYFRAME_SELECT="0"
FEATURE_DATASET=""
KEYFRAME_DATASET=""
INFERENCE_DATASET=""
INFERENCE_CONFIG=""
DATASET_SLUG=""

RUN_ROOT_DIR=""
UNIQUE_EXP_ID=""

NUM_KEYFRAMES="12"
LAMBDA_EVENT="0.5"
ALPHA_GAP="0.6"
BETA_REDUNDANCY="0.8"
SCORE_NORMALIZER="minmax"
MAX_FRAMES_TO_EXTRACT="5400"
KEYFRAME_JSON=""
KEYFRAME_IMPL="single"

OUTPUT_NAME="predictions"
NUM_FRAMES="6"
PRUNE_MODE="null"
RATE="null"
TOKENS_NUM="936"

FEATURE_OVERRIDES=()
KEYFRAME_OVERRIDES=()
INFERENCE_OVERRIDES=()
declare -A EXPLICIT_FLAGS=()

mark_explicit() {
  EXPLICIT_FLAGS["$1"]=1
}

is_explicit() {
  [[ -n "${EXPLICIT_FLAGS[$1]:-}" ]]
}

restore_logged_array() {
  local -n target_ref="$1"
  local raw_value="$2"
  target_ref=()
  if [[ -n "${raw_value}" ]]; then
    read -r -a target_ref <<< "${raw_value}"
  fi
}

apply_resume_parameter() {
  local key="$1"
  local value="$2"

  case "${key}" in
    UNIQUE_EXP_ID) if ! is_explicit "UNIQUE_EXP_ID"; then UNIQUE_EXP_ID="${value}"; fi ;;
    RUN_ROOT_DIR) if ! is_explicit "RUN_ROOT_DIR"; then RUN_ROOT_DIR="${value}"; fi ;;
    DATASET) if ! is_explicit "DATASET"; then DATASET="${value}"; fi ;;
    FEATURE_DATASET) if ! is_explicit "FEATURE_DATASET"; then FEATURE_DATASET="${value}"; fi ;;
    KEYFRAME_DATASET) if ! is_explicit "KEYFRAME_DATASET"; then KEYFRAME_DATASET="${value}"; fi ;;
    INFERENCE_DATASET) if ! is_explicit "INFERENCE_DATASET"; then INFERENCE_DATASET="${value}"; fi ;;
    INFERENCE_CONFIG) if ! is_explicit "INFERENCE_CONFIG"; then INFERENCE_CONFIG="${value}"; fi ;;
    DATASET_SLUG) if ! is_explicit "DATASET_SLUG"; then DATASET_SLUG="${value}"; fi ;;
    DEVICE) if ! is_explicit "DEVICE"; then DEVICE="${value}"; fi ;;
    RUN_KEYFRAME_SELECT) if ! is_explicit "RUN_KEYFRAME_SELECT"; then RUN_KEYFRAME_SELECT="${value}"; fi ;;
    NUM_KEYFRAMES) if ! is_explicit "NUM_KEYFRAMES"; then NUM_KEYFRAMES="${value}"; fi ;;
    LAMBDA_EVENT) if ! is_explicit "LAMBDA_EVENT"; then LAMBDA_EVENT="${value}"; fi ;;
    ALPHA_GAP) if ! is_explicit "ALPHA_GAP"; then ALPHA_GAP="${value}"; fi ;;
    BETA_REDUNDANCY) if ! is_explicit "BETA_REDUNDANCY"; then BETA_REDUNDANCY="${value}"; fi ;;
    SCORE_NORMALIZER) if ! is_explicit "SCORE_NORMALIZER"; then SCORE_NORMALIZER="${value}"; fi ;;
    MAX_FRAMES_TO_EXTRACT) if ! is_explicit "MAX_FRAMES_TO_EXTRACT"; then MAX_FRAMES_TO_EXTRACT="${value}"; fi ;;
    KEYFRAME_JSON) if ! is_explicit "KEYFRAME_JSON"; then KEYFRAME_JSON="${value}"; fi ;;
    KEYFRAME_IMPL) if ! is_explicit "KEYFRAME_IMPL"; then KEYFRAME_IMPL="${value}"; fi ;;
    OUTPUT_NAME) if ! is_explicit "OUTPUT_NAME"; then OUTPUT_NAME="${value}"; fi ;;
    NUM_FRAMES) if ! is_explicit "NUM_FRAMES"; then NUM_FRAMES="${value}"; fi ;;
    PRUNE_MODE) if ! is_explicit "PRUNE_MODE"; then PRUNE_MODE="${value}"; fi ;;
    RATE) if ! is_explicit "RATE"; then RATE="${value}"; fi ;;
    TOKENS_NUM) if ! is_explicit "TOKENS_NUM"; then TOKENS_NUM="${value}"; fi ;;
    FEATURE_OVERRIDES)
      if ! is_explicit "FEATURE_OVERRIDES"; then
        restore_logged_array FEATURE_OVERRIDES "${value}"
      fi
      ;;
    KEYFRAME_OVERRIDES)
      if ! is_explicit "KEYFRAME_OVERRIDES"; then
        restore_logged_array KEYFRAME_OVERRIDES "${value}"
      fi
      ;;
    INFERENCE_OVERRIDES)
      if ! is_explicit "INFERENCE_OVERRIDES"; then
        restore_logged_array INFERENCE_OVERRIDES "${value}"
      fi
      ;;
  esac
}

load_resume_parameters() {
  local resume_path="$1"
  local line key value

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    value="$(printf '%b' "${value}")"
    apply_resume_parameter "${key}" "${value}"
  done < "${resume_path}"
}

write_logged_parameter() {
  local key="$1"
  local value="$2"
  printf '%s=%q\n' "${key}" "${value}"
}

forbid_override_key() {
  local stage="$1"
  local override_value="$2"
  local forbidden_key="$3"
  if [[ "${override_value}" == "${forbidden_key}="* ]]; then
    echo "Error: ${stage} override '${forbidden_key}=...' is not allowed here because this runner manages experiment output paths." >&2
    exit 1
  fi
}

validate_path_overrides() {
  local override_value
  for override_value in "${KEYFRAME_OVERRIDES[@]}"; do
    forbid_override_key "keyframe" "${override_value}" "save_cluster_path"
    forbid_override_key "keyframe" "${override_value}" "combined_output_path"
  done
  for override_value in "${INFERENCE_OVERRIDES[@]}"; do
    forbid_override_key "inference" "${override_value}" "output_dir"
    forbid_override_key "inference" "${override_value}" "key_frame_path"
  done
}

# Step 1: Parse CLI flags strictly (unknown flags are rejected).
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) require_value "$1" "${2:-}"; DATASET="$2"; mark_explicit "DATASET"; shift 2 ;;
    --device) require_value "$1" "${2:-}"; DEVICE="$2"; mark_explicit "DEVICE"; shift 2 ;;
    --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; mark_explicit "OUTPUT_DIR"; shift 2 ;;

    --run-keyframe-select) require_value "$1" "${2:-}"; RUN_KEYFRAME_SELECT="$2"; mark_explicit "RUN_KEYFRAME_SELECT"; shift 2 ;;
    --feature-dataset) require_value "$1" "${2:-}"; FEATURE_DATASET="$2"; mark_explicit "FEATURE_DATASET"; shift 2 ;;
    --keyframe-dataset) require_value "$1" "${2:-}"; KEYFRAME_DATASET="$2"; mark_explicit "KEYFRAME_DATASET"; shift 2 ;;
    --inference-dataset) require_value "$1" "${2:-}"; INFERENCE_DATASET="$2"; mark_explicit "INFERENCE_DATASET"; shift 2 ;;
    --inference-config) require_value "$1" "${2:-}"; INFERENCE_CONFIG="$2"; mark_explicit "INFERENCE_CONFIG"; shift 2 ;;

    --dataset-slug) require_value "$1" "${2:-}"; DATASET_SLUG="$2"; mark_explicit "DATASET_SLUG"; shift 2 ;;
    --run-root-dir) require_value "$1" "${2:-}"; RUN_ROOT_DIR="$2"; mark_explicit "RUN_ROOT_DIR"; shift 2 ;;
    --unique-exp-id) require_value "$1" "${2:-}"; UNIQUE_EXP_ID="$2"; mark_explicit "UNIQUE_EXP_ID"; shift 2 ;;

    --num-keyframes) require_value "$1" "${2:-}"; NUM_KEYFRAMES="$2"; mark_explicit "NUM_KEYFRAMES"; shift 2 ;;
    --lambda-event) require_value "$1" "${2:-}"; LAMBDA_EVENT="$2"; mark_explicit "LAMBDA_EVENT"; shift 2 ;;
    --alpha-gap) require_value "$1" "${2:-}"; ALPHA_GAP="$2"; mark_explicit "ALPHA_GAP"; shift 2 ;;
    --beta-redundancy) require_value "$1" "${2:-}"; BETA_REDUNDANCY="$2"; mark_explicit "BETA_REDUNDANCY"; shift 2 ;;
    --score-normalizer) require_value "$1" "${2:-}"; SCORE_NORMALIZER="$2"; mark_explicit "SCORE_NORMALIZER"; shift 2 ;;
    --max-frames-to-extract) require_value "$1" "${2:-}"; MAX_FRAMES_TO_EXTRACT="$2"; mark_explicit "MAX_FRAMES_TO_EXTRACT"; shift 2 ;;
    --keyframe-json) require_value "$1" "${2:-}"; KEYFRAME_JSON="$2"; mark_explicit "KEYFRAME_JSON"; shift 2 ;;
    --keyframe-impl) require_value "$1" "${2:-}"; KEYFRAME_IMPL="$2"; mark_explicit "KEYFRAME_IMPL"; shift 2 ;;

    --output-name) require_value "$1" "${2:-}"; OUTPUT_NAME="$2"; mark_explicit "OUTPUT_NAME"; shift 2 ;;
    --num-frames) require_value "$1" "${2:-}"; NUM_FRAMES="$2"; mark_explicit "NUM_FRAMES"; shift 2 ;;
    --prune-mode) require_value "$1" "${2:-}"; PRUNE_MODE="$2"; mark_explicit "PRUNE_MODE"; shift 2 ;;
    --rate) require_value "$1" "${2:-}"; RATE="$2"; mark_explicit "RATE"; shift 2 ;;
    --tokens-num) require_value "$1" "${2:-}"; TOKENS_NUM="$2"; mark_explicit "TOKENS_NUM"; shift 2 ;;

    --feature-override) require_value "$1" "${2:-}"; FEATURE_OVERRIDES+=("$2"); mark_explicit "FEATURE_OVERRIDES"; shift 2 ;;
    --keyframe-override) require_value "$1" "${2:-}"; KEYFRAME_OVERRIDES+=("$2"); mark_explicit "KEYFRAME_OVERRIDES"; shift 2 ;;
    --inference-override) require_value "$1" "${2:-}"; INFERENCE_OVERRIDES+=("$2"); mark_explicit "INFERENCE_OVERRIDES"; shift 2 ;;

    --debug) DEBUG=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) die_unknown_option "$1" ;;
  esac
done

if is_explicit "OUTPUT_DIR"; then
  RESUME_PARAM_LOG_PATH="${OUTPUT_DIR}/chosen_parameters.env"
  if [[ -f "${RESUME_PARAM_LOG_PATH}" ]]; then
    load_resume_parameters "${RESUME_PARAM_LOG_PATH}"
  fi
fi

validate_path_overrides

# Step 2: Enable debug mode if requested.
if [[ "${DEBUG}" == "1" ]]; then
  export HYDRA_FULL_ERROR=1
  export PYTHONFAULTHANDLER=1
  export PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
  set -x
fi

# Step 3: Resolve derived defaults.
export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

FEATURE_DATASET="${FEATURE_DATASET:-${DATASET}}"
FEATURE_DATASET="${FEATURE_DATASET%_temporal_chain}"
KEYFRAME_DATASET="${KEYFRAME_DATASET:-${FEATURE_DATASET}_temporal_chain}"
INFERENCE_DATASET="${INFERENCE_DATASET:-${FEATURE_DATASET}}"
DATASET_SLUG="${DATASET_SLUG:-${FEATURE_DATASET}}"
UNIQUE_EXP_ID="${UNIQUE_EXP_ID:-$(date -u +%Y%m%dT%H%M%SZ)_pid$$}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/${DATASET}/temporal_chain}"

if [[ -n "${OUTPUT_DIR}" ]]; then
  RUN_DIR="${OUTPUT_DIR}"
else
  OUTPUT_DIR="${RUN_ROOT_DIR}/${UNIQUE_EXP_ID}"
  RUN_DIR="${OUTPUT_DIR}"
fi

KEYFRAME_PER_SAMPLE_DIR="${OUTPUT_DIR}/keyframes"
KEYFRAME_JSON="${KEYFRAME_JSON:-${OUTPUT_DIR}/keyframe6_order.json}"
PRED_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}.json"
ACC_PATH="${OUTPUT_DIR}/accuracy.txt"
ACC_JSON_PATH="${OUTPUT_DIR}/accuracy.json"

FEATURE_TENSOR_DIR="dataset-level (from feature/keyframe experiment config)"

OUTPUT_DIR_ABS="$(realpath -m "${OUTPUT_DIR}")"
KEYFRAME_JSON_DIR_ABS="$(realpath -m "$(dirname "${KEYFRAME_JSON}")")"
if [[ "${KEYFRAME_JSON_DIR_ABS}" != "${OUTPUT_DIR_ABS}" ]]; then
  echo "Error: --keyframe-json must point inside the experiment output folder (${OUTPUT_DIR})." >&2
  exit 1
fi

case "${INFERENCE_DATASET}" in
  videomme) DEFAULT_INFERENCE_DATASET="videomme_6keyframe" ;;
  *) DEFAULT_INFERENCE_DATASET="${INFERENCE_DATASET}" ;;
esac
INFERENCE_CONFIG="${INFERENCE_CONFIG:-${DEFAULT_INFERENCE_DATASET}}"

require_bool_01 "--run-keyframe-select" "${RUN_KEYFRAME_SELECT}"

case "${KEYFRAME_IMPL}" in
  single) KEYFRAME_SCRIPT="temporal_chain_rank_keyframes.py" ;;
  multiprocess) KEYFRAME_SCRIPT="temporal_chain_rank_keyframes_multiprocess.py" ;;
  *)
    echo "Error: --keyframe-impl must be either 'single' or 'multiprocess'." >&2
    exit 1
    ;;
esac

# Step 4: Prepare the experiment folder and save a reproducibility log.
HYDRA_LOG_DIR="${RUN_DIR}/hydra_resolved_configs"
mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}" "${HYDRA_LOG_DIR}"

PARAM_LOG_PATH="${RUN_DIR}/chosen_parameters.env"
{
  write_logged_parameter "UNIQUE_EXP_ID" "${UNIQUE_EXP_ID}"
  write_logged_parameter "RUN_DIR" "${RUN_DIR}"
  write_logged_parameter "RUN_ROOT_DIR" "${RUN_ROOT_DIR}"
  write_logged_parameter "RUN_TIMESTAMP_UTC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_logged_parameter "DATASET" "${DATASET}"
  write_logged_parameter "FEATURE_DATASET" "${FEATURE_DATASET}"
  write_logged_parameter "KEYFRAME_DATASET" "${KEYFRAME_DATASET}"
  write_logged_parameter "INFERENCE_DATASET" "${INFERENCE_DATASET}"
  write_logged_parameter "INFERENCE_CONFIG" "${INFERENCE_CONFIG}"
  write_logged_parameter "DATASET_SLUG" "${DATASET_SLUG}"
  write_logged_parameter "DEVICE" "${DEVICE}"
  write_logged_parameter "OUTPUT_DIR" "${OUTPUT_DIR}"
  write_logged_parameter "RUN_KEYFRAME_SELECT" "${RUN_KEYFRAME_SELECT}"
  write_logged_parameter "NUM_KEYFRAMES" "${NUM_KEYFRAMES}"
  write_logged_parameter "LAMBDA_EVENT" "${LAMBDA_EVENT}"
  write_logged_parameter "ALPHA_GAP" "${ALPHA_GAP}"
  write_logged_parameter "BETA_REDUNDANCY" "${BETA_REDUNDANCY}"
  write_logged_parameter "SCORE_NORMALIZER" "${SCORE_NORMALIZER}"
  write_logged_parameter "MAX_FRAMES_TO_EXTRACT" "${MAX_FRAMES_TO_EXTRACT}"
  write_logged_parameter "KEYFRAME_IMPL" "${KEYFRAME_IMPL}"
  write_logged_parameter "FEATURE_TENSOR_DIR" "${FEATURE_TENSOR_DIR}"
  write_logged_parameter "KEYFRAME_PER_SAMPLE_DIR" "${KEYFRAME_PER_SAMPLE_DIR}"
  write_logged_parameter "KEYFRAME_JSON" "${KEYFRAME_JSON}"
  write_logged_parameter "OUTPUT_NAME" "${OUTPUT_NAME}"
  write_logged_parameter "NUM_FRAMES" "${NUM_FRAMES}"
  write_logged_parameter "PRUNE_MODE" "${PRUNE_MODE}"
  write_logged_parameter "RATE" "${RATE}"
  write_logged_parameter "TOKENS_NUM" "${TOKENS_NUM}"
  write_logged_parameter "FEATURE_OVERRIDES" "${FEATURE_OVERRIDES[*]:-}"
  write_logged_parameter "KEYFRAME_OVERRIDES" "${KEYFRAME_OVERRIDES[*]:-}"
  write_logged_parameter "INFERENCE_OVERRIDES" "${INFERENCE_OVERRIDES[*]:-}"
} > "${PARAM_LOG_PATH}"

prepare_experiment_log_files "${RUN_DIR}" "${DATASET_SLUG}_temporal_chain"
enable_current_shell_logging
start_parent_mlflow_run "${RUN_DIR}" "${DATASET_SLUG}_temporal_chain"

# Step 5 (optional): run DINOv2 feature extraction.
if [[ "${RUN_KEYFRAME_SELECT}" == "1" ]]; then
  uv run python extract_frame_features.py \
    --cfg job \
    --resolve \
    experiment="${FEATURE_DATASET}" \
    device="${DEVICE}" \
    "${FEATURE_OVERRIDES[@]}" \
    > "${HYDRA_LOG_DIR}/extract_frame_features.yaml"

  uv run python extract_frame_features.py \
    experiment="${FEATURE_DATASET}" \
    device="${DEVICE}" \
    "${FEATURE_OVERRIDES[@]}"
fi

# Step 6: run temporal-chain keyframe ranking and emit combined keyframe JSON.
uv run python "${KEYFRAME_SCRIPT}" \
  --cfg job \
  --resolve \
  experiment="${KEYFRAME_DATASET}" \
  device="${DEVICE}" \
  save_cluster_path="${KEYFRAME_PER_SAMPLE_DIR}" \
  combined_output_path="${KEYFRAME_JSON}" \
  num_keyframes="${NUM_KEYFRAMES}" \
  lambda_event="${LAMBDA_EVENT}" \
  alpha_gap="${ALPHA_GAP}" \
  beta_redundancy="${BETA_REDUNDANCY}" \
  score_normalizer="${SCORE_NORMALIZER}" \
  max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}" \
  "${KEYFRAME_OVERRIDES[@]}" \
  > "${HYDRA_LOG_DIR}/temporal_chain_rank_keyframes.yaml"

uv run python "${KEYFRAME_SCRIPT}" \
  experiment="${KEYFRAME_DATASET}" \
  device="${DEVICE}" \
  save_cluster_path="${KEYFRAME_PER_SAMPLE_DIR}" \
  combined_output_path="${KEYFRAME_JSON}" \
  num_keyframes="${NUM_KEYFRAMES}" \
  lambda_event="${LAMBDA_EVENT}" \
  alpha_gap="${ALPHA_GAP}" \
  beta_redundancy="${BETA_REDUNDANCY}" \
  score_normalizer="${SCORE_NORMALIZER}" \
  max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}" \
  "${KEYFRAME_OVERRIDES[@]}"

# Step 7: run QA inference using selected keyframes.
uv run python run_inference_multiple_choice_qa.py \
  --cfg job \
  --resolve \
  experiment="${INFERENCE_CONFIG}" \
  device="${DEVICE}" \
  output_dir="${OUTPUT_DIR}" \
  output_name="${OUTPUT_NAME}" \
  key_frame_path="${KEYFRAME_JSON}" \
  num_frames="${NUM_FRAMES}" \
  prune_mode="${PRUNE_MODE}" \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}" \
  "${INFERENCE_OVERRIDES[@]}" \
  > "${HYDRA_LOG_DIR}/run_inference_multiple_choice_qa.yaml"

uv run python run_inference_multiple_choice_qa.py \
  experiment="${INFERENCE_CONFIG}" \
  device="${DEVICE}" \
  output_dir="${OUTPUT_DIR}" \
  output_name="${OUTPUT_NAME}" \
  key_frame_path="${KEYFRAME_JSON}" \
  num_frames="${NUM_FRAMES}" \
  prune_mode="${PRUNE_MODE}" \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}" \
  "${INFERENCE_OVERRIDES[@]}"

echo "Computing accuracy for ${PRED_PATH}"
uv run python eval/compute_accuracy.py "${PRED_PATH}" --json-output "${ACC_JSON_PATH}" | tee "${ACC_PATH}"
echo "Saved accuracy report: ${ACC_PATH}"
