#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/utils/cli_utils.sh
source "${SCRIPT_DIR}/utils/cli_utils.sh"

print_help() {
  cat <<'USAGE'
Usage:
  scripts/run_temporal_chain_all_datasets.sh [options]

Runs temporal-chain experiments for every available QA dataset:
  frame-feature extraction + temporal-chain keyframe ranking + QA inference.

Defaults:
  datasets: nextqa videomme intentqa egoschema
  variant: temporal_chain_keyframe_selection_l0.7_a0.8_b0.6
  lambda-event (l): 0.7
  alpha-gap (a): 0.8
  beta-redundancy (b): 0.6

Strict options:
  --datasets "<names>"              Space-separated dataset list.
                                   Default: "nextqa videomme intentqa egoschema"
  --device <name>                  Device override for all stages. Default: auto
  --run-feature-prep <0|1>         Run frame-feature extraction before ranking. Default: 1
  --skip-existing-keyframes <0|1>  Skip keyframe generation when combined keyframe JSON already exists. Default: 0
  --output-root <path>             Root for dataset outputs. Default: outputs
  --variant-name <name>            Per-dataset output folder. Default: temporal_chain_keyframe_selection_l0.7_a0.8_b0.6
  --num-keyframes <int>            Temporal-chain candidate count. Default: 12
  --lambda-event <float>           Temporal-chain event weight. Default: 0.7
  --alpha-gap <float>              Temporal-chain temporal-gap weight. Default: 0.8
  --beta-redundancy <float>        Temporal-chain redundancy weight. Default: 0.6
  --score-normalizer <name>        Score normalizer for temporal-chain terms. Default: minmax
  --max-frames-to-extract <int>    Temporal-chain frame cap. Default: 5400
  --keyframe-impl <single|multiprocess>
                                   Temporal-chain script variant. Default: single
  --output-name <name>             Inference output file stem. Default: predictions
  --num-frames <int>               Inference frame count. Default: 6
  --prune-mode <mode>              Inference prune mode. Default: null
  --rate <value>                   Inference pruning rate. Default: null
  --tokens-num <int>               Inference token budget. Default: 936

Optional repeatable Hydra override flags:
  --feature-override <key=value>   Extra override for extract_frame_features.py
  --keyframe-override <key=value>  Extra override for temporal_chain_rank_keyframes.py
  --inference-override <key=value> Extra override for run_inference_multiple_choice_qa.py

Other:
  --debug                          Enable verbose shell/Python debugging
  -h, --help                       Show this help
USAGE
}

DATASETS=(nextqa videomme intentqa egoschema)
DEVICE="auto"
RUN_FEATURE_PREP="1"
SKIP_EXISTING_KEYFRAMES="0"
OUTPUT_ROOT="outputs"
VARIANT_NAME="temporal_chain_keyframe_selection_l0.7_a0.8_b0.6"
NUM_KEYFRAMES="12"
LAMBDA_EVENT="0.7"
ALPHA_GAP="0.8"
BETA_REDUNDANCY="0.6"
SCORE_NORMALIZER="minmax"
MAX_FRAMES_TO_EXTRACT="5400"
KEYFRAME_IMPL="single"
OUTPUT_NAME="predictions"
NUM_FRAMES="6"
PRUNE_MODE="null"
RATE="null"
TOKENS_NUM="936"
DEBUG=0

FEATURE_OVERRIDES=()
KEYFRAME_OVERRIDES=()
INFERENCE_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      require_value "$1" "${2:-}"
      read -r -a DATASETS <<< "$2"
      shift 2
      ;;
    --device) require_value "$1" "${2:-}"; DEVICE="$2"; shift 2 ;;
    --run-feature-prep) require_value "$1" "${2:-}"; RUN_FEATURE_PREP="$2"; shift 2 ;;
    --skip-existing-keyframes) require_value "$1" "${2:-}"; SKIP_EXISTING_KEYFRAMES="$2"; shift 2 ;;
    --output-root) require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
    --variant-name) require_value "$1" "${2:-}"; VARIANT_NAME="$2"; shift 2 ;;
    --num-keyframes) require_value "$1" "${2:-}"; NUM_KEYFRAMES="$2"; shift 2 ;;
    --lambda-event) require_value "$1" "${2:-}"; LAMBDA_EVENT="$2"; shift 2 ;;
    --alpha-gap) require_value "$1" "${2:-}"; ALPHA_GAP="$2"; shift 2 ;;
    --beta-redundancy) require_value "$1" "${2:-}"; BETA_REDUNDANCY="$2"; shift 2 ;;
    --score-normalizer) require_value "$1" "${2:-}"; SCORE_NORMALIZER="$2"; shift 2 ;;
    --max-frames-to-extract) require_value "$1" "${2:-}"; MAX_FRAMES_TO_EXTRACT="$2"; shift 2 ;;
    --keyframe-impl) require_value "$1" "${2:-}"; KEYFRAME_IMPL="$2"; shift 2 ;;
    --output-name) require_value "$1" "${2:-}"; OUTPUT_NAME="$2"; shift 2 ;;
    --num-frames) require_value "$1" "${2:-}"; NUM_FRAMES="$2"; shift 2 ;;
    --prune-mode) require_value "$1" "${2:-}"; PRUNE_MODE="$2"; shift 2 ;;
    --rate) require_value "$1" "${2:-}"; RATE="$2"; shift 2 ;;
    --tokens-num) require_value "$1" "${2:-}"; TOKENS_NUM="$2"; shift 2 ;;
    --feature-override) require_value "$1" "${2:-}"; FEATURE_OVERRIDES+=("$2"); shift 2 ;;
    --keyframe-override) require_value "$1" "${2:-}"; KEYFRAME_OVERRIDES+=("$2"); shift 2 ;;
    --inference-override) require_value "$1" "${2:-}"; INFERENCE_OVERRIDES+=("$2"); shift 2 ;;
    --debug) DEBUG=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) die_unknown_option "$1" ;;
  esac
done

require_bool_01 "--run-feature-prep" "${RUN_FEATURE_PREP}"
require_bool_01 "--skip-existing-keyframes" "${SKIP_EXISTING_KEYFRAMES}"

if [[ "${DEBUG}" == "1" ]]; then
  export HYDRA_FULL_ERROR=1
  export PYTHONFAULTHANDLER=1
  export PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
  set -x
fi

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

resolve_inference_experiment() {
  local dataset="$1"
  case "${dataset}" in
    videomme) echo "videomme_6keyframe" ;;
    *) echo "${dataset}" ;;
  esac
}

case "${KEYFRAME_IMPL}" in
  single) KEYFRAME_SCRIPT="temporal_chain_rank_keyframes.py" ;;
  multiprocess) KEYFRAME_SCRIPT="temporal_chain_rank_keyframes_multiprocess.py" ;;
  *)
    echo "Error: --keyframe-impl must be either 'single' or 'multiprocess'." >&2
    exit 1
    ;;
esac

validate_dataset() {
  local dataset="$1"
  local inference_experiment
  inference_experiment="$(resolve_inference_experiment "${dataset}")"

  if [[ ! -f "${REPO_ROOT}/configs/frame_feature_extraction/experiment/${dataset}.yaml" ]]; then
    echo "Error: missing frame feature config for dataset '${dataset}'." >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/configs/keyframe_ranking/experiment/${dataset}_temporal_chain.yaml" ]]; then
    echo "Error: missing temporal-chain keyframe config for dataset '${dataset}'." >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/configs/qa_inference/experiment/${inference_experiment}.yaml" ]]; then
    echo "Error: missing QA inference config '${inference_experiment}' for dataset '${dataset}'." >&2
    exit 1
  fi
}

run_temporal_chain_for_dataset() {
  local dataset="$1"
  local inference_experiment
  local setting_dir
  local keyframe_json
  local keyframe_dir
  local pred_path
  local acc_path
  local hydra_log_dir

  inference_experiment="$(resolve_inference_experiment "${dataset}")"
  setting_dir="${OUTPUT_ROOT}/${dataset}/${VARIANT_NAME}"
  keyframe_json="${setting_dir}/keyframe6_order.json"
  keyframe_dir="${setting_dir}/keyframes"
  pred_path="${setting_dir}/${OUTPUT_NAME}.json"
  acc_path="${setting_dir}/${OUTPUT_NAME}_accuracy.txt"
  hydra_log_dir="${setting_dir}/hydra_resolved_configs"

  mkdir -p "${setting_dir}" "${keyframe_dir}" "${hydra_log_dir}"

  echo "=================================================="
  echo "Dataset: ${dataset}"
  echo "Inference config: ${inference_experiment}"
  echo "Temporal-chain output dir: ${setting_dir}"
  echo "Keyframe JSON: ${keyframe_json}"
  echo "Predictions: ${pred_path}"
  echo "Weights: l=${LAMBDA_EVENT}, a=${ALPHA_GAP}, b=${BETA_REDUNDANCY}"
  echo "Score normalizer: ${SCORE_NORMALIZER}"
  echo "=================================================="

  if [[ "${SKIP_EXISTING_KEYFRAMES}" == "1" && -f "${keyframe_json}" ]]; then
    echo "Skipping keyframe generation because ${keyframe_json} already exists."
  else
    if [[ "${RUN_FEATURE_PREP}" == "1" ]]; then
      uv run python extract_frame_features.py \
        --cfg job \
        --resolve \
        experiment="${dataset}" \
        device="${DEVICE}" \
        "${FEATURE_OVERRIDES[@]}" \
        > "${hydra_log_dir}/extract_frame_features.yaml"

      uv run python extract_frame_features.py \
        experiment="${dataset}" \
        device="${DEVICE}" \
        "${FEATURE_OVERRIDES[@]}"
    fi

    uv run python "${KEYFRAME_SCRIPT}" \
      --cfg job \
      --resolve \
      experiment="${dataset}_temporal_chain" \
      device="${DEVICE}" \
      save_cluster_path="${keyframe_dir}" \
      combined_output_path="${keyframe_json}" \
      num_keyframes="${NUM_KEYFRAMES}" \
      lambda_event="${LAMBDA_EVENT}" \
      alpha_gap="${ALPHA_GAP}" \
      beta_redundancy="${BETA_REDUNDANCY}" \
      score_normalizer="${SCORE_NORMALIZER}" \
      max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}" \
      "${KEYFRAME_OVERRIDES[@]}" \
      > "${hydra_log_dir}/temporal_chain_rank_keyframes.yaml"

    uv run python "${KEYFRAME_SCRIPT}" \
      experiment="${dataset}_temporal_chain" \
      device="${DEVICE}" \
      save_cluster_path="${keyframe_dir}" \
      combined_output_path="${keyframe_json}" \
      num_keyframes="${NUM_KEYFRAMES}" \
      lambda_event="${LAMBDA_EVENT}" \
      alpha_gap="${ALPHA_GAP}" \
      beta_redundancy="${BETA_REDUNDANCY}" \
      score_normalizer="${SCORE_NORMALIZER}" \
      max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}" \
      "${KEYFRAME_OVERRIDES[@]}"
  fi

  if [[ ! -f "${keyframe_json}" ]]; then
    echo "Error: keyframe JSON not found at ${keyframe_json}." >&2
    exit 1
  fi

  uv run python run_inference_multiple_choice_qa.py \
    --cfg job \
    --resolve \
    experiment="${inference_experiment}" \
    device="${DEVICE}" \
    output_dir="${setting_dir}" \
    output_name="${OUTPUT_NAME}" \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode="${PRUNE_MODE}" \
    rate="${RATE}" \
    tokens_num="${TOKENS_NUM}" \
    "${INFERENCE_OVERRIDES[@]}" \
    > "${hydra_log_dir}/run_inference_multiple_choice_qa.yaml"

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${inference_experiment}" \
    device="${DEVICE}" \
    output_dir="${setting_dir}" \
    output_name="${OUTPUT_NAME}" \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode="${PRUNE_MODE}" \
    rate="${RATE}" \
    tokens_num="${TOKENS_NUM}" \
    "${INFERENCE_OVERRIDES[@]}"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" --json-output "${acc_path%.txt}.json" | tee "${acc_path}"
  echo "Saved temporal-chain keyframes to ${keyframe_json}"
  echo "Saved accuracy report: ${acc_path}"
}

cd "${REPO_ROOT}"

for dataset in "${DATASETS[@]}"; do
  validate_dataset "${dataset}"
done

for dataset in "${DATASETS[@]}"; do
  setting_dir="${OUTPUT_ROOT}/${dataset}/${VARIANT_NAME}"
  run_with_experiment_logging "${setting_dir}" "${dataset}_${VARIANT_NAME}" run_temporal_chain_for_dataset "${dataset}"
done
