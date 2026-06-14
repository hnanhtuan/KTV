#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/utils/cli_utils.sh
source "${SCRIPT_DIR}/utils/cli_utils.sh"

print_help() {
  cat <<'USAGE'
Usage:
  scripts/run_original_ktv_all_datasets.sh [options]

Runs the original KTV setting for every available QA dataset:
  keyframe selection + cls_new_token_sim token selection.

Defaults:
  datasets: nextqa videomme intentqa egoschema
  variant: ktv_full_cls_new_token_sim
  rate: 0.2
  tokens: 1872

Strict options:
  --datasets "<names>"              Space-separated dataset list.
                                   Default: "nextqa videomme intentqa egoschema"
  --device <name>                  Device override for all stages. Default: auto
  --run-keyframe-prep <0|1>        Run feature extraction and keyframe ranking. Default: 1
  --skip-existing-keyframes <0|1>  When prep is enabled, skip datasets whose keyframe JSON exists. Default: 0
  --tokens-num <int>               Inference token budget. Default: 1872
  --rate <value>                   Token-pruning rate. Default: 0.2
  --num-frames <int>               Inference frame count. Default: 6
  --output-root <path>             Root for dataset outputs. Default: outputs
  --variant-name <name>            Output variant folder. Default: ktv_full_cls_new_token_sim

Optional repeatable Hydra override flags:
  --feature-override <key=value>   Extra override for extract_frame_features.py
  --keyframe-override <key=value>  Extra override for cluster_and_rank_keyframes.py
  --inference-override <key=value> Extra override for run_inference_multiple_choice_qa.py

Other:
  --debug                          Enable verbose shell/Python debugging
  -h, --help                       Show this help
USAGE
}

DATASETS=(nextqa videomme intentqa egoschema)
DEVICE="auto"
RUN_KEYFRAME_PREP="1"
SKIP_EXISTING_KEYFRAMES="0"
TOKENS_NUM="1872"
RATE="0.2"
NUM_FRAMES="6"
OUTPUT_ROOT="outputs"
VARIANT_NAME="ktv_full_cls_new_token_sim"
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
    --run-keyframe-prep) require_value "$1" "${2:-}"; RUN_KEYFRAME_PREP="$2"; shift 2 ;;
    --skip-existing-keyframes) require_value "$1" "${2:-}"; SKIP_EXISTING_KEYFRAMES="$2"; shift 2 ;;
    --tokens-num) require_value "$1" "${2:-}"; TOKENS_NUM="$2"; shift 2 ;;
    --rate) require_value "$1" "${2:-}"; RATE="$2"; shift 2 ;;
    --num-frames) require_value "$1" "${2:-}"; NUM_FRAMES="$2"; shift 2 ;;
    --output-root) require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
    --variant-name) require_value "$1" "${2:-}"; VARIANT_NAME="$2"; shift 2 ;;
    --feature-override) require_value "$1" "${2:-}"; FEATURE_OVERRIDES+=("$2"); shift 2 ;;
    --keyframe-override) require_value "$1" "${2:-}"; KEYFRAME_OVERRIDES+=("$2"); shift 2 ;;
    --inference-override) require_value "$1" "${2:-}"; INFERENCE_OVERRIDES+=("$2"); shift 2 ;;
    --debug) DEBUG=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) die_unknown_option "$1" ;;
  esac
done

require_bool_01 "--run-keyframe-prep" "${RUN_KEYFRAME_PREP}"
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

validate_dataset() {
  local dataset="$1"
  local inference_experiment
  inference_experiment="$(resolve_inference_experiment "${dataset}")"

  if [[ ! -f "${REPO_ROOT}/configs/frame_feature_extraction/experiment/${dataset}.yaml" ]]; then
    echo "Error: missing frame feature config for dataset '${dataset}'." >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/configs/keyframe_ranking/experiment/${dataset}.yaml" ]]; then
    echo "Error: missing keyframe ranking config for dataset '${dataset}'." >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/configs/qa_inference/experiment/${inference_experiment}.yaml" ]]; then
    echo "Error: missing QA inference config '${inference_experiment}' for dataset '${dataset}'." >&2
    exit 1
  fi
}

run_original_ktv_for_dataset() {
  local dataset="$1"
  local inference_experiment
  local dataset_output_root
  local keyframe_json
  local setting_dir
  local output_name
  local pred_path
  local acc_path
  local hydra_log_dir

  inference_experiment="$(resolve_inference_experiment "${dataset}")"
  dataset_output_root="${OUTPUT_ROOT}/${dataset}"
  keyframe_json="${dataset_output_root}/keyframe_selection/keyframe6_order.json"
  setting_dir="${dataset_output_root}/${VARIANT_NAME}"
  output_name="predictions_tokens${TOKENS_NUM}"
  pred_path="${setting_dir}/${output_name}.json"
  acc_path="${setting_dir}/accuracy_tokens${TOKENS_NUM}.txt"
  hydra_log_dir="${setting_dir}/hydra_resolved_configs"

  mkdir -p "${setting_dir}" "${hydra_log_dir}"

  echo "=================================================="
  echo "Dataset: ${dataset}"
  echo "Inference config: ${inference_experiment}"
  echo "Original KTV output: ${pred_path}"
  echo "=================================================="

  if [[ "${RUN_KEYFRAME_PREP}" == "1" ]]; then
    if [[ "${SKIP_EXISTING_KEYFRAMES}" == "1" && -f "${keyframe_json}" ]]; then
      echo "Skipping keyframe prep because ${keyframe_json} already exists."
    else
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

      uv run python cluster_and_rank_keyframes.py \
        --cfg job \
        --resolve \
        experiment="${dataset}" \
        device="${DEVICE}" \
        "${KEYFRAME_OVERRIDES[@]}" \
        > "${hydra_log_dir}/cluster_and_rank_keyframes.yaml"

      uv run python cluster_and_rank_keyframes.py \
        experiment="${dataset}" \
        device="${DEVICE}" \
        "${KEYFRAME_OVERRIDES[@]}"
    fi
  fi

  if [[ ! -f "${keyframe_json}" ]]; then
    echo "Error: keyframe JSON not found at ${keyframe_json}." >&2
    echo "Run with --run-keyframe-prep 1, or provide compatible keyframe outputs first." >&2
    exit 1
  fi

  uv run python run_inference_multiple_choice_qa.py \
    --cfg job \
    --resolve \
    experiment="${inference_experiment}" \
    device="${DEVICE}" \
    output_dir="${setting_dir}" \
    output_name="${output_name}" \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode=cls_new_token_sim \
    rate="${RATE}" \
    tokens_num="${TOKENS_NUM}" \
    "${INFERENCE_OVERRIDES[@]}" \
    > "${hydra_log_dir}/run_inference_multiple_choice_qa.yaml"

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${inference_experiment}" \
    device="${DEVICE}" \
    output_dir="${setting_dir}" \
    output_name="${output_name}" \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode=cls_new_token_sim \
    rate="${RATE}" \
    tokens_num="${TOKENS_NUM}" \
    "${INFERENCE_OVERRIDES[@]}"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" --json-output "${acc_path%.txt}.json" | tee "${acc_path}"
  echo "Saved accuracy report: ${acc_path}"
}

cd "${REPO_ROOT}"

for dataset in "${DATASETS[@]}"; do
  validate_dataset "${dataset}"
done

for dataset in "${DATASETS[@]}"; do
  setting_dir="${OUTPUT_ROOT}/${dataset}/${VARIANT_NAME}"
  run_with_experiment_logging "${setting_dir}" "${dataset}_${VARIANT_NAME}" run_original_ktv_for_dataset "${dataset}"
done
