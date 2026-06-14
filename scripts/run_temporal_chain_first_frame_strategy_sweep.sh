#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/utils/cli_utils.sh
source "${SCRIPT_DIR}/utils/cli_utils.sh"

print_help() {
  cat <<'USAGE'
Usage:
  scripts/run_temporal_chain_first_frame_strategy_sweep.sh [options]

Runs temporal-chain seed-strategy ablations. Each strategy writes its own
keyframe JSON, inference predictions, accuracy report, and contributes to a
summary CSV.

Defaults:
  datasets: nextqa videomme intentqa egoschema
  strategies: cluster_event cluster_only event_only start middle balanced lookahead
  variant: temporal_chain_first_frame_strategy_sweep
  num-keyframes: 6
  num-frames: 6
  enable-query-aware-ranking: 0

Strict options:
  --datasets "<names>"                 Space-separated dataset list.
  --strategies "<names>"               Space-separated first-frame strategies.
  --device <name>                     Device override for all stages. Default: auto
  --run-feature-prep <0|1>            Run frame-feature extraction first. Default: 1
  --skip-existing-keyframes <0|1>     Skip strategy when keyframe JSON exists. Default: 0
  --output-root <path>                Root for outputs. Default: outputs
  --variant-name <name>               Per-dataset output folder.
  --num-keyframes <int>               Temporal-chain selected frame count. Default: 6
  --num-frames <int>                  Inference num_frames value. Default: 6
  --enable-query-aware-ranking <0|1>  Use CLIP ranking inside keyframe JSON. Default: 0
  --lambda-event <float>              Temporal-chain event weight. Default: 0.5
  --alpha-gap <float>                 Temporal-chain temporal-gap weight. Default: 0.6
  --beta-redundancy <float>           Temporal-chain redundancy weight. Default: 0.8
  --score-normalizer <name>           Score normalizer for temporal-chain terms. Default: minmax
  --max-frames-to-extract <int>       Temporal-chain frame cap. Default: 5400
  --seed-pool-size <int>              Lookahead seed candidate count. Default: 16
  --seed-bins <int>                   Lookahead temporal-bin count. Default: 6
  --output-name <name>                Inference output file stem. Default: predictions
  --tokens-num <int>                  Inference token budget. Default: 1872

Optional repeatable Hydra override flags:
  --feature-override <key=value>      Extra override for extract_frame_features.py
  --keyframe-override <key=value>     Extra override for temporal_chain_rank_keyframes_first_frame.py
  --inference-override <key=value>    Extra override for run_inference_multiple_choice_qa.py

Other:
  --debug                             Enable verbose shell/Python debugging
  -h, --help                          Show this help
USAGE
}

DATASETS=(nextqa videomme intentqa egoschema)
STRATEGIES=(cluster_event cluster_only event_only start middle balanced lookahead)
DEVICE="auto"
RUN_FEATURE_PREP="1"
SKIP_EXISTING_KEYFRAMES="0"
OUTPUT_ROOT="outputs"
VARIANT_NAME="temporal_chain_first_frame_strategy_sweep"
NUM_KEYFRAMES="6"
NUM_FRAMES="6"
ENABLE_QUERY_AWARE_RANKING="0"
LAMBDA_EVENT="0.5"
ALPHA_GAP="0.6"
BETA_REDUNDANCY="0.8"
SCORE_NORMALIZER="minmax"
MAX_FRAMES_TO_EXTRACT="5400"
SEED_POOL_SIZE="16"
SEED_BINS="6"
OUTPUT_NAME="predictions"
TOKENS_NUM="1872"
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
    --strategies)
      require_value "$1" "${2:-}"
      read -r -a STRATEGIES <<< "$2"
      shift 2
      ;;
    --device) require_value "$1" "${2:-}"; DEVICE="$2"; shift 2 ;;
    --run-feature-prep) require_value "$1" "${2:-}"; RUN_FEATURE_PREP="$2"; shift 2 ;;
    --skip-existing-keyframes) require_value "$1" "${2:-}"; SKIP_EXISTING_KEYFRAMES="$2"; shift 2 ;;
    --output-root) require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
    --variant-name) require_value "$1" "${2:-}"; VARIANT_NAME="$2"; shift 2 ;;
    --num-keyframes) require_value "$1" "${2:-}"; NUM_KEYFRAMES="$2"; shift 2 ;;
    --num-frames) require_value "$1" "${2:-}"; NUM_FRAMES="$2"; shift 2 ;;
    --enable-query-aware-ranking) require_value "$1" "${2:-}"; ENABLE_QUERY_AWARE_RANKING="$2"; shift 2 ;;
    --lambda-event) require_value "$1" "${2:-}"; LAMBDA_EVENT="$2"; shift 2 ;;
    --alpha-gap) require_value "$1" "${2:-}"; ALPHA_GAP="$2"; shift 2 ;;
    --beta-redundancy) require_value "$1" "${2:-}"; BETA_REDUNDANCY="$2"; shift 2 ;;
    --score-normalizer) require_value "$1" "${2:-}"; SCORE_NORMALIZER="$2"; shift 2 ;;
    --max-frames-to-extract) require_value "$1" "${2:-}"; MAX_FRAMES_TO_EXTRACT="$2"; shift 2 ;;
    --seed-pool-size) require_value "$1" "${2:-}"; SEED_POOL_SIZE="$2"; shift 2 ;;
    --seed-bins) require_value "$1" "${2:-}"; SEED_BINS="$2"; shift 2 ;;
    --output-name) require_value "$1" "${2:-}"; OUTPUT_NAME="$2"; shift 2 ;;
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
require_bool_01 "--enable-query-aware-ranking" "${ENABLE_QUERY_AWARE_RANKING}"

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
  if [[ ! -f "${REPO_ROOT}/configs/keyframe_ranking/experiment/${dataset}_temporal_chain.yaml" ]]; then
    echo "Error: missing temporal-chain keyframe config for dataset '${dataset}'." >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/configs/qa_inference/experiment/${inference_experiment}.yaml" ]]; then
    echo "Error: missing QA inference config '${inference_experiment}' for dataset '${dataset}'." >&2
    exit 1
  fi
}

run_feature_prep_for_dataset() {
  local dataset="$1"
  local dataset_root="$2"
  local hydra_log_dir="${dataset_root}/hydra_resolved_configs"

  mkdir -p "${hydra_log_dir}"

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
}

run_strategy_for_dataset() {
  local dataset="$1"
  local strategy="$2"
  local inference_experiment
  local dataset_root
  local strategy_dir
  local keyframe_dir
  local keyframe_json
  local pred_path
  local acc_path
  local hydra_log_dir

  inference_experiment="$(resolve_inference_experiment "${dataset}")"
  dataset_root="${OUTPUT_ROOT}/${dataset}/${VARIANT_NAME}"
  strategy_dir="${dataset_root}/keyframes_${NUM_KEYFRAMES}/strategy_${strategy}"
  keyframe_dir="${strategy_dir}/per_question_keyframes"
  keyframe_json="${strategy_dir}/keyframe_order.json"
  pred_path="${strategy_dir}/${OUTPUT_NAME}.json"
  acc_path="${strategy_dir}/${OUTPUT_NAME}_accuracy.txt"
  hydra_log_dir="${strategy_dir}/hydra_resolved_configs"

  mkdir -p "${strategy_dir}" "${keyframe_dir}" "${hydra_log_dir}"

  echo "=================================================="
  echo "Dataset: ${dataset}"
  echo "Inference config: ${inference_experiment}"
  echo "Strategy: ${strategy}"
  echo "Strategy dir: ${strategy_dir}"
  echo "Keyframes: ${NUM_KEYFRAMES}"
  echo "Query-aware ranking: ${ENABLE_QUERY_AWARE_RANKING}"
  echo "Weights: l=${LAMBDA_EVENT}, a=${ALPHA_GAP}, b=${BETA_REDUNDANCY}"
  echo "Score normalizer: ${SCORE_NORMALIZER}"
  echo "=================================================="

  if [[ "${SKIP_EXISTING_KEYFRAMES}" == "1" && -f "${keyframe_json}" ]]; then
    echo "Skipping keyframe generation because ${keyframe_json} already exists."
  else
    uv run python temporal_chain_rank_keyframes_first_frame.py \
      --cfg job \
      --resolve \
      experiment="${dataset}_temporal_chain" \
      device="${DEVICE}" \
      save_cluster_path="${keyframe_dir}" \
      combined_output_path="${keyframe_json}" \
      num_keyframes="${NUM_KEYFRAMES}" \
      enable_query_aware_ranking="${ENABLE_QUERY_AWARE_RANKING}" \
      lambda_event="${LAMBDA_EVENT}" \
      alpha_gap="${ALPHA_GAP}" \
      beta_redundancy="${BETA_REDUNDANCY}" \
      score_normalizer="${SCORE_NORMALIZER}" \
      max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}" \
      first_frame_strategy="${strategy}" \
      seed_pool_size="${SEED_POOL_SIZE}" \
      seed_bins="${SEED_BINS}" \
      "${KEYFRAME_OVERRIDES[@]}" \
      > "${hydra_log_dir}/temporal_chain_rank_keyframes_first_frame.yaml"

    uv run python temporal_chain_rank_keyframes_first_frame.py \
      experiment="${dataset}_temporal_chain" \
      device="${DEVICE}" \
      save_cluster_path="${keyframe_dir}" \
      combined_output_path="${keyframe_json}" \
      num_keyframes="${NUM_KEYFRAMES}" \
      enable_query_aware_ranking="${ENABLE_QUERY_AWARE_RANKING}" \
      lambda_event="${LAMBDA_EVENT}" \
      alpha_gap="${ALPHA_GAP}" \
      beta_redundancy="${BETA_REDUNDANCY}" \
      score_normalizer="${SCORE_NORMALIZER}" \
      max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}" \
      first_frame_strategy="${strategy}" \
      seed_pool_size="${SEED_POOL_SIZE}" \
      seed_bins="${SEED_BINS}" \
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
    output_dir="${strategy_dir}" \
    output_name="${OUTPUT_NAME}" \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode=null \
    rate=null \
    tokens_num="${TOKENS_NUM}" \
    "${INFERENCE_OVERRIDES[@]}" \
    > "${hydra_log_dir}/run_inference_multiple_choice_qa.yaml"

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${inference_experiment}" \
    device="${DEVICE}" \
    output_dir="${strategy_dir}" \
    output_name="${OUTPUT_NAME}" \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode=null \
    rate=null \
    tokens_num="${TOKENS_NUM}" \
    "${INFERENCE_OVERRIDES[@]}"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" --json-output "${acc_path%.txt}.json" | tee "${acc_path}"
  echo "Saved keyframes to ${keyframe_json}"
  echo "Saved accuracy report to ${acc_path}"
}

cd "${REPO_ROOT}"

for dataset in "${DATASETS[@]}"; do
  validate_dataset "${dataset}"
done

generate_first_frame_summary() {
  summary_csv="${OUTPUT_ROOT}/temporal_chain_first_frame_strategy_sweep_summary.csv"
  uv run python report_temporal_chain_first_frame_sweep.py     --root "${OUTPUT_ROOT}"     --output-name "${OUTPUT_NAME}"     --output-csv "${summary_csv}"
}

for dataset in "${DATASETS[@]}"; do
  dataset_root="${OUTPUT_ROOT}/${dataset}/${VARIANT_NAME}"
  if [[ "${RUN_FEATURE_PREP}" == "1" ]]; then
    run_with_experiment_logging "${dataset_root}" "${dataset}_${VARIANT_NAME}_feature_prep" run_feature_prep_for_dataset "${dataset}" "${dataset_root}"
  fi

  for strategy in "${STRATEGIES[@]}"; do
    strategy_dir="${dataset_root}/keyframes_${NUM_KEYFRAMES}/strategy_${strategy}"
    run_with_experiment_logging "${strategy_dir}" "${dataset}_${VARIANT_NAME}_${strategy}" run_strategy_for_dataset "${dataset}" "${strategy}"
  done
done

run_with_experiment_logging "${OUTPUT_ROOT}" "temporal_chain_first_frame_strategy_summary" generate_first_frame_summary
