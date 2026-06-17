#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=utils/cli_utils.sh
source "${SCRIPT_DIR}/utils/cli_utils.sh"

print_help() {
  cat <<'USAGE'
Usage:
  scripts/run_temporal_chain_clustering_sweep.sh [options]

Runs temporal-chain clustering method sweep for Video-MME (or other datasets).

Defaults:
  datasets: videomme
  strategies: cluster_event
  clustering-methods: kmeans kmedoids_cosine kmedoids_l2 agglomerative_cosine agglomerative_l2
  score-normalizer: minmax
  num-keyframes: 12
  num-frames: 6
  tokens-num: 1872

Strict options:
  --datasets "<names>"                 Space-separated dataset list.
  --strategies "<names>"               Space-separated first-frame strategies. Default: cluster_event
  --clustering-methods "<names>"       Space-separated clustering methods.
  --score-normalizer <name>            Score normalizer to use. Default: minmax
  --device <name>                      Device override for all stages. Default: auto
  --output-root <path>                 Root for outputs. Default: outputs
  --num-keyframes <int>                Temporal-chain selected frame count. Default: 12
  --num-frames <int>                   Inference num_frames value. Default: 6
  --tokens-num <int>                   Inference token budget. Default: 1872
  --num-workers <int>                  Number of workers for keyframe selection. Default: 8
  --worker-blas-threads <int>          BLAS threads per worker. Default: 1
  --skip-existing <0|1>                Skip execution if output accuracy.txt exists. Default: 1

Other:
  --debug                              Enable verbose shell/Python debugging
  -h, --help                           Show this help
USAGE
}

DATASETS=(videomme)
STRATEGIES=(cluster_event)
CLUSTERING_METHODS=(kmeans kmedoids_cosine kmedoids_l2 agglomerative_cosine agglomerative_l2)
SCORE_NORMALIZER="minmax"
DEVICE="auto"
OUTPUT_ROOT="outputs"
NUM_KEYFRAMES="12"
NUM_FRAMES="6"
TOKENS_NUM="1872"
NUM_WORKERS="8"
WORKER_BLAS_THREADS="1"
SKIP_EXISTING="1"
DEBUG=0

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
    --clustering-methods)
      require_value "$1" "${2:-}"
      read -r -a CLUSTERING_METHODS <<< "$2"
      shift 2
      ;;
    --score-normalizer) require_value "$1" "${2:-}"; SCORE_NORMALIZER="$2"; shift 2 ;;
    --device) require_value "$1" "${2:-}"; DEVICE="$2"; shift 2 ;;
    --output-root) require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
    --num-keyframes) require_value "$1" "${2:-}"; NUM_KEYFRAMES="$2"; shift 2 ;;
    --num-frames) require_value "$1" "${2:-}"; NUM_FRAMES="$2"; shift 2 ;;
    --tokens-num) require_value "$1" "${2:-}"; TOKENS_NUM="$2"; shift 2 ;;
    --num-workers) require_value "$1" "${2:-}"; NUM_WORKERS="$2"; shift 2 ;;
    --worker-blas-threads) require_value "$1" "${2:-}"; WORKER_BLAS_THREADS="$2"; shift 2 ;;
    --skip-existing) require_value "$1" "${2:-}"; SKIP_EXISTING="$2"; shift 2 ;;
    --debug) DEBUG=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) die_unknown_option "$1" ;;
  esac
done

require_bool_01 "--skip-existing" "${SKIP_EXISTING}"

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

  if [[ ! -f "${REPO_ROOT}/configs/keyframe_ranking/experiment/${dataset}_temporal_chain.yaml" ]]; then
    echo "Error: missing temporal-chain keyframe config for dataset '${dataset}'." >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/configs/qa_inference/experiment/${inference_experiment}.yaml" ]]; then
    echo "Error: missing QA inference config '${inference_experiment}' for dataset '${dataset}'." >&2
    exit 1
  fi
}

run_setting() {
  local dataset="$1"
  local strategy="$2"
  local method="$3"
  local output_dir="$4"
  local keyframe_dir="${output_dir}/keyframes"
  local keyframe_json="${output_dir}/keyframe6_order.json"
  local hydra_log_dir="${output_dir}/hydra_resolved_configs"
  local inference_experiment
  inference_experiment="$(resolve_inference_experiment "${dataset}")"

  echo "=================================================="
  echo "Dataset: ${dataset}"
  echo "Strategy: ${strategy}"
  echo "Clustering Method: ${method}"
  echo "Output Dir: ${output_dir}"
  echo "=================================================="

  # Inject custom MLflow tags via environment variable so child runs inherit them
  local output_dir_abs
  output_dir_abs="$(realpath -m "${output_dir}")"
  export KTV_MLFLOW_TAGS_JSON="{\"workflow\":\"clustering_sweep\",\"clustering_method\":\"${method}\",\"first_frame_strategy\":\"${strategy}\",\"dataset\":\"${dataset}\",\"output_dir\":\"${output_dir_abs}\"}"

  mkdir -p "${output_dir}" "${keyframe_dir}" "${hydra_log_dir}"

  # Step 1: Multiprocess Keyframe Ranking
  uv run python temporal_chain_rank_keyframes_multiprocess.py \
    --cfg job \
    --resolve \
    experiment="${dataset}_temporal_chain" \
    device="${DEVICE}" \
    save_cluster_path="${keyframe_dir}" \
    combined_output_path="${keyframe_json}" \
    num_keyframes="${NUM_KEYFRAMES}" \
    enable_query_aware_ranking=true \
    lambda_event=0.5 \
    alpha_gap=0.6 \
    beta_redundancy=0.8 \
    score_normalizer="${SCORE_NORMALIZER}" \
    first_frame_strategy="${strategy}" \
    clustering_method="${method}" \
    +num_workers="${NUM_WORKERS}" \
    +worker_blas_threads="${WORKER_BLAS_THREADS}" \
    > "${hydra_log_dir}/temporal_chain_rank_keyframes_multiprocess.yaml"

  uv run python temporal_chain_rank_keyframes_multiprocess.py \
    experiment="${dataset}_temporal_chain" \
    device="${DEVICE}" \
    save_cluster_path="${keyframe_dir}" \
    combined_output_path="${keyframe_json}" \
    num_keyframes="${NUM_KEYFRAMES}" \
    enable_query_aware_ranking=true \
    lambda_event=0.5 \
    alpha_gap=0.6 \
    beta_redundancy=0.8 \
    score_normalizer="${SCORE_NORMALIZER}" \
    first_frame_strategy="${strategy}" \
    clustering_method="${method}" \
    +num_workers="${NUM_WORKERS}" \
    +worker_blas_threads="${WORKER_BLAS_THREADS}"

  # Step 2: QA Inference
  uv run python run_inference_multiple_choice_qa.py \
    --cfg job \
    --resolve \
    experiment="${inference_experiment}" \
    device="${DEVICE}" \
    output_dir="${output_dir}" \
    output_name=predictions \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode=null \
    rate=null \
    tokens_num="${TOKENS_NUM}" \
    > "${hydra_log_dir}/run_inference_multiple_choice_qa.yaml"

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${inference_experiment}" \
    device="${DEVICE}" \
    output_dir="${output_dir}" \
    output_name=predictions \
    key_frame_path="${keyframe_json}" \
    num_frames="${NUM_FRAMES}" \
    prune_mode=null \
    rate=null \
    tokens_num="${TOKENS_NUM}"

  # Step 3: Compute accuracy
  local pred_path="${output_dir}/predictions.json"
  local acc_path="${output_dir}/predictions_accuracy.txt"
  local acc_json_path="${output_dir}/predictions_accuracy.json"
  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" --json-output "${acc_json_path}" | tee "${acc_path}"
}

cd "${REPO_ROOT}"

for dataset in "${DATASETS[@]}"; do
  validate_dataset "${dataset}"
done

for dataset in "${DATASETS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    for method in "${CLUSTERING_METHODS[@]}"; do
      exp_id="clustering_${method}_strat_${strategy}"
      output_dir="${OUTPUT_ROOT}/${dataset}/clustering_sweep/strategy_${strategy}/method_${method}"
      
      if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/predictions_accuracy.txt" && -s "${output_dir}/predictions_accuracy.txt" ]]; then
        echo "[SKIP] ${dataset} strategy=${strategy} method=${method} (already completed)"
        continue
      fi

      run_with_experiment_logging "${output_dir}" "${exp_id}" run_setting "${dataset}" "${strategy}" "${method}" "${output_dir}"
    done
  done
done
