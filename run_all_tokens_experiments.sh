#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/utils/cli_utils.sh
source "${SCRIPT_DIR}/scripts/utils/cli_utils.sh"

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

DATASETS=(${DATASETS:-nextqa videomme})
TOKEN_LIST=(${TOKEN_LIST:-504 936 1872})
RATE="${RATE:-0.2}"
UPPER_BOUND_NUM_FRAMES_LIST=(${UPPER_BOUND_NUM_FRAMES_LIST:-12 16 20 24 28 32 36 40 44 48})

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

prepare_keyframes_for_dataset() {
  local dataset="$1"
  local experiment_name="$2"

  echo "Phase 1: preparing keyframes for ${dataset}"
  uv run python extract_frame_features.py experiment="${experiment_name}"
  uv run python cluster_and_rank_keyframes.py experiment="${experiment_name}"
}

run_and_eval() {
  local dataset="$1"
  local keyframe_json="$2"
  local tokens_num="$3"
  local variant_name="$4"
  shift 4

  local uses_tokens_num=0
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == tokens_num=* ]]; then
      uses_tokens_num=1
      break
    fi
  done

  local experiment_name
  experiment_name="$(resolve_inference_experiment "${dataset}")"
  local setting_dir="outputs/${dataset}/${variant_name}"
  local output_name
  if [[ "${uses_tokens_num}" -eq 1 ]]; then
    output_name="predictions_tokens${tokens_num}"
  else
    output_name="predictions"
  fi
  local pred_path="${setting_dir}/${output_name}.json"
  local acc_path="${setting_dir}/accuracy.txt"

  mkdir -p "${setting_dir}"

  echo "=================================================="
  echo "Dataset: ${dataset} | Tokens: ${tokens_num} | Variant: ${variant_name}"
  echo "=================================================="

  uv run python run_inference_multiple_choice_qa.py     experiment="${experiment_name}"     output_dir="${setting_dir}"     output_name="${output_name}"     "$@"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" --json-output "${acc_path%.txt}.json" | tee "${acc_path}"
  echo "Saved accuracy report: ${acc_path}"
}

for dataset in "${DATASETS[@]}"; do
  experiment_name="$(resolve_keyframe_experiment "${dataset}")"
  keyframe_json="outputs/${dataset}/keyframe_selection/keyframe6_order.json"

  run_with_experiment_logging "outputs/${dataset}/keyframe_selection" "${dataset}_keyframe_prep" prepare_keyframes_for_dataset "${dataset}" "${experiment_name}"

  echo "Phase 1: running non-upper-bound variants for ${dataset}"
  for tokens_num in "${TOKEN_LIST[@]}"; do
    run_with_experiment_logging "outputs/${dataset}/baseline_uniform_frames" "${dataset}_baseline_uniform_frames" run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "baseline_uniform_frames"       key_frame_path=null       prune_mode=null       rate=null

    run_with_experiment_logging "outputs/${dataset}/ktv_keyframe_only" "${dataset}_ktv_keyframe_only" run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_keyframe_only"       key_frame_path="${keyframe_json}"       prune_mode=null       rate=null

    run_with_experiment_logging "outputs/${dataset}/ktv_token_only_cls_new_token_sim" "${dataset}_ktv_token_only_cls_new_token_sim" run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_token_only_cls_new_token_sim"       key_frame_path=null       prune_mode=cls_new_token_sim       rate="${RATE}"       tokens_num="${tokens_num}"

    run_with_experiment_logging "outputs/${dataset}/ktv_token_only_uniform_token" "${dataset}_ktv_token_only_uniform_token" run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_token_only_uniform_token"       key_frame_path=null       prune_mode=uniform_token       rate="${RATE}"       tokens_num="${tokens_num}"

    run_with_experiment_logging "outputs/${dataset}/ktv_full_cls_new_token_sim" "${dataset}_ktv_full_cls_new_token_sim" run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_full_cls_new_token_sim"       key_frame_path="${keyframe_json}"       prune_mode=cls_new_token_sim       rate="${RATE}"       tokens_num="${tokens_num}"

    run_with_experiment_logging "outputs/${dataset}/ktv_full_uniform_token" "${dataset}_ktv_full_uniform_token" run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_full_uniform_token"       key_frame_path="${keyframe_json}"       prune_mode=uniform_token       rate="${RATE}"       tokens_num="${tokens_num}"
  done
done

for dataset in "${DATASETS[@]}"; do
  keyframe_json="outputs/${dataset}/keyframe_selection/keyframe6_order.json"
  echo "Phase 2: running upper-bound variants for ${dataset}"
  for tokens_num in "${TOKEN_LIST[@]}"; do
    for upper_bound_num_frames in "${UPPER_BOUND_NUM_FRAMES_LIST[@]}"; do
      run_with_experiment_logging "outputs/${dataset}/upper_bound_dense_uniform_frames_f${upper_bound_num_frames}" "${dataset}_upper_bound_dense_uniform_frames_f${upper_bound_num_frames}" run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "upper_bound_dense_uniform_frames_f${upper_bound_num_frames}"         key_frame_path=null         prune_mode=null         rate=null         num_frames="${upper_bound_num_frames}"
    done
  done
done
