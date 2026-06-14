#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/utils/cli_utils.sh
source "${SCRIPT_DIR}/scripts/utils/cli_utils.sh"

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

EXPERIMENT="${EXPERIMENT:-nextqa}"
RUN_PREFIX="${RUN_PREFIX:-nextqa}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${RUN_PREFIX}}"
KEYFRAME_JSON="${KEYFRAME_JSON:-${BASE_OUTPUT_DIR}/keyframe_selection/keyframe6_order.json}"
RATE="${RATE:-0.2}"
TOKENS_NUM="${TOKENS_NUM:-1872}"
UPPER_BOUND_NUM_FRAMES="${UPPER_BOUND_NUM_FRAMES:-48}"

prepare_keyframes() {
  echo "Step 1/2: preparing keyframes for KTV variants"
  uv run python extract_frame_features.py experiment="${EXPERIMENT}"
  uv run python cluster_and_rank_keyframes.py experiment="${EXPERIMENT}"
}

run_and_eval() {
  local variant_name="$1"
  shift

  local uses_tokens_num=0
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == tokens_num=* ]]; then
      uses_tokens_num=1
      break
    fi
  done

  local setting_dir="${BASE_OUTPUT_DIR}/${variant_name}"
  local output_name
  if [[ "${uses_tokens_num}" -eq 1 ]]; then
    output_name="predictions_tokens${TOKENS_NUM}"
  else
    output_name="predictions"
  fi
  local pred_path="${setting_dir}/${output_name}.json"
  local acc_path="${setting_dir}/accuracy.txt"

  mkdir -p "${setting_dir}"

  echo "=================================================="
  echo "Running variant: ${variant_name}"
  echo "=================================================="

  uv run python run_inference_multiple_choice_qa.py     experiment="${EXPERIMENT}"     output_dir="${setting_dir}"     output_name="${output_name}"     "$@"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" --json-output "${acc_path%.txt}.json" | tee "${acc_path}"
  echo "Saved accuracy report: ${acc_path}"
}

run_logged_variant() {
  local variant_name="$1"
  shift
  run_with_experiment_logging "${BASE_OUTPUT_DIR}/${variant_name}" "${RUN_PREFIX}_${variant_name}" run_and_eval "${variant_name}" "$@"
}

run_with_experiment_logging "${BASE_OUTPUT_DIR}/keyframe_selection" "${RUN_PREFIX}_keyframe_prep" prepare_keyframes

echo "Step 2/2: running inference variants"

run_logged_variant "baseline_uniform_frames"   key_frame_path=null   prune_mode=null   rate=null

run_logged_variant "upper_bound_dense_uniform_frames"   key_frame_path=null   prune_mode=null   rate=null   num_frames="${UPPER_BOUND_NUM_FRAMES}"

run_logged_variant "ktv_keyframe_only"   key_frame_path="${KEYFRAME_JSON}"   prune_mode=null   rate=null

run_logged_variant "ktv_token_only_cls_new_token_sim"   key_frame_path=null   prune_mode=cls_new_token_sim   rate="${RATE}"   tokens_num="${TOKENS_NUM}"

run_logged_variant "ktv_token_only_uniform_token"   key_frame_path=null   prune_mode=uniform_token   rate="${RATE}"   tokens_num="${TOKENS_NUM}"

run_logged_variant "ktv_full_cls_new_token_sim"   key_frame_path="${KEYFRAME_JSON}"   prune_mode=cls_new_token_sim   rate="${RATE}"   tokens_num="${TOKENS_NUM}"

run_logged_variant "ktv_full_uniform_token"   key_frame_path="${KEYFRAME_JSON}"   prune_mode=uniform_token   rate="${RATE}"   tokens_num="${TOKENS_NUM}"
