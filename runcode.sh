#!/bin/bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

EXPERIMENT="${EXPERIMENT:-nextqa}"
RUN_PREFIX="${RUN_PREFIX:-nextqa}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${RUN_PREFIX}}"
KEYFRAME_JSON="${KEYFRAME_JSON:-${BASE_OUTPUT_DIR}/keyframe_selection/keyframe6_order.json}"
RATE="${RATE:-0.2}"
TOKENS_NUM="${TOKENS_NUM:-1872}"
UPPER_BOUND_NUM_FRAMES="${UPPER_BOUND_NUM_FRAMES:-48}"

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

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${EXPERIMENT}" \
    output_dir="${setting_dir}" \
    output_name="${output_name}" \
    "$@"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" | tee "${acc_path}"
  echo "Saved accuracy report: ${acc_path}"
}

echo "Step 1/2: preparing keyframes for KTV variants"
uv run python extract_frame_features.py experiment="${EXPERIMENT}"
uv run python cluster_and_rank_keyframes.py experiment="${EXPERIMENT}"

echo "Step 2/2: running inference variants"

# Baseline: no keyframe file -> uniform sampling across num_frames.
run_and_eval "baseline_uniform_frames" \
  key_frame_path=null \
  prune_mode=null \
  rate=null

# Upper-bound proxy: denser uniform frame sampling without keyframe/token pruning.
run_and_eval "upper_bound_dense_uniform_frames" \
  key_frame_path=null \
  prune_mode=null \
  rate=null \
  num_frames="${UPPER_BOUND_NUM_FRAMES}"

# KTV keyframe-only.
run_and_eval "ktv_keyframe_only" \
  key_frame_path="${KEYFRAME_JSON}" \
  prune_mode=null \
  rate=null

# KTV token-only variants (uniformly sampled frames + token selection).
run_and_eval "ktv_token_only_cls_new_token_sim" \
  key_frame_path=null \
  prune_mode=cls_new_token_sim \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}"

run_and_eval "ktv_token_only_uniform_token" \
  key_frame_path=null \
  prune_mode=uniform_token \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}"

# Full KTV variants (keyframe selection + token selection).
run_and_eval "ktv_full_cls_new_token_sim" \
  key_frame_path="${KEYFRAME_JSON}" \
  prune_mode=cls_new_token_sim \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}"

run_and_eval "ktv_full_uniform_token" \
  key_frame_path="${KEYFRAME_JSON}" \
  prune_mode=uniform_token \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}"
