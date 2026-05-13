#!/bin/bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

EXPERIMENT="${EXPERIMENT:-nextqa}"
RUN_PREFIX="${RUN_PREFIX:-nextqa}"
KEYFRAME_JSON="${KEYFRAME_JSON:-outputs/nextqa_keyframe6_order.json}"
RATE="${RATE:-0.2}"
TOKENS_NUM="${TOKENS_NUM:-1872}"

run_and_eval() {
  local variant_name="$1"
  shift

  local output_name="${RUN_PREFIX}_${variant_name}"
  local pred_path="outputs/${output_name}.json"
  local acc_path="outputs/${output_name}_accuracy.txt"

  echo "=================================================="
  echo "Running variant: ${variant_name}"
  echo "=================================================="

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${EXPERIMENT}" \
    output_name="${output_name}" \
    "$@"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" | tee "${acc_path}"
  echo "Saved accuracy report: ${acc_path}"
}

echo "Step 1/2: preparing keyframes for KTV variants"
uv run python keyframe_select_new.py experiment=nextqa
uv run python cluster_keyframe_and_order.py experiment=nextqa

echo "Step 2/2: running inference variants"

# Baseline: no keyframe file -> uniform sampling across num_frames.
run_and_eval "baseline_uniform_frames" \
  key_frame_path=null \
  prune_mode=null \
  rate=null

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
