#!/bin/bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

DATASETS=(${DATASETS:-nextqa videomme})
TOKEN_LIST=(${TOKEN_LIST:-504 936 1872})
RATE="${RATE:-0.2}"
UPPER_BOUND_NUM_FRAMES_LIST=(${UPPER_BOUND_NUM_FRAMES_LIST:-12 16 20 24 28 32 36 40 44 48})

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

  local run_prefix="${dataset}"
  local output_name
  if [[ "${uses_tokens_num}" -eq 1 ]]; then
    output_name="${run_prefix}_${variant_name}_tokens${tokens_num}"
  else
    output_name="${run_prefix}_${variant_name}"
  fi
  local pred_path="outputs/${output_name}.json"
  local acc_path="outputs/${output_name}_accuracy.txt"

  echo "=================================================="
  echo "Dataset: ${dataset} | Tokens: ${tokens_num} | Variant: ${variant_name}"
  echo "=================================================="

  uv run python run_inference_multiple_choice_qa.py \
    experiment="${dataset}" \
    output_name="${output_name}" \
    "$@"

  echo "Computing accuracy for ${pred_path}"
  uv run python eval/compute_accuracy.py "${pred_path}" | tee "${acc_path}"
  echo "Saved accuracy report: ${acc_path}"
}

# Phase 1: keyframe prep + all non-upper-bound variants.
for dataset in "${DATASETS[@]}"; do
  keyframe_json="outputs/${dataset}_keyframe6_order.json"

  echo "Phase 1: preparing keyframes for ${dataset}"
  uv run python keyframe_select_new.py experiment="${dataset}"
  uv run python cluster_keyframe_and_order.py experiment="${dataset}"

  echo "Phase 1: running non-upper-bound variants for ${dataset}"
  for tokens_num in "${TOKEN_LIST[@]}"; do
    # Baseline: no keyframe file -> uniform sampling across num_frames.
    run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "baseline_uniform_frames" \
      key_frame_path=null \
      prune_mode=null \
      rate=null

    # KTV keyframe-only.
    run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_keyframe_only" \
      key_frame_path="${keyframe_json}" \
      prune_mode=null \
      rate=null

    # KTV token-only variants (uniformly sampled frames + token selection).
    run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_token_only_cls_new_token_sim" \
      key_frame_path=null \
      prune_mode=cls_new_token_sim \
      rate="${RATE}" \
      tokens_num="${tokens_num}"

    run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_token_only_uniform_token" \
      key_frame_path=null \
      prune_mode=uniform_token \
      rate="${RATE}" \
      tokens_num="${tokens_num}"

    # Full KTV variants (keyframe selection + token selection).
    run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_full_cls_new_token_sim" \
      key_frame_path="${keyframe_json}" \
      prune_mode=cls_new_token_sim \
      rate="${RATE}" \
      tokens_num="${tokens_num}"

    run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "ktv_full_uniform_token" \
      key_frame_path="${keyframe_json}" \
      prune_mode=uniform_token \
      rate="${RATE}" \
      tokens_num="${tokens_num}"
  done
done

# Phase 2: run upper-bound variants in a separate dataset/token/frame sweep.
for dataset in "${DATASETS[@]}"; do
  keyframe_json="outputs/${dataset}_keyframe6_order.json"
  echo "Phase 2: running upper-bound variants for ${dataset}"
  for tokens_num in "${TOKEN_LIST[@]}"; do
    for upper_bound_num_frames in "${UPPER_BOUND_NUM_FRAMES_LIST[@]}"; do
      run_and_eval "${dataset}" "${keyframe_json}" "${tokens_num}" "upper_bound_dense_uniform_frames_f${upper_bound_num_frames}" \
        key_frame_path=null \
        prune_mode=null \
        rate=null \
        num_frames="${upper_bound_num_frames}"
    done
  done
done
