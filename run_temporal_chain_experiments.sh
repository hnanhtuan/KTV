#!/bin/bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

EXPERIMENT="${EXPERIMENT:-nextqa}"
DEVICE="${DEVICE:-auto}"

# Step 1: optional DINO frame-feature preparation.
RUN_KEYFRAME_SELECT="${RUN_KEYFRAME_SELECT:-0}"

# Step 2: temporal-chain keyframe ordering.
NUM_KEYFRAMES="${NUM_KEYFRAMES:-12}"
LAMBDA_EVENT="${LAMBDA_EVENT:-0.5}"
ALPHA_GAP="${ALPHA_GAP:-0.6}"
BETA_REDUNDANCY="${BETA_REDUNDANCY:-0.8}"
MAX_FRAMES_TO_EXTRACT="${MAX_FRAMES_TO_EXTRACT:-5400}"
KEYFRAME_JSON="${KEYFRAME_JSON:-outputs/${EXPERIMENT}_keyframe6_order.json}"

# Step 3: multiple-choice QA inference using the generated keyframes.
case "${EXPERIMENT}" in
  videomme) DEFAULT_INFERENCE_EXPERIMENT="videomme_6keyframe" ;;
  *) DEFAULT_INFERENCE_EXPERIMENT="${EXPERIMENT}" ;;
esac
INFERENCE_EXPERIMENT="${INFERENCE_EXPERIMENT:-${DEFAULT_INFERENCE_EXPERIMENT}}"
OUTPUT_NAME="${OUTPUT_NAME:-${EXPERIMENT}_temporal_chain_qa}"
NUM_FRAMES="${NUM_FRAMES:-6}"
PRUNE_MODE="${PRUNE_MODE:-null}"
RATE="${RATE:-null}"
TOKENS_NUM="${TOKENS_NUM:-936}"

if [[ "${RUN_KEYFRAME_SELECT}" == "1" ]]; then
  uv run python extract_frame_features.py \
    experiment="${EXPERIMENT}" \
    device="${DEVICE}"
fi

uv run python temporal_chain_rank_keyframes.py \
  experiment="${EXPERIMENT}" \
  device="${DEVICE}" \
  combined_output_path="${KEYFRAME_JSON}" \
  num_keyframes="${NUM_KEYFRAMES}" \
  lambda_event="${LAMBDA_EVENT}" \
  alpha_gap="${ALPHA_GAP}" \
  beta_redundancy="${BETA_REDUNDANCY}" \
  max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}"

uv run python run_inference_multiple_choice_qa.py \
  experiment="${INFERENCE_EXPERIMENT}" \
  device="${DEVICE}" \
  output_name="${OUTPUT_NAME}" \
  key_frame_path="${KEYFRAME_JSON}" \
  num_frames="${NUM_FRAMES}" \
  prune_mode="${PRUNE_MODE}" \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}" \
  "$@"
