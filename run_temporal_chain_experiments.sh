#!/bin/bash
set -euo pipefail

# Enable with DEBUG=1 to get command traces and richer Python/Hydra errors.
DEBUG="${DEBUG:-0}"
if [[ "${DEBUG}" == "1" ]]; then
  export HYDRA_FULL_ERROR=1
  export PYTHONFAULTHANDLER=1
  export PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
  set -x
fi


export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

EXPERIMENT="${EXPERIMENT:-nextqa}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXPERIMENT}}"

# Ensure temporal-chain and original KTV share the same extracted DINO features.
# If EXPERIMENT already ends with "_temporal_chain", normalize it back.
FEATURE_EXPERIMENT="${FEATURE_EXPERIMENT:-${EXPERIMENT}}"
FEATURE_EXPERIMENT="${FEATURE_EXPERIMENT%_temporal_chain}"
KEYFRAME_EXPERIMENT="${KEYFRAME_EXPERIMENT:-${FEATURE_EXPERIMENT}_temporal_chain}"
DATASET_SLUG="${DATASET_SLUG:-${FEATURE_EXPERIMENT}}"

# Per-run metadata directory: temporal_chain/<dataset>/<unique_exp_id>
RUN_ROOT_DIR="${RUN_ROOT_DIR:-temporal_chain}"
UNIQUE_EXP_ID="${UNIQUE_EXP_ID:-$(date -u +%Y%m%dT%H%M%SZ)_pid$$}"
RUN_DIR="${RUN_ROOT_DIR}/${DATASET_SLUG}/${UNIQUE_EXP_ID}"

# Step 1: optional DINO frame-feature preparation.
RUN_KEYFRAME_SELECT="${RUN_KEYFRAME_SELECT:-0}"

# Step 2: temporal-chain keyframe ordering.
NUM_KEYFRAMES="${NUM_KEYFRAMES:-12}"
LAMBDA_EVENT="${LAMBDA_EVENT:-0.5}"
ALPHA_GAP="${ALPHA_GAP:-0.6}"
BETA_REDUNDANCY="${BETA_REDUNDANCY:-0.8}"
MAX_FRAMES_TO_EXTRACT="${MAX_FRAMES_TO_EXTRACT:-5400}"
KEYFRAME_JSON="${KEYFRAME_JSON:-${OUTPUT_DIR}/${EXPERIMENT}_temporal_chain_keyframe6_order.json}"

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

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RUN_DIR}"

PARAM_LOG_PATH="${RUN_DIR}/chosen_parameters.env"
{
  echo "UNIQUE_EXP_ID=${UNIQUE_EXP_ID}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "RUN_TIMESTAMP_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "EXPERIMENT=${EXPERIMENT}"
  echo "FEATURE_EXPERIMENT=${FEATURE_EXPERIMENT}"
  echo "KEYFRAME_EXPERIMENT=${KEYFRAME_EXPERIMENT}"
  echo "DATASET_SLUG=${DATASET_SLUG}"
  echo "DEVICE=${DEVICE}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "RUN_KEYFRAME_SELECT=${RUN_KEYFRAME_SELECT}"
  echo "NUM_KEYFRAMES=${NUM_KEYFRAMES}"
  echo "LAMBDA_EVENT=${LAMBDA_EVENT}"
  echo "ALPHA_GAP=${ALPHA_GAP}"
  echo "BETA_REDUNDANCY=${BETA_REDUNDANCY}"
  echo "MAX_FRAMES_TO_EXTRACT=${MAX_FRAMES_TO_EXTRACT}"
  echo "KEYFRAME_JSON=${KEYFRAME_JSON}"
  echo "INFERENCE_EXPERIMENT=${INFERENCE_EXPERIMENT}"
  echo "OUTPUT_NAME=${OUTPUT_NAME}"
  echo "NUM_FRAMES=${NUM_FRAMES}"
  echo "PRUNE_MODE=${PRUNE_MODE}"
  echo "RATE=${RATE}"
  echo "TOKENS_NUM=${TOKENS_NUM}"
  echo "EXTRA_ARGS=$*"
} > "${PARAM_LOG_PATH}"

HYDRA_LOG_DIR="${RUN_DIR}/hydra_resolved_configs"
mkdir -p "${HYDRA_LOG_DIR}"

if [[ "${RUN_KEYFRAME_SELECT}" == "1" ]]; then
  uv run python extract_frame_features.py \
    --cfg job \
    --resolve \
    experiment="${FEATURE_EXPERIMENT}" \
    device="${DEVICE}" \
    > "${HYDRA_LOG_DIR}/extract_frame_features.yaml"

  uv run python extract_frame_features.py \
    experiment="${FEATURE_EXPERIMENT}" \
    device="${DEVICE}"
fi

uv run python temporal_chain_rank_keyframes.py \
  --cfg job \
  --resolve \
  experiment="${KEYFRAME_EXPERIMENT}" \
  device="${DEVICE}" \
  combined_output_path="${KEYFRAME_JSON}" \
  num_keyframes="${NUM_KEYFRAMES}" \
  lambda_event="${LAMBDA_EVENT}" \
  alpha_gap="${ALPHA_GAP}" \
  beta_redundancy="${BETA_REDUNDANCY}" \
  max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}" \
  > "${HYDRA_LOG_DIR}/temporal_chain_rank_keyframes.yaml"

uv run python temporal_chain_rank_keyframes.py \
  experiment="${KEYFRAME_EXPERIMENT}" \
  device="${DEVICE}" \
  combined_output_path="${KEYFRAME_JSON}" \
  num_keyframes="${NUM_KEYFRAMES}" \
  lambda_event="${LAMBDA_EVENT}" \
  alpha_gap="${ALPHA_GAP}" \
  beta_redundancy="${BETA_REDUNDANCY}" \
  max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}"

uv run python run_inference_multiple_choice_qa.py \
  --cfg job \
  --resolve \
  experiment="${INFERENCE_EXPERIMENT}" \
  device="${DEVICE}" \
  output_dir="${OUTPUT_DIR}" \
  output_name="${OUTPUT_NAME}" \
  key_frame_path="${KEYFRAME_JSON}" \
  num_frames="${NUM_FRAMES}" \
  prune_mode="${PRUNE_MODE}" \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}" \
  "$@" \
  > "${HYDRA_LOG_DIR}/run_inference_multiple_choice_qa.yaml"

uv run python run_inference_multiple_choice_qa.py \
  experiment="${INFERENCE_EXPERIMENT}" \
  device="${DEVICE}" \
  output_dir="${OUTPUT_DIR}" \
  output_name="${OUTPUT_NAME}" \
  key_frame_path="${KEYFRAME_JSON}" \
  num_frames="${NUM_FRAMES}" \
  prune_mode="${PRUNE_MODE}" \
  rate="${RATE}" \
  tokens_num="${TOKENS_NUM}" \
  "$@"
