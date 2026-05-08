#!/bin/bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/ktv_hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

KEYFRAME_JSON="outputs/nextqa_keyframe6_order.json"

uv run python keyframe_select_new.py experiment=nextqa

uv run python cluster_keyframe_and_order.py experiment=nextqa

uv run python run_inference_multiple_choice_qa.py \
  experiment=nextqa_test_cpu \
  key_frame_path="$KEYFRAME_JSON" \
  prune_mode=cls_new_token_sim \
  rate=0.2 \
  tokens_num=1872
