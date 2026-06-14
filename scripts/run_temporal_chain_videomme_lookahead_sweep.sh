#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils/cli_utils.sh
source "${SCRIPT_DIR}/utils/cli_utils.sh"

LAMBDAS=(0.2 0.4 0.6 0.8 1.0)
ALPHAS=(0.2 0.4 0.6 0.8 1.0)
BETAS=(0.2 0.4 0.6 0.8 1.0)

RUN_ROOT_DIR="outputs/videomme/temporal_chain_sweeps"
DEVICE="auto"
NUM_KEYFRAMES="12"
NUM_FRAMES="6"
TOKENS_NUM="1872"
MAX_FRAMES_TO_EXTRACT="5400"
FIRST_FRAME_STRATEGY="lookahead"
SEED_POOL_SIZE="16"
SEED_BINS="6"

run_lookahead_setting() {
  local exp_id="$1"
  local output_dir="$2"
  local l="$3"
  local a="$4"
  local b="$5"
  local keyframe_dir="${output_dir}/keyframes"
  local keyframe_json="${output_dir}/keyframe6_order.json"
  local hydra_log_dir="${output_dir}/hydra_resolved_configs"

  echo "[RUN] ${exp_id}"
  mkdir -p "${output_dir}" "${keyframe_dir}" "${hydra_log_dir}"

  uv run python temporal_chain_rank_keyframes_first_frame.py     --cfg job     --resolve     experiment=videomme_temporal_chain     device="${DEVICE}"     save_cluster_path="${keyframe_dir}"     combined_output_path="${keyframe_json}"     num_keyframes="${NUM_KEYFRAMES}"     enable_query_aware_ranking=true     lambda_event="${l}"     alpha_gap="${a}"     beta_redundancy="${b}"     max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}"     first_frame_strategy="${FIRST_FRAME_STRATEGY}"     seed_pool_size="${SEED_POOL_SIZE}"     seed_bins="${SEED_BINS}"     > "${hydra_log_dir}/temporal_chain_rank_keyframes_first_frame.yaml"

  uv run python temporal_chain_rank_keyframes_first_frame.py     experiment=videomme_temporal_chain     device="${DEVICE}"     save_cluster_path="${keyframe_dir}"     combined_output_path="${keyframe_json}"     num_keyframes="${NUM_KEYFRAMES}"     enable_query_aware_ranking=true     lambda_event="${l}"     alpha_gap="${a}"     beta_redundancy="${b}"     max_frames_to_extract="${MAX_FRAMES_TO_EXTRACT}"     first_frame_strategy="${FIRST_FRAME_STRATEGY}"     seed_pool_size="${SEED_POOL_SIZE}"     seed_bins="${SEED_BINS}"

  uv run python run_inference_multiple_choice_qa.py     --cfg job     --resolve     experiment=videomme_6keyframe     device="${DEVICE}"     output_dir="${output_dir}"     output_name=predictions     key_frame_path="${keyframe_json}"     num_frames="${NUM_FRAMES}"     prune_mode=null     rate=null     tokens_num="${TOKENS_NUM}"     > "${hydra_log_dir}/run_inference_multiple_choice_qa.yaml"

  uv run python run_inference_multiple_choice_qa.py     experiment=videomme_6keyframe     device="${DEVICE}"     output_dir="${output_dir}"     output_name=predictions     key_frame_path="${keyframe_json}"     num_frames="${NUM_FRAMES}"     prune_mode=null     rate=null     tokens_num="${TOKENS_NUM}"
}

mkdir -p "${RUN_ROOT_DIR}"

for l in "${LAMBDAS[@]}"; do
  for a in "${ALPHAS[@]}"; do
    for b in "${BETAS[@]}"; do
      exp_id="lookahead_l${l}_a${a}_b${b}_$(date -u +%Y%m%dT%H%M%SZ)"
      output_dir="${RUN_ROOT_DIR}/${exp_id}"
      run_with_experiment_logging "${output_dir}" "${exp_id}" run_lookahead_setting "${exp_id}" "${output_dir}" "${l}" "${a}" "${b}"
    done
  done
done
