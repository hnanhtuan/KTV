#!/usr/bin/env bash
set -euo pipefail

# Sweep temporal-chain weights for NextQA.
# Edit these arrays to control the search grid.
LAMBDAS=(0.2 0.4 0.6 0.8 1.0)
ALPHAS=(0.2 0.4 0.6 0.8 1.0)
BETAS=(0.2 0.4 0.6 0.8 1.0)

RUN_ROOT_DIR="outputs/nextqa/temporal_chain_sweeps"

mkdir -p "${RUN_ROOT_DIR}"

for l in "${LAMBDAS[@]}"; do
  for a in "${ALPHAS[@]}"; do
    for b in "${BETAS[@]}"; do
      exp_id="l${l}_a${a}_b${b}_$(date -u +%Y%m%dT%H%M%SZ)"
      echo "[RUN] ${exp_id}"

      bash run_temporal_chain_experiments.sh \
        --dataset nextqa \
        --tokens-num 1872 \
        --num-frames 6 \
        --keyframe-impl multiprocess \
        --keyframe-override +num_workers=8 \
        --keyframe-override +worker_blas_threads=1 \
        --lambda-event "${l}" \
        --alpha-gap "${a}" \
        --beta-redundancy "${b}" \
        --run-root-dir "${RUN_ROOT_DIR}" \
        --unique-exp-id "${exp_id}"
    done
  done
done
