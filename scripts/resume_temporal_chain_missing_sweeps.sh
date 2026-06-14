#!/usr/bin/env bash
set -euo pipefail

# Resume only the temporal-chain sweep settings that are missing on disk.
# This reruns the incomplete combinations for EgoSchema and NextQA.

run_one() {
  local dataset="$1"
  local l="$2"
  local a="$3"
  local b="$4"
  local run_root_dir="outputs/${dataset}/temporal_chain_sweeps"
  local exp_id="l${l}_a${a}_b${b}_$(date -u +%Y%m%dT%H%M%SZ)"

  echo "[RUN] ${dataset} ${exp_id}"

  bash run_temporal_chain_experiments.sh \
    --dataset "${dataset}" \
    --tokens-num 1872 \
    --num-frames 6 \
    --keyframe-impl multiprocess \
    --keyframe-override +num_workers=8 \
    --keyframe-override +worker_blas_threads=1 \
    --lambda-event "${l}" \
    --alpha-gap "${a}" \
    --beta-redundancy "${b}" \
    --run-root-dir "${run_root_dir}" \
    --unique-exp-id "${exp_id}"
}

run_grid() {
  local dataset="$1"
  local l="$2"
  shift 2
  local alpha
  local beta

  for alpha in "$@"; do
    for beta in 0.2 0.4 0.6 0.8 1.0; do
      run_one "${dataset}" "${l}" "${alpha}" "${beta}"
    done
  done
}

run_egoschema() {
  run_one egoschema 0.4 0.8 0.8
  run_one egoschema 0.4 0.8 1.0

  for beta in 0.2 0.4 0.6 0.8 1.0; do
    run_one egoschema 0.4 1.0 "${beta}"
  done

  for l in 0.6 0.8 1.0; do
    for alpha in 0.2 0.4 0.6 0.8 1.0; do
      for beta in 0.2 0.4 0.6 0.8 1.0; do
        run_one egoschema "${l}" "${alpha}" "${beta}"
      done
    done
  done
}

run_nextqa() {
  for beta in 0.4 0.6 0.8 1.0; do
    run_one nextqa 0.2 0.8 "${beta}"
  done

  for beta in 0.2 0.4 0.6 0.8 1.0; do
    run_one nextqa 0.2 1.0 "${beta}"
  done

  for l in 0.4 0.6 0.8 1.0; do
    for alpha in 0.2 0.4 0.6 0.8 1.0; do
      for beta in 0.2 0.4 0.6 0.8 1.0; do
        run_one nextqa "${l}" "${alpha}" "${beta}"
      done
    done
  done
}

run_egoschema
run_nextqa
