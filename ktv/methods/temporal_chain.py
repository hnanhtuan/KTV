import os
import time
import json
import pickle
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import cv2
import torch
import clip
import multiprocessing as mp
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm

import ktv.methods.clustering as base
from ktv.core.dataset import load_video
from ktv.core.tracking import write_summary_json

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


# --- Normalizers ---
DEFAULT_SCORE_NORMALIZER = "minmax"

SCORE_NORMALIZER_DESCRIPTIONS = {
    "minmax": "Scale scores to [0, 1] with the observed minimum and maximum.",
    "percentile_minmax": "Clip scores to the 5th/95th percentiles before min-max scaling.",
    "zscore_sigmoid": "Standardize by mean/std, then squash to [0, 1] with a sigmoid.",
    "robust_zscore_sigmoid": "Standardize by median/MAD or IQR, then squash with a sigmoid.",
    "rank": "Map scores to percentile ranks, ignoring score magnitude.",
    "softmax": "Map scores to a probability distribution over the current candidates.",
    "log_minmax": "Apply log1p after shifting scores non-negative, then min-max scale.",
}

SCORE_NORMALIZER_ALIASES = {
    "min_max": "minmax",
    "min-max": "minmax",
    "robust_minmax": "percentile_minmax",
    "robust_min_max": "percentile_minmax",
    "percentile": "percentile_minmax",
    "percentile_min_max": "percentile_minmax",
    "zscore": "zscore_sigmoid",
    "z_score": "zscore_sigmoid",
    "z_score_sigmoid": "zscore_sigmoid",
    "robust_zscore": "robust_zscore_sigmoid",
    "robust_z_score": "robust_zscore_sigmoid",
    "robust_z_score_sigmoid": "robust_zscore_sigmoid",
    "rank_percentile": "rank",
    "percentile_rank": "rank",
    "log": "log_minmax",
    "log_min_max": "log_minmax",
}


def normalize_score_normalizer_name(normalizer):
    if normalizer is None:
        normalizer = DEFAULT_SCORE_NORMALIZER
    normalized = str(normalizer).strip().lower().replace("-", "_")
    normalized = SCORE_NORMALIZER_ALIASES.get(normalized, normalized)
    if normalized not in SCORE_NORMALIZER_DESCRIPTIONS:
        valid = ", ".join(sorted(SCORE_NORMALIZER_DESCRIPTIONS))
        raise ValueError(f"Unknown score_normalizer={normalizer!r}. Valid values: {valid}")
    return normalized


def minmax_normalize(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(np.max(values) - np.min(values)) < eps:
        return np.zeros_like(values, dtype=np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    return (values - vmin) / (vmax - vmin + eps)


def percentile_minmax_normalize(values, lower=5.0, upper=95.0, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(np.max(values) - np.min(values)) < eps:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(values, [float(lower), float(upper)])
    if float(hi - lo) < eps:
        return np.zeros_like(values, dtype=np.float32)
    clipped = np.clip(values, lo, hi)
    return ((clipped - lo) / (hi - lo + eps)).astype(np.float32)


def _sigmoid(values):
    values = np.clip(values, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def zscore_sigmoid_normalize(values, temperature=1.0, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.std(values))
    if scale < eps:
        return np.zeros_like(values, dtype=np.float32)
    temperature = max(float(temperature), eps)
    z_scores = (values - float(np.mean(values))) / (scale + eps)
    return _sigmoid(z_scores / temperature)


def robust_zscore_sigmoid_normalize(values, temperature=1.0, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(np.max(values) - np.min(values)) < eps:
        return np.zeros_like(values, dtype=np.float32)

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad

    if scale < eps:
        q25, q75 = np.percentile(values, [25.0, 75.0])
        scale = float(q75 - q25) / 1.349
    if scale < eps:
        scale = float(np.std(values))
    if scale < eps:
        return np.zeros_like(values, dtype=np.float32)

    temperature = max(float(temperature), eps)
    z_scores = (values - center) / (scale + eps)
    return _sigmoid(z_scores / temperature)


def rank_normalize(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    n = values.size
    if n <= 1 or float(np.max(values) - np.min(values)) < eps:
        return np.zeros_like(values, dtype=np.float32)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float32)

    start = 0
    while start < n:
        end = start + 1
        while end < n and np.isclose(sorted_values[end], sorted_values[start], atol=eps, rtol=0.0):
            end += 1
        average_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end

    return ranks / float(n - 1)


def softmax_normalize(values, temperature=1.0, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(np.max(values) - np.min(values)) < eps:
        return np.zeros_like(values, dtype=np.float32)

    temperature = max(float(temperature), eps)
    logits = values / temperature
    logits = logits - float(np.max(logits))
    exp_values = np.exp(logits)
    total = float(np.sum(exp_values))
    if total < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (exp_values / (total + eps)).astype(np.float32)


def log_minmax_normalize(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(np.max(values) - np.min(values)) < eps:
        return np.zeros_like(values, dtype=np.float32)
    shifted = values - float(np.min(values))
    return minmax_normalize(np.log1p(shifted), eps=eps)


def normalize_scores(values, normalizer=DEFAULT_SCORE_NORMALIZER, eps=1e-8):
    normalizer = normalize_score_normalizer_name(normalizer)
    if normalizer == "minmax":
        return minmax_normalize(values, eps=eps)
    if normalizer == "percentile_minmax":
        return percentile_minmax_normalize(values, eps=eps)
    if normalizer == "zscore_sigmoid":
        return zscore_sigmoid_normalize(values, eps=eps)
    if normalizer == "robust_zscore_sigmoid":
        return robust_zscore_sigmoid_normalize(values, eps=eps)
    if normalizer == "rank":
        return rank_normalize(values, eps=eps)
    if normalizer == "softmax":
        return softmax_normalize(values, eps=eps)
    if normalizer == "log_minmax":
        return log_minmax_normalize(values, eps=eps)
    raise AssertionError(f"Unhandled score normalizer: {normalizer}")


# --- Seed Strategies ---
STRATEGY_DESCRIPTIONS = {
    "cluster_event": "Current baseline: maximize representativeness plus eventness.",
    "cluster_only": "Pick the most representative frame before applying the chain.",
    "event_only": "Pick the strongest local temporal transition before applying the chain.",
    "start": "Use the chronological first sampled frame as a lower-bound sanity check.",
    "middle": "Use the temporal midpoint as a neutral coverage anchor.",
    "early_event": "Pick the strongest event inside the first third of the video.",
    "balanced": "Baseline score with a small temporal-center prior to avoid brittle edge seeds.",
    "lookahead": "Try several seed candidates, complete each chain, and keep the best global chain.",
}

_STRATEGY_ALIASES = {
    "baseline": "cluster_event",
    "current": "cluster_event",
    "current_baseline": "cluster_event",
    "representative": "cluster_only",
    "event": "event_only",
    "first": "start",
    "center": "middle",
    "centre": "middle",
    "multi_start": "lookahead",
    "multi-start": "lookahead",
}


@dataclass(frozen=True)
class ChainScoringState:
    r_cluster: np.ndarray
    r_event: np.ndarray
    cluster_score: np.ndarray
    event_score: np.ndarray
    first_score: np.ndarray
    norm_features: np.ndarray


def normalize_strategy_name(strategy: str) -> str:
    normalized = str(strategy).strip().lower().replace("-", "_")
    normalized = _STRATEGY_ALIASES.get(normalized, normalized)
    if normalized not in STRATEGY_DESCRIPTIONS:
        valid = ", ".join(sorted(STRATEGY_DESCRIPTIONS))
        raise ValueError(f"Unknown first_frame_strategy={strategy!r}. Valid values: {valid}")
    return normalized


def compute_event_score(frame_features, num_keyframes):
    n = frame_features.shape[0]
    delta = max(1, n // max(1, (10 * num_keyframes)))
    score = np.zeros(n, dtype=np.float32)
    for t in range(n):
        left = max(0, t - delta)
        right = min(n - 1, t + delta)
        score[t] = np.linalg.norm(frame_features[t] - frame_features[left]) + np.linalg.norm(
            frame_features[right] - frame_features[t]
        )
    return score


def _as_numpy_features(frame_features):
    if hasattr(frame_features, "detach"):
        frame_features = frame_features.detach().cpu().numpy()
    frame_features = np.asarray(frame_features, dtype=np.float32)
    if frame_features.ndim != 2:
        raise ValueError(
            f"Expected frame_features with shape [num_frames, feature_dim], got {frame_features.shape}"
        )
    return frame_features


def _build_scoring_state(frame_features, num_keyframes, lambda_event, score_normalizer):
    kmeans = KMeans(
        n_clusters=num_keyframes, random_state=0, init="k-means++", n_init=10
    ).fit(frame_features)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    r_cluster = -np.linalg.norm(frame_features - centers[labels], axis=1)
    r_event = compute_event_score(frame_features, num_keyframes)
    cluster_score = normalize_scores(r_cluster, score_normalizer)
    event_score = normalize_scores(r_event, score_normalizer)
    first_score = cluster_score + lambda_event * event_score
    norm_features = frame_features / (
        np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8
    )

    return ChainScoringState(
        r_cluster=r_cluster,
        r_event=r_event,
        cluster_score=cluster_score,
        event_score=event_score,
        first_score=first_score,
        norm_features=norm_features,
    )


def _center_prior(num_frames):
    if num_frames <= 1:
        return np.ones(num_frames, dtype=np.float32)
    positions = np.arange(num_frames, dtype=np.float32)
    midpoint = 0.5 * (num_frames - 1)
    max_distance = max(midpoint, 1.0)
    return 1.0 - np.abs(positions - midpoint) / max_distance


def select_first_frame(frame_features, state, strategy):
    del frame_features
    strategy = normalize_strategy_name(strategy)
    n = len(state.first_score)

    if strategy == "cluster_event":
        return int(np.argmax(state.first_score))
    if strategy == "cluster_only":
        return int(np.argmax(state.cluster_score))
    if strategy == "event_only":
        return int(np.argmax(state.event_score))
    if strategy == "start":
        return 0
    if strategy == "middle":
        return int((n - 1) // 2)
    if strategy == "early_event":
        end = max(1, int(np.ceil(n / 3.0)))
        return int(np.argmax(state.event_score[:end]))
    if strategy == "balanced":
        return int(np.argmax(state.first_score + 0.25 * _center_prior(n)))

    raise ValueError("lookahead produces a complete chain, not a single seed")


def _greedy_complete_from_seed(
    seed,
    state,
    num_keyframes,
    lambda_event,
    alpha_gap,
    beta_redundancy,
    score_normalizer,
):
    n = len(state.first_score)
    seed = int(seed)
    if seed < 0 or seed >= n:
        raise ValueError(f"Seed index {seed} is out of bounds for {n} frames")

    selected = [seed]
    remaining = np.ones(n, dtype=bool)
    remaining[seed] = False

    while len(selected) < num_keyframes and np.any(remaining):
        candidates = np.flatnonzero(remaining).astype(np.int32)

        c_cluster = normalize_scores(state.r_cluster[candidates], score_normalizer)
        c_event = normalize_scores(state.r_event[candidates], score_normalizer)

        selected_array = np.asarray(selected, dtype=np.int32)
        c_gap_raw = np.min(
            np.abs(candidates[:, None] - selected_array[None, :]),
            axis=1,
        ).astype(np.float32)
        c_gap = normalize_scores(c_gap_raw, score_normalizer)

        selected_feats = state.norm_features[selected_array]
        sims = state.norm_features[candidates] @ selected_feats.T
        c_red_raw = np.max(sims, axis=1)
        c_red = normalize_scores(c_red_raw, score_normalizer)

        scores = c_cluster + lambda_event * c_event + alpha_gap * c_gap - beta_redundancy * c_red
        best = int(candidates[int(np.argmax(scores))])
        selected.append(best)
        remaining[best] = False

    return selected


def _coverage_score(selected, num_frames):
    if num_frames <= 1:
        return 1.0
    selected_array = np.asarray(selected, dtype=np.int32)
    timeline = np.arange(num_frames, dtype=np.int32)
    min_distances = np.min(
        np.abs(timeline[:, None] - selected_array[None, :]),
        axis=1,
    )
    return 1.0 - float(np.mean(min_distances)) / float(max(1, num_frames - 1))


def _redundancy_score(selected, state):
    if len(selected) <= 1:
        return 0.0
    feats = state.norm_features[np.asarray(selected, dtype=np.int32)]
    sims = feats @ feats.T
    upper = sims[np.triu_indices(len(selected), k=1)]
    return float(np.clip(np.mean(upper), 0.0, 1.0))


def _score_completed_chain(selected, state, lambda_event, alpha_gap, beta_redundancy):
    selected_array = np.asarray(selected, dtype=np.int32)
    quality = float(np.mean(state.cluster_score[selected_array]))
    quality += lambda_event * float(np.mean(state.event_score[selected_array]))
    coverage = _coverage_score(selected, len(state.first_score))
    redundancy = _redundancy_score(selected, state)
    return quality + alpha_gap * coverage - beta_redundancy * redundancy


def _add_top_indices(pool, scores, count):
    if count <= 0:
        return
    for index in np.argsort(scores)[::-1][:count]:
        pool.append(int(index))


def _dedupe_preserve_order(indices):
    seen = set()
    deduped = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        deduped.append(int(index))
    return deduped


def _build_lookahead_seed_pool(state, seed_pool_size, seed_bins):
    n = len(state.first_score)
    seed_pool_size = max(1, min(int(seed_pool_size), n))
    seed_bins = max(1, min(int(seed_bins), n))

    pool = []
    primary_count = max(1, seed_pool_size // 3)
    _add_top_indices(pool, state.first_score, primary_count)

    bin_edges = np.linspace(0, n, seed_bins + 1, dtype=np.int32)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end <= start:
            continue
        local_indices = np.arange(start, end, dtype=np.int32)
        best_local = local_indices[int(np.argmax(state.first_score[local_indices]))]
        pool.append(int(best_local))

    _add_top_indices(pool, state.cluster_score, max(1, seed_pool_size // 4))
    _add_top_indices(pool, state.event_score, max(1, seed_pool_size // 4))
    pool.extend([0, int((n - 1) // 2), n - 1])

    return _dedupe_preserve_order(pool)[:seed_pool_size]


def _select_chain_by_lookahead(
    state,
    num_keyframes,
    lambda_event,
    alpha_gap,
    beta_redundancy,
    score_normalizer,
    seed_pool_size,
    seed_bins,
):
    best_chain = None
    best_score = -np.inf
    for seed in _build_lookahead_seed_pool(state, seed_pool_size, seed_bins):
        chain = _greedy_complete_from_seed(
            seed,
            state,
            num_keyframes,
            lambda_event,
            alpha_gap,
            beta_redundancy,
            score_normalizer,
        )
        score = _score_completed_chain(
            chain,
            state,
            lambda_event,
            alpha_gap,
            beta_redundancy,
        )
        if score > best_score:
            best_score = score
            best_chain = chain
    return best_chain or []


def temporal_chain_select(
    frame_features,
    num_keyframes=12,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    first_frame_strategy="cluster_event",
    seed_pool_size=16,
    seed_bins=6,
    score_normalizer=DEFAULT_SCORE_NORMALIZER,
):
    frame_features = _as_numpy_features(frame_features)
    n = frame_features.shape[0]
    if num_keyframes <= 0:
        return []
    if n == 0:
        return []
    if n <= num_keyframes:
        return list(range(n))

    strategy = normalize_strategy_name(first_frame_strategy)
    score_normalizer = normalize_score_normalizer_name(score_normalizer)
    k = min(int(num_keyframes), n)
    state = _build_scoring_state(frame_features, k, lambda_event, score_normalizer)

    if strategy == "lookahead":
        return _select_chain_by_lookahead(
            state,
            k,
            lambda_event,
            alpha_gap,
            beta_redundancy,
            score_normalizer,
            seed_pool_size,
            seed_bins,
        )

    seed = select_first_frame(frame_features, state, strategy)
    return _greedy_complete_from_seed(
        seed,
        state,
        k,
        lambda_event,
        alpha_gap,
        beta_redundancy,
        score_normalizer,
    )


# --- Multiprocessing Globals and Workers ---
_WORKER_VIDEO_FRAME_TENSOR = None
_WORKER_NUM_KEYFRAMES = 12
_WORKER_LAMBDA_EVENT = 0.5
_WORKER_ALPHA_GAP = 0.6
_WORKER_BETA_REDUNDANCY = 0.8
_WORKER_FIRST_FRAME_STRATEGY = "cluster_event"
_WORKER_SEED_POOL_SIZE = 16
_WORKER_SEED_BINS = 6
_WORKER_SCORE_NORMALIZER = DEFAULT_SCORE_NORMALIZER
_WORKER_MAX_FRAMES_TO_EXTRACT = 5400
_WORKER_BLAS_THREADS = 1


def _limit_threads_context():
    if threadpool_limits is None:
        return None
    return threadpool_limits(limits=max(1, int(_WORKER_BLAS_THREADS)))


def _init_stage1_worker(
    video_frame_tensor,
    num_keyframes,
    lambda_event,
    alpha_gap,
    beta_redundancy,
    first_frame_strategy,
    seed_pool_size,
    seed_bins,
    score_normalizer,
    max_frames_to_extract,
    worker_blas_threads,
):
    global _WORKER_VIDEO_FRAME_TENSOR, _WORKER_NUM_KEYFRAMES, _WORKER_LAMBDA_EVENT, _WORKER_ALPHA_GAP
    global _WORKER_BETA_REDUNDANCY, _WORKER_FIRST_FRAME_STRATEGY, _WORKER_SEED_POOL_SIZE, _WORKER_SEED_BINS
    global _WORKER_SCORE_NORMALIZER, _WORKER_MAX_FRAMES_TO_EXTRACT, _WORKER_BLAS_THREADS

    _WORKER_VIDEO_FRAME_TENSOR = video_frame_tensor
    _WORKER_NUM_KEYFRAMES = num_keyframes
    _WORKER_LAMBDA_EVENT = lambda_event
    _WORKER_ALPHA_GAP = alpha_gap
    _WORKER_BETA_REDUNDANCY = beta_redundancy
    _WORKER_FIRST_FRAME_STRATEGY = normalize_strategy_name(first_frame_strategy)
    _WORKER_SEED_POOL_SIZE = seed_pool_size
    _WORKER_SEED_BINS = seed_bins
    _WORKER_SCORE_NORMALIZER = normalize_score_normalizer_name(score_normalizer)
    _WORKER_MAX_FRAMES_TO_EXTRACT = max_frames_to_extract
    _WORKER_BLAS_THREADS = max(1, int(worker_blas_threads))


def _prepare_stage1_for_video(task):
    video_name, full_video_path = task

    try:
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        tensor = base.get_tensor_for_video(_WORKER_VIDEO_FRAME_TENSOR, video_name)
        if tensor is None:
            return {
                "video_name": video_name,
                "status": "missing_tensor",
                "candidate_frame_indices": None,
                "fps": fps,
                "total_frames": total_frames,
            }

        ctx = _limit_threads_context()
        if ctx is None:
            selected_feature_indices = temporal_chain_select(
                tensor,
                num_keyframes=_WORKER_NUM_KEYFRAMES,
                lambda_event=_WORKER_LAMBDA_EVENT,
                alpha_gap=_WORKER_ALPHA_GAP,
                beta_redundancy=_WORKER_BETA_REDUNDANCY,
                first_frame_strategy=_WORKER_FIRST_FRAME_STRATEGY,
                seed_pool_size=_WORKER_SEED_POOL_SIZE,
                seed_bins=_WORKER_SEED_BINS,
                score_normalizer=_WORKER_SCORE_NORMALIZER,
            )
        else:
            with ctx:
                selected_feature_indices = temporal_chain_select(
                    tensor,
                    num_keyframes=_WORKER_NUM_KEYFRAMES,
                    lambda_event=_WORKER_LAMBDA_EVENT,
                    alpha_gap=_WORKER_ALPHA_GAP,
                    beta_redundancy=_WORKER_BETA_REDUNDANCY,
                    first_frame_strategy=_WORKER_FIRST_FRAME_STRATEGY,
                    seed_pool_size=_WORKER_SEED_POOL_SIZE,
                    seed_bins=_WORKER_SEED_BINS,
                    score_normalizer=_WORKER_SCORE_NORMALIZER,
                )

        candidate_frame_indices = [
            base.get_original_frame_number(
                total_frames,
                index,
                fps=fps,
                max_frames_to_extract=_WORKER_MAX_FRAMES_TO_EXTRACT,
            )
            for index in selected_feature_indices
        ]

        if len(set(candidate_frame_indices)) != _WORKER_NUM_KEYFRAMES:
            candidate_frame_indices = [
                int(index)
                for index in np.linspace(
                    0, float(total_frames), _WORKER_NUM_KEYFRAMES, endpoint=False
                )
            ]

        return {
            "video_name": video_name,
            "status": "ok",
            "candidate_frame_indices": candidate_frame_indices,
            "fps": fps,
            "total_frames": total_frames,
        }
    except Exception as exc:
        return {
            "video_name": video_name,
            "status": "error",
            "candidate_frame_indices": None,
            "error": str(exc),
        }


def run_temporal_chain(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    dataset,
    combined_output_path=None,
    num_keyframes=12,
    enable_query_aware_ranking=True,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    first_frame_strategy="cluster_event",
    seed_pool_size=16,
    seed_bins=6,
    score_normalizer=DEFAULT_SCORE_NORMALIZER,
    max_frames_to_extract=5400,
    num_workers=1,
    worker_blas_threads=1,
    start_method="fork",
    chunksize=None,
):
    start_time = time.time()
    os.makedirs(save_cluster_path, exist_ok=True)
    video_frame_tensor = base.load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Filter already completed json files
    combined_results = {}
    pending_samples = []
    for sample in qa_data:
        question_id = sample["question_id"]
        output_path = os.path.join(save_cluster_path, f"{question_id}.json")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                combined_results.update(json.load(f))
        else:
            pending_samples.append(sample)

    if not pending_samples:
        if combined_output_path:
            combined_output_dir = os.path.dirname(combined_output_path)
            if combined_output_dir:
                os.makedirs(combined_output_dir, exist_ok=True)
            with open(combined_output_path, "w", encoding="utf-8") as f:
                json.dump(combined_results, f, ensure_ascii=False, indent=4)
        return {"saved_count": 0, "skipped_existing_count": len(qa_data), "missing_tensor_count": 0, "skipped_empty_frame_count": 0, "duration_seconds": 0.0}

    # Parallel selection
    video_names = sorted({s["video_name"] for s in pending_samples})
    video_tasks = [(vname, os.path.join(video_path, vname)) for vname in video_names]

    num_workers = min(max(1, num_workers), len(video_tasks))
    print(f"Running keyframe select with {num_workers} worker(s)")

    stage1_results = {}
    if num_workers == 1:
        _init_stage1_worker(
            video_frame_tensor,
            num_keyframes,
            lambda_event,
            alpha_gap,
            beta_redundancy,
            first_frame_strategy,
            seed_pool_size,
            seed_bins,
            score_normalizer,
            max_frames_to_extract,
            worker_blas_threads,
        )
        for task in tqdm(video_tasks, total=len(video_tasks), desc="Stage 1 keyframe select"):
            item = _prepare_stage1_for_video(task)
            stage1_results[item["video_name"]] = item
    else:
        if chunksize is None:
            chunksize = max(1, len(video_tasks) // (num_workers * 4))
        ctx = mp.get_context(start_method)
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_stage1_worker,
            initargs=(
                video_frame_tensor,
                num_keyframes,
                lambda_event,
                alpha_gap,
                beta_redundancy,
                first_frame_strategy,
                seed_pool_size,
                seed_bins,
                score_normalizer,
                max_frames_to_extract,
                worker_blas_threads,
            ),
        ) as pool:
            iterator = pool.imap_unordered(_prepare_stage1_for_video, video_tasks, chunksize=chunksize)
            for item in tqdm(iterator, total=len(video_tasks), desc="Stage 1 keyframe select"):
                stage1_results[item["video_name"]] = item

    # Stage 2 CLIP text-matching ranking
    if enable_query_aware_ranking:
        base.load_clip_model("auto") # load CLIP model
    missing_tensor_count = 0
    skipped_empty_frame_count = 0
    saved_count = 0

    for sample in tqdm(pending_samples, desc="Stage 2 CLIP ranking"):
        video_name = sample["video_name"]
        prompt = sample["question"]
        question_id = sample["question_id"]
        output_path = os.path.join(save_cluster_path, f"{question_id}.json")

        item = stage1_results.get(video_name)
        if not item or item["status"] != "ok":
            if item and item["status"] == "missing_tensor":
                missing_tensor_count += 1
            continue

        candidate_frame_indices = item["candidate_frame_indices"]
        full_video_path = os.path.join(video_path, video_name)

        if not enable_query_aware_ranking:
            key_frame_order = [
                [frame_index, rank]
                for rank, frame_index in enumerate(candidate_frame_indices)
            ]
            result = {question_id: key_frame_order}
            combined_results.update(result)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    default=lambda o: int(o) if isinstance(o, np.integer) else o,
                )
            saved_count += 1
            continue

        frames_cluster, _ = load_video(
            full_video_path,
            candidate_frame_indices,
            start=None,
            end=None,
        )
        image_batch = [base.preprocess_clip(frame) for frame in frames_cluster]
        if not image_batch:
            skipped_empty_frame_count += 1
            continue

        prompt_words = prompt.split(" ")
        if len(prompt_words) > 40:
            prompt = " ".join(prompt_words[10:50])

        image_input = torch.stack(image_batch).to(base.device)
        with torch.no_grad():
            image_features = base.model_clip.encode_image(image_input)
            text_input = clip.tokenize([prompt]).to(base.device)
            text_features = base.model_clip.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * image_features @ text_features.T

        top_k_count = min(num_keyframes, similarity.numel())
        _, top_k_indices = torch.topk(similarity.squeeze(), top_k_count)
        top_keyframes = [candidate_frame_indices[int(index)] for index in top_k_indices]
        key_frame_order = [[frame_index, rank] for rank, frame_index in enumerate(top_keyframes)]

        result = {question_id: key_frame_order}
        combined_results.update(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=4,
                default=lambda o: int(o) if isinstance(o, np.integer) else o,
            )
        saved_count += 1

    if combined_output_path:
        combined_output_dir = os.path.dirname(combined_output_path)
        if combined_output_dir:
            os.makedirs(combined_output_dir, exist_ok=True)
        with open(combined_output_path, "w", encoding="utf-8") as f:
            json.dump(
                combined_results,
                f,
                ensure_ascii=False,
                indent=4,
                default=lambda o: int(o) if isinstance(o, np.integer) else o,
            )

    summary_path = (
        Path(combined_output_path).with_name(f"{Path(combined_output_path).stem}_summary.json")
        if combined_output_path
        else Path(save_cluster_path) / "temporal_chain_keyframe_ranking_summary.json"
    )
    summary = {
        "dataset": dataset,
        "json_path": str(Path(json_path).resolve()),
        "video_path": str(Path(video_path).resolve()),
        "video_frame_tensor_path": str(Path(video_frame_tensor_path).resolve()),
        "save_cluster_path": str(Path(save_cluster_path).resolve()),
        "combined_output_path": str(Path(combined_output_path).resolve()) if combined_output_path else None,
        "summary_path": str(summary_path.resolve()),
        "total_questions": len(qa_data),
        "saved_count": saved_count,
        "skipped_existing_count": len(qa_data) - len(pending_samples),
        "missing_tensor_count": missing_tensor_count,
        "skipped_empty_frame_count": skipped_empty_frame_count,
        "num_keyframes": num_keyframes,
        "lambda_event": lambda_event,
        "alpha_gap": alpha_gap,
        "beta_redundancy": beta_redundancy,
        "first_frame_strategy": first_frame_strategy,
        "score_normalizer": score_normalizer,
        "duration_seconds": time.time() - start_time,
    }
    summary["summary_path"] = write_summary_json(summary_path, summary)
    return summary
