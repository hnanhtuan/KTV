import json
import os

import clip
import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import cluster_and_rank_keyframes as base


def minmax_normalize(values, eps=1e-8):
    """
    Normalize a 1D score vector into [0, 1] using min-max normalization.

    Why we need this:
    - The temporal-chain objective combines multiple terms (cluster score,
      event score, temporal gap, redundancy).
    - These terms naturally live on different numeric scales.
    - Min-max normalization makes each term comparable before weighted sum.

    Args:
    - values: Array-like scores for frames.
    - eps: Small constant to avoid division-by-zero.

    Returns:
    - np.ndarray with the same shape as input, scaled to [0, 1].
      If all values are nearly identical, returns all zeros.
    """
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin + eps)


def compute_event_score(frame_features, num_keyframes):
    """
    Compute query-agnostic event/transition score per frame.

    Formula:
      R_event(t) = ||f_t - f_{t-delta}||_2 + ||f_{t+delta} - f_t||_2

    Intuition:
    - Frames around scene/action changes differ more from neighbors.
    - Higher score means more likely near a temporal transition.

    Delta rule (adaptive window):
    - delta = max(1, floor(N / (10K)))
    - N: number of sampled feature frames for the video
    - K: target keyframe count

    Args:
    - frame_features: np.ndarray [N, D], DINO features.
    - num_keyframes: K in the formula above.

    Returns:
    - np.ndarray [N], transition score per frame.
    """
    n = frame_features.shape[0]
    delta = max(1, n // max(1, (10 * num_keyframes)))
    score = np.zeros(n, dtype=np.float32)
    for t in range(n):
        # Clamp neighborhood indices at sequence boundaries.
        left = max(0, t - delta)
        right = min(n - 1, t + delta)
        score[t] = np.linalg.norm(frame_features[t] - frame_features[left]) + np.linalg.norm(
            frame_features[right] - frame_features[t]
        )
    return score


def temporal_chain_select(
    frame_features,
    num_keyframes=12,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
):
    """
    Select K candidate keyframes using greedy temporal-chain scoring.

    This is Stage 1 (candidate generation) and is query-agnostic.
    It replaces "pick nearest frame to each centroid independently" with
    sequential selection that jointly optimizes:
    - Representativeness (cluster-center closeness)
    - Eventness (local temporal change)
    - Temporal coverage (gap from already selected frames)
    - Redundancy suppression (cosine similarity to selected frames)

    Greedy objective at each step:
      G(t|S) = R_cluster~ + lambda*R_event~ + alpha*T_gap~ - beta*R_red~

    Args:
    - frame_features: np.ndarray [N, D], DINO features.
    - num_keyframes: number of candidate frames to return (K).
    - lambda_event: weight for transition/event term.
    - alpha_gap: weight for temporal coverage term.
    - beta_redundancy: weight for redundancy penalty.

    Returns:
    - list[int]: selected indices in feature-frame index space
      (not yet mapped to original video frame IDs).
    """
    frame_features = np.asarray(frame_features, dtype=np.float32)
    n = frame_features.shape[0]
    if n == 0:
        return []
    if n <= num_keyframes:
        # If there are too few frames, keep all of them.
        return list(range(n))

    k = min(num_keyframes, n)
    # K-means provides global visual grouping; we still use it, but only as
    # one score term, not as final one-shot frame selection.
    kmeans = base.KMeans(
        n_clusters=k, random_state=0, init="k-means++", n_init=10
    ).fit(frame_features)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Representativeness: closer to own centroid => larger score.
    r_cluster = -np.linalg.norm(frame_features - centers[labels], axis=1)
    # Eventness: larger local change => larger score.
    r_event = compute_event_score(frame_features, k)

    # Pre-normalize features for cosine similarity in redundancy term.
    norm_features = frame_features / (
        np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8
    )

    selected = []
    remaining = set(range(n))

    # First frame bootstraps from representativeness + eventness only,
    # because gap and redundancy are undefined when S is empty.
    base_first = minmax_normalize(r_cluster) + lambda_event * minmax_normalize(r_event)
    first = int(np.argmax(base_first))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        candidates = np.array(sorted(remaining), dtype=np.int32)

        c_cluster = minmax_normalize(r_cluster[candidates])
        c_event = minmax_normalize(r_event[candidates])

        # Encourage timeline coverage by preferring frames farther from selected ones.
        c_gap_raw = np.array(
            [min(abs(int(t) - int(s)) for s in selected) for t in candidates],
            dtype=np.float32,
        )
        c_gap = minmax_normalize(c_gap_raw)

        # Redundancy term: max cosine similarity to any already selected frame.
        # High similarity means likely duplicate information -> penalize.
        selected_feats = norm_features[np.array(selected, dtype=np.int32)]
        sims = norm_features[candidates] @ selected_feats.T
        c_red_raw = np.max(sims, axis=1)
        c_red = minmax_normalize(c_red_raw)

        # Greedy temporal-chain score for current step.
        scores = c_cluster + lambda_event * c_event + alpha_gap * c_gap - beta_redundancy * c_red
        best = int(candidates[int(np.argmax(scores))])
        selected.append(best)
        remaining.remove(best)

    return selected


def cluster(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    dataset,
    combined_output_path=None,
    num_keyframes=12,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    max_frames_to_extract=5400,
):
    """
    End-to-end pipeline:
    1) Read QA items and precomputed DINO features.
    2) Stage 1: temporal-chain keyframe candidate selection (query-agnostic).
    3) Map selected feature indices back to original video frame numbers.
    4) Stage 2: CLIP question-aware ranking of those candidates.
    5) Save per-question JSON and optional combined JSON.

    Output format per question (same as original script):
      {question_id: [[frame_index, rank], ...]}
    where rank=0 is the most question-relevant candidate (from CLIP sort).
    """
    os.makedirs(save_cluster_path, exist_ok=True)
    # Load DINO feature tensors generated beforehand (pickle file or folder).
    video_frame_tensor = base.load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path, "r") as f:
        qa_data = json.load(f)

    combined_results = {}
    missing_tensor_count = 0

    for sample in tqdm(qa_data, total=len(qa_data)):
        # Per QA sample metadata.
        video_name = sample["video_name"]
        prompt = sample["question"]
        question_id = sample["question_id"]

        output_path = os.path.join(save_cluster_path, f"{question_id}.json")
        if os.path.exists(output_path):
            # Resume-friendly behavior: if already processed, reuse stored result.
            with open(output_path, "r", encoding="utf-8") as f:
                combined_results.update(json.load(f))
            continue

        # Read video metadata to map feature-frame indices -> original frame indices.
        full_video_path = os.path.join(video_path, video_name)
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        tensor = base.get_tensor_for_video(video_frame_tensor, video_name)
        if tensor is None:
            missing_tensor_count += 1
            print(
                f"missing_feature_tensor question_id={question_id} video_name={video_name}"
            )
            continue

        # Stage 1: temporal-chain selection in feature index space.
        selected_feature_indices = temporal_chain_select(
            tensor,
            num_keyframes=num_keyframes,
            lambda_event=lambda_event,
            alpha_gap=alpha_gap,
            beta_redundancy=beta_redundancy,
        )

        # Convert feature-list indices back to original video frame IDs.
        candidate_frame_indices = [
            base.get_original_frame_number(
                total_frames,
                index,
                fps=fps,
                max_frames_to_extract=max_frames_to_extract,
            )
            for index in selected_feature_indices
        ]

        # If mapping collisions happen, fallback to uniform temporal spacing.
        # This keeps exactly num_keyframes candidates for downstream CLIP ranking.
        if len(set(candidate_frame_indices)) != num_keyframes:
            candidate_frame_indices = [
                int(index)
                for index in np.linspace(
                    0, float(total_frames), num_keyframes, endpoint=False
                )
            ]

        frames_cluster, _ = base.load_video(
            full_video_path,
            candidate_frame_indices,
            start=None,
            end=None,
        )
        # Preprocess candidate frames for CLIP image encoder.
        image_batch = [base.preprocess_clip(frame) for frame in frames_cluster]
        if not image_batch:
            print(f"skip_empty_frames question_id={question_id} video_name={video_name}")
            continue

        # Keep question text within a manageable token span.
        prompt_words = prompt.split(" ")
        if len(prompt_words) > 40:
            prompt = " ".join(prompt_words[10:50])

        # Stage 2: question-aware ranking using CLIP image-text similarity.
        image_input = torch.stack(image_batch).to(base.device)
        with torch.no_grad():
            image_features = base.model_clip.encode_image(image_input)
            text_input = clip.tokenize([prompt]).to(base.device)
            text_features = base.model_clip.encode_text(text_input)

            # Normalize before cosine-like dot-product similarity.
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * image_features @ text_features.T

        # Higher similarity => more relevant frame for this question.
        top_k_count = min(num_keyframes, similarity.numel())
        _, top_k_indices = torch.topk(similarity.squeeze(), top_k_count)
        top_keyframes = [candidate_frame_indices[int(index)] for index in top_k_indices]

        # Store as [frame_index, rank], rank starts from 0 (best).
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

    if combined_output_path:
        # Optionally aggregate all question outputs into one file.
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

    print(
        f"Finished temporal-chain clustering. skipped_missing_tensor={missing_tensor_count} total_questions={len(qa_data)}"
    )


@hydra.main(
    config_path="configs/keyframe_ranking", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    """Hydra entrypoint. Mirrors original script interface and adds optional weights."""
    base.load_clip_model(cfg.device)
    cluster(
        base.resolve_path(cfg.json_path),
        base.resolve_path(cfg.video_path),
        base.resolve_path(cfg.video_frame_tensor_path),
        base.resolve_path(cfg.save_cluster_path),
        cfg.dataset,
        combined_output_path=base.resolve_path(cfg.combined_output_path),
        num_keyframes=getattr(cfg, "num_keyframes", 12),
        lambda_event=getattr(cfg, "lambda_event", 0.5),
        alpha_gap=getattr(cfg, "alpha_gap", 0.6),
        beta_redundancy=getattr(cfg, "beta_redundancy", 0.8),
        max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
    )


if __name__ == "__main__":
    main()
