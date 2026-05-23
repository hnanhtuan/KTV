import json
import multiprocessing as mp
import os
import shutil
import tempfile

import clip
import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import cluster_and_rank_keyframes as base

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


_WORKER_VIDEO_FRAME_TENSOR = None
_WORKER_NUM_KEYFRAMES = 12
_WORKER_LAMBDA_EVENT = 0.5
_WORKER_ALPHA_GAP = 0.6
_WORKER_BETA_REDUNDANCY = 0.8
_WORKER_MAX_FRAMES_TO_EXTRACT = 5400
_WORKER_BLAS_THREADS = 1


def minmax_normalize(values, eps=1e-8):
    """Normalize a 1D score vector into [0, 1] using min-max normalization."""
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin + eps)


def compute_event_score(frame_features, num_keyframes):
    """Compute query-agnostic event/transition score per frame."""
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
    """
    frame_features = np.asarray(frame_features, dtype=np.float32)
    n = frame_features.shape[0]
    if n == 0:
        return []
    if n <= num_keyframes:
        return list(range(n))

    k = min(num_keyframes, n)
    kmeans = base.KMeans(
        n_clusters=k, random_state=0, init="k-means++", n_init=10
    ).fit(frame_features)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    r_cluster = -np.linalg.norm(frame_features - centers[labels], axis=1)
    r_event = compute_event_score(frame_features, k)

    norm_features = frame_features / (
        np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8
    )

    selected = []
    remaining = set(range(n))

    base_first = minmax_normalize(r_cluster) + lambda_event * minmax_normalize(r_event)
    first = int(np.argmax(base_first))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        candidates = np.array(sorted(remaining), dtype=np.int32)
        c_cluster = minmax_normalize(r_cluster[candidates])
        c_event = minmax_normalize(r_event[candidates])

        c_gap_raw = np.array(
            [min(abs(int(t) - int(s)) for s in selected) for t in candidates],
            dtype=np.float32,
        )
        c_gap = minmax_normalize(c_gap_raw)

        selected_feats = norm_features[np.array(selected, dtype=np.int32)]
        sims = norm_features[candidates] @ selected_feats.T
        c_red_raw = np.max(sims, axis=1)
        c_red = minmax_normalize(c_red_raw)

        scores = c_cluster + lambda_event * c_event + alpha_gap * c_gap - beta_redundancy * c_red
        best = int(candidates[int(np.argmax(scores))])
        selected.append(best)
        remaining.remove(best)

    return selected


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
    max_frames_to_extract,
    worker_blas_threads,
):
    global _WORKER_VIDEO_FRAME_TENSOR
    global _WORKER_NUM_KEYFRAMES
    global _WORKER_LAMBDA_EVENT
    global _WORKER_ALPHA_GAP
    global _WORKER_BETA_REDUNDANCY
    global _WORKER_MAX_FRAMES_TO_EXTRACT
    global _WORKER_BLAS_THREADS

    _WORKER_VIDEO_FRAME_TENSOR = video_frame_tensor
    _WORKER_NUM_KEYFRAMES = num_keyframes
    _WORKER_LAMBDA_EVENT = lambda_event
    _WORKER_ALPHA_GAP = alpha_gap
    _WORKER_BETA_REDUNDANCY = beta_redundancy
    _WORKER_MAX_FRAMES_TO_EXTRACT = max_frames_to_extract
    _WORKER_BLAS_THREADS = max(1, int(worker_blas_threads))


def _prepare_stage1_for_video(task):
    video_name, full_video_path = task

    try:
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
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
            )
        else:
            with ctx:
                selected_feature_indices = temporal_chain_select(
                    tensor,
                    num_keyframes=_WORKER_NUM_KEYFRAMES,
                    lambda_event=_WORKER_LAMBDA_EVENT,
                    alpha_gap=_WORKER_ALPHA_GAP,
                    beta_redundancy=_WORKER_BETA_REDUNDANCY,
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


def _resolve_num_workers(requested_workers):
    if requested_workers is None:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, 8))
    return max(1, int(requested_workers))


def _resolve_blas_threads(requested_threads):
    if requested_threads is None:
        return 1
    return max(1, int(requested_threads))


def _resolve_chunksize(video_count, num_workers, requested_chunksize):
    if requested_chunksize is not None:
        return max(1, int(requested_chunksize))
    if video_count <= 0:
        return 1
    return max(1, video_count // max(1, num_workers * 4))


def _resolve_start_method(requested_method):
    if requested_method:
        return requested_method

    available_methods = mp.get_all_start_methods()
    if "fork" in available_methods:
        return "fork"
    return "spawn"


def _load_existing_results(qa_data, save_cluster_path):
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

    return combined_results, pending_samples


def _build_video_tasks(pending_samples, video_path):
    video_names = sorted({sample["video_name"] for sample in pending_samples})
    return [
        (video_name, os.path.join(video_path, video_name))
        for video_name in video_names
    ]


def _run_stage1_parallel(
    video_tasks,
    video_frame_tensor,
    num_keyframes,
    lambda_event,
    alpha_gap,
    beta_redundancy,
    max_frames_to_extract,
    num_workers,
    worker_blas_threads,
    start_method,
    chunksize,
):
    if not video_tasks:
        return {}

    num_workers = min(max(1, num_workers), len(video_tasks))
    if num_workers == 1:
        _init_stage1_worker(
            video_frame_tensor,
            num_keyframes,
            lambda_event,
            alpha_gap,
            beta_redundancy,
            max_frames_to_extract,
            worker_blas_threads,
        )
        results = {}
        for task in tqdm(video_tasks, total=len(video_tasks), desc="Stage 1 keyframe select"):
            item = _prepare_stage1_for_video(task)
            results[item["video_name"]] = item
        return results

    ctx = mp.get_context(start_method)
    results = {}
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_stage1_worker,
        initargs=(
            video_frame_tensor,
            num_keyframes,
            lambda_event,
            alpha_gap,
            beta_redundancy,
            max_frames_to_extract,
            worker_blas_threads,
        ),
    ) as pool:
        iterator = pool.imap_unordered(_prepare_stage1_for_video, video_tasks, chunksize=chunksize)
        for item in tqdm(iterator, total=len(video_tasks), desc="Stage 1 keyframe select"):
            results[item["video_name"]] = item
    return results


def _trim_prompt(prompt):
    prompt_words = prompt.split(" ")
    if len(prompt_words) > 40:
        return " ".join(prompt_words[10:50])
    return prompt


def _feature_path_for_video(feature_dir, video_name):
    safe_name = video_name.replace(os.sep, "__").replace("/", "__")
    return os.path.join(feature_dir, f"{safe_name}.pt")


def _ordered_candidate_frame_indices(full_video_path, candidate_frame_indices):
    if os.path.isdir(full_video_path):
        return list(candidate_frame_indices)
    return sorted(candidate_frame_indices)


def _flush_image_feature_batch(batch_items, image_batch, feature_buffers):
    if not image_batch:
        return

    image_input = torch.stack(image_batch).to(base.device)
    with torch.no_grad():
        features = base.model_clip.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)

    features = features.detach().cpu()
    for (video_name, frame_index), feature in zip(batch_items, features):
        feature_buffers[video_name]["frame_indices"].append(int(frame_index))
        feature_buffers[video_name]["features"].append(feature)

    batch_items.clear()
    image_batch.clear()


def _precompute_candidate_image_features(
    stage1_results,
    pending_video_names,
    video_path,
    feature_dir,
    image_batch_size,
):
    image_batch_size = max(1, int(image_batch_size))
    batch_items = []
    image_batch = []
    feature_buffers = {}
    feature_index = {}
    skipped_count = 0

    for video_name in tqdm(
        pending_video_names,
        total=len(pending_video_names),
        desc="Precompute CLIP image features",
    ):
        stage1_item = stage1_results.get(video_name)
        if not stage1_item or stage1_item["status"] != "ok":
            continue

        full_video_path = os.path.join(video_path, video_name)
        candidate_frame_indices = _ordered_candidate_frame_indices(
            full_video_path, stage1_item["candidate_frame_indices"]
        )
        frames_cluster, _ = base.load_video(
            full_video_path,
            candidate_frame_indices,
            start=None,
            end=None,
        )
        if not frames_cluster:
            skipped_count += 1
            continue

        feature_buffers[video_name] = {"frame_indices": [], "features": []}
        for frame_index, frame in zip(candidate_frame_indices, frames_cluster):
            image_batch.append(base.preprocess_clip(frame))
            batch_items.append((video_name, frame_index))
            if len(image_batch) >= image_batch_size:
                _flush_image_feature_batch(batch_items, image_batch, feature_buffers)

    _flush_image_feature_batch(batch_items, image_batch, feature_buffers)

    for video_name, buffer in feature_buffers.items():
        if not buffer["features"]:
            skipped_count += 1
            continue
        feature_path = _feature_path_for_video(feature_dir, video_name)
        payload = {
            "frame_indices": buffer["frame_indices"],
            "image_features": torch.stack(buffer["features"]),
        }
        torch.save(payload, feature_path)
        feature_index[video_name] = feature_path

    return feature_index, skipped_count


def _encode_question_text_features(samples, text_batch_size):
    text_batch_size = max(1, int(text_batch_size))
    text_features_by_question = {}

    for start in tqdm(
        range(0, len(samples), text_batch_size),
        total=(len(samples) + text_batch_size - 1) // text_batch_size,
        desc="Precompute CLIP text features",
    ):
        batch = samples[start : start + text_batch_size]
        prompts = [_trim_prompt(sample["question"]) for sample in batch]
        text_input = clip.tokenize(prompts).to(base.device)
        with torch.no_grad():
            text_features = base.model_clip.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        for sample, feature in zip(batch, text_features.detach().cpu()):
            text_features_by_question[sample["question_id"]] = feature

    return text_features_by_question


def _rank_candidates_from_features(
    text_feature,
    candidate_feature_path,
    num_keyframes,
):
    payload = torch.load(candidate_feature_path, map_location="cpu")
    frame_indices = payload["frame_indices"]
    image_features = payload["image_features"].to(base.device)
    text_feature = text_feature.to(base.device)

    with torch.no_grad():
        similarity = 100.0 * image_features @ text_feature.unsqueeze(1)

    top_k_count = min(num_keyframes, similarity.numel())
    _, top_k_indices = torch.topk(similarity.squeeze(), top_k_count)
    top_keyframes = [frame_indices[int(index)] for index in top_k_indices]
    return [[frame_index, rank] for rank, frame_index in enumerate(top_keyframes)]


def _rank_candidates_for_question(prompt, candidate_frame_indices, full_video_path, num_keyframes):
    frames_cluster, _ = base.load_video(
        full_video_path,
        candidate_frame_indices,
        start=None,
        end=None,
    )
    image_batch = [base.preprocess_clip(frame) for frame in frames_cluster]
    if not image_batch:
        return None

    prompt = _trim_prompt(prompt)
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
    return [[frame_index, rank] for rank, frame_index in enumerate(top_keyframes)]


def cluster(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    dataset,
    device,
    combined_output_path=None,
    num_keyframes=12,
    lambda_event=0.5,
    alpha_gap=0.6,
    beta_redundancy=0.8,
    max_frames_to_extract=5400,
    num_workers=None,
    worker_blas_threads=1,
    mp_start_method=None,
    mp_chunksize=None,
    clip_image_batch_size=128,
    clip_text_batch_size=128,
    temp_feature_dir=None,
):
    """
    End-to-end pipeline with multiprocessing for Stage 1 candidate selection.

    Stage 1 remains query-agnostic and runs in worker processes across videos.
    Stage 2 CLIP ranking stays in the main process to avoid CUDA multiprocessing
    issues and to keep output behavior aligned with the single-process script.
    """
    del dataset

    os.makedirs(save_cluster_path, exist_ok=True)
    video_frame_tensor = base.load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path, "r") as f:
        qa_data = json.load(f)

    combined_results, pending_samples = _load_existing_results(qa_data, save_cluster_path)
    video_tasks = _build_video_tasks(pending_samples, video_path)

    resolved_num_workers = _resolve_num_workers(num_workers)
    resolved_blas_threads = _resolve_blas_threads(worker_blas_threads)
    resolved_start_method = _resolve_start_method(mp_start_method)
    resolved_chunksize = _resolve_chunksize(
        len(video_tasks), resolved_num_workers, mp_chunksize
    )

    print(
        "Stage 1 multiprocessing config:",
        {
            "num_workers": resolved_num_workers,
            "worker_blas_threads": resolved_blas_threads,
            "start_method": resolved_start_method,
            "chunksize": resolved_chunksize,
            "videos_to_process": len(video_tasks),
            "questions_to_process": len(pending_samples),
        },
    )

    stage1_results = _run_stage1_parallel(
        video_tasks=video_tasks,
        video_frame_tensor=video_frame_tensor,
        num_keyframes=num_keyframes,
        lambda_event=lambda_event,
        alpha_gap=alpha_gap,
        beta_redundancy=beta_redundancy,
        max_frames_to_extract=max_frames_to_extract,
        num_workers=resolved_num_workers,
        worker_blas_threads=resolved_blas_threads,
        start_method=resolved_start_method,
        chunksize=resolved_chunksize,
    )

    missing_tensor_count = 0
    skipped_stage1_count = 0

    temp_dir_created = False
    feature_index = {}
    skipped_feature_count = 0

    try:
        if pending_samples:
            base.load_clip_model(device)
            if temp_feature_dir:
                temp_feature_dir = base.resolve_path(temp_feature_dir)
                os.makedirs(temp_feature_dir, exist_ok=True)
            else:
                temp_feature_dir = tempfile.mkdtemp(prefix="temporal_chain_clip_features_")
                temp_dir_created = True

            pending_video_names = [task[0] for task in video_tasks]
            print(
                "Stage 2 CLIP precompute config:",
                {
                    "image_batch_size": max(1, int(clip_image_batch_size)),
                    "text_batch_size": max(1, int(clip_text_batch_size)),
                    "temp_feature_dir": temp_feature_dir,
                    "videos_to_encode": len(pending_video_names),
                    "questions_to_encode": len(pending_samples),
                },
            )
            feature_index, skipped_feature_count = _precompute_candidate_image_features(
                stage1_results=stage1_results,
                pending_video_names=pending_video_names,
                video_path=video_path,
                feature_dir=temp_feature_dir,
                image_batch_size=clip_image_batch_size,
            )
            text_features_by_question = _encode_question_text_features(
                pending_samples, clip_text_batch_size
            )
        else:
            text_features_by_question = {}

        for sample in tqdm(pending_samples, total=len(pending_samples), desc="Stage 2 CLIP rank"):
            video_name = sample["video_name"]
            question_id = sample["question_id"]
            output_path = os.path.join(save_cluster_path, f"{question_id}.json")

            stage1_item = stage1_results.get(video_name)
            if stage1_item is None:
                skipped_stage1_count += 1
                continue
            if stage1_item["status"] == "missing_tensor":
                missing_tensor_count += 1
                continue
            if stage1_item["status"] != "ok":
                skipped_stage1_count += 1
                print(
                    f"stage1_failed video_name={video_name} error={stage1_item.get('error', 'unknown')}"
                )
                continue

            candidate_feature_path = feature_index.get(video_name)
            text_feature = text_features_by_question.get(question_id)
            if candidate_feature_path is None or text_feature is None:
                continue

            key_frame_order = _rank_candidates_from_features(
                text_feature=text_feature,
                candidate_feature_path=candidate_feature_path,
                num_keyframes=num_keyframes,
            )

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
    finally:
        if temp_feature_dir and temp_dir_created:
            shutil.rmtree(temp_feature_dir, ignore_errors=True)

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

    print(
        "Finished temporal-chain multiprocessing clustering."
        f" skipped_missing_tensor={missing_tensor_count}"
        f" skipped_stage1={skipped_stage1_count}"
        f" skipped_feature={skipped_feature_count}"
        f" total_questions={len(qa_data)}"
    )


@hydra.main(
    config_path="configs/keyframe_ranking", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    """Hydra entrypoint for the multiprocessing keyframe selector."""
    cluster(
        base.resolve_path(cfg.json_path),
        base.resolve_path(cfg.video_path),
        base.resolve_path(cfg.video_frame_tensor_path),
        base.resolve_path(cfg.save_cluster_path),
        cfg.dataset,
        cfg.device,
        combined_output_path=base.resolve_path(cfg.combined_output_path),
        num_keyframes=getattr(cfg, "num_keyframes", 12),
        lambda_event=getattr(cfg, "lambda_event", 0.5),
        alpha_gap=getattr(cfg, "alpha_gap", 0.6),
        beta_redundancy=getattr(cfg, "beta_redundancy", 0.8),
        max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
        num_workers=getattr(cfg, "num_workers", None),
        worker_blas_threads=getattr(cfg, "worker_blas_threads", 1),
        mp_start_method=getattr(cfg, "mp_start_method", None),
        mp_chunksize=getattr(cfg, "mp_chunksize", None),
        clip_image_batch_size=getattr(cfg, "clip_image_batch_size", 128),
        clip_text_batch_size=getattr(cfg, "clip_text_batch_size", 128),
        temp_feature_dir=getattr(cfg, "temp_feature_dir", None),
    )


if __name__ == "__main__":
    main()
