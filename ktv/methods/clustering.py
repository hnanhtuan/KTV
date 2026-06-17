import os
import time
import json
import pickle
from pathlib import Path
import numpy as np
import cv2
import av
import torch
import clip
from sklearn.cluster import KMeans
from PIL import Image

from ktv.core.dataset import load_video
from ktv.core.tracking import write_summary_json
from ktv.core.utils import resolve_path

device = "cpu"
model_clip = None
preprocess_clip = None


def select_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU instead.")
        device_name = "cpu"
    return device_name


def load_clip_model(device_name):
    global device, model_clip, preprocess_clip
    device = select_device(device_name)
    if model_clip is None:
        model_clip, preprocess_clip = clip.load("ViT-L/14", device=device)
    else:
        model_clip.to(device)
    print(f"Using CLIP device: {device}")


def run_kmedoids(features: np.ndarray, num_clusters: int, metric: str = "cosine", max_iter: int = 100, random_state: int = 0):
    np.random.seed(random_state)
    n_samples = features.shape[0]
    if n_samples <= num_clusters:
        return list(range(n_samples)) + [0] * (num_clusters - n_samples)
    
    if metric == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = features / norms
        dist_matrix = 1.0 - np.dot(normalized, normalized.T)
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    else:  # l2
        sq_norms = np.sum(features ** 2, axis=1)
        dot_prods = np.dot(features, features.T)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_prods
        dist_matrix = np.sqrt(np.maximum(dist_sq, 0.0))
        
    # k-means++ initialization on distance matrix
    medoids = [np.random.choice(n_samples)]
    for _ in range(1, num_clusters):
        min_dist = np.min(dist_matrix[:, medoids], axis=1)
        sq_dist = min_dist ** 2
        sum_sq_dist = np.sum(sq_dist)
        if sum_sq_dist == 0:
            probs = np.ones(n_samples) / n_samples
        else:
            probs = sq_dist / sum_sq_dist
        new_medoid = np.random.choice(n_samples, p=probs)
        medoids.append(new_medoid)
    medoids = np.array(medoids)
    
    for _ in range(max_iter):
        labels = np.argmin(dist_matrix[:, medoids], axis=1)
        new_medoids = medoids.copy()
        
        for k in range(num_clusters):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                continue
            sub_dist = dist_matrix[cluster_indices][:, cluster_indices]
            costs = np.sum(sub_dist, axis=1)
            new_medoids[k] = cluster_indices[np.argmin(costs)]
            
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
        
    return medoids.tolist()


def perform_clustering(frame_features: np.ndarray, num_clusters: int, clustering_method: str = "kmeans"):
    n_samples = frame_features.shape[0]
    num_clusters = min(num_clusters, n_samples)
    
    if num_clusters <= 0:
        return np.zeros(n_samples, dtype=int), np.zeros((0, frame_features.shape[1]), dtype=np.float32), np.zeros(n_samples, dtype=np.float32)

    method = clustering_method.lower().strip()
    
    if method == "kmeans":
        kmeans = KMeans(
            n_clusters=num_clusters, random_state=0, init="k-means++", n_init=10
        ).fit(frame_features)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        r_cluster = -np.linalg.norm(frame_features - centers[labels], axis=1)
        return labels, centers, r_cluster
        
    elif method.startswith("kmedoids"):
        metric = "cosine" if "cosine" in method else "l2"
        medoid_indices = run_kmedoids(frame_features, num_clusters, metric=metric, random_state=0)
        centers = frame_features[medoid_indices]
        
        if metric == "cosine":
            norm_features = frame_features / (np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8)
            norm_centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
            dist_matrix = 1.0 - np.dot(norm_features, norm_centers.T)
        else:
            dist_matrix = np.linalg.norm(frame_features[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            
        labels = np.argmin(dist_matrix, axis=1)
        r_cluster = -dist_matrix[np.arange(n_samples), labels]
        return labels, centers, r_cluster
        
    elif method.startswith("agglomerative"):
        from sklearn.cluster import AgglomerativeClustering
        metric = "cosine" if "cosine" in method else "l2"
        linkage = "average"
        sklearn_metric = "cosine" if metric == "cosine" else "euclidean"
        
        agg = AgglomerativeClustering(
            n_clusters=num_clusters, metric=sklearn_metric, linkage=linkage
        )
        labels = agg.fit_predict(frame_features)
        
        centers = []
        for k in range(num_clusters):
            cluster_members = frame_features[labels == k]
            if len(cluster_members) == 0:
                centers.append(np.zeros_like(frame_features[0]))
            else:
                centers.append(cluster_members.mean(axis=0))
        centers = np.array(centers)
        
        if metric == "cosine":
            norm_features = frame_features / (np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8)
            norm_centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
            dist_matrix = 1.0 - np.dot(norm_features, norm_centers.T)
        else:
            dist_matrix = np.linalg.norm(frame_features[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            
        r_cluster = -dist_matrix[np.arange(n_samples), labels]
        return labels, centers, r_cluster
        
    else:
        raise ValueError(f"Unknown clustering_method: {clustering_method}")


def video_frame_clustering(frame_features: np.ndarray, num_cluster: int = 5, clustering_method: str = "kmeans"):
    labels, centers, r_cluster = perform_clustering(frame_features, num_cluster, clustering_method)
    
    metric = "cosine" if "cosine" in clustering_method.lower() else "l2"
    if metric == "cosine":
        norm_features = frame_features / (np.linalg.norm(frame_features, axis=1, keepdims=True) + 1e-8)
        norm_centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        distances = 1.0 - np.dot(norm_centers, norm_features.T)
    else:
        distances = np.linalg.norm(
            frame_features - centers[:, np.newaxis, :], axis=2
        )
    closest_frames = np.argmin(distances, axis=1)
    cluster_center_indices = [int(closest_frames[j]) for j in range(num_cluster)]
    return cluster_center_indices


def get_original_frame_number(
    total_original_frames: int,
    index_in_extracted_list: int,
    ts: list = None,
    fps: int = None,
    max_frames_to_extract: int = 5400,
) -> int:
    num_actually_extracted = min(total_original_frames, max_frames_to_extract)

    if index_in_extracted_list >= num_actually_extracted:
        raise ValueError(
            f"index_in_extracted_list ({index_in_extracted_list}) out range"
        )

    if total_original_frames <= max_frames_to_extract:
        if ts:
            return int(ts[0] * fps) + index_in_extracted_list
        else:
            return index_in_extracted_list
    else:
        if num_actually_extracted == 1:
            if index_in_extracted_list == 0:
                return 0
            raise ValueError(
                "If only one frame is extracted, index_in_extracted_list must be 0."
            )
        changed_list = np.linspace(
            0, total_original_frames - 1, max_frames_to_extract, dtype=int
        )
        return int(changed_list[index_in_extracted_list])


def load_video_frame_tensor(video_frame_tensor_path):
    if os.path.isdir(video_frame_tensor_path):
        video_frame_tensor = {}
        for filename in sorted(os.listdir(video_frame_tensor_path)):
            if not filename.endswith(".pkl"):
                continue
            with open(os.path.join(video_frame_tensor_path, filename), "rb") as f:
                video_frame_tensor.update(pickle.load(f))
        return video_frame_tensor

    with open(video_frame_tensor_path, "rb") as f:
        return dict(pickle.load(f))


def get_tensor_for_video(video_frame_tensor, video_name):
    tensor = video_frame_tensor.get(video_name)
    if tensor is not None:
        return tensor

    stem, _ = os.path.splitext(video_name)
    if stem:
        tensor = video_frame_tensor.get(stem)
        if tensor is not None:
            return tensor

    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        tensor = video_frame_tensor.get(f"{video_name}{ext}")
        if tensor is not None:
            return tensor
    return None


def build_uniform_keyframe_fallback(total_frames: int, num_keyframes: int):
    if num_keyframes <= 0:
        return []
    safe_total_frames = max(int(total_frames), 1)
    return [
        int(index)
        for index in np.linspace(
            0, float(safe_total_frames), num_keyframes, endpoint=False
        )
    ]


def run_clustering(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    dataset,
    combined_output_path=None,
    num_keyframes=12,
    enable_query_aware_ranking=True,
    max_frames_to_extract=5400,
    clustering_method="kmeans",
):
    start_time = time.time()
    os.makedirs(save_cluster_path, exist_ok=True)
    video_frame_tensor = load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    combined_results = {}
    missing_tensor_count = 0
    skipped_existing_count = 0
    skipped_empty_frame_count = 0
    saved_count = 0

    for sample in tqdm(qa_data, total=len(qa_data)) if "tqdm" in globals() or "tqdm" in sys.modules else qa_data:
        from tqdm import tqdm as tqdm_impl
        video_name = sample["video_name"]
        prompt = sample["question"]
        question_id = sample["question_id"]

        output_path = os.path.join(save_cluster_path, f"{question_id}.json")
        if os.path.exists(output_path):
            skipped_existing_count += 1
            with open(output_path, "r", encoding="utf-8") as f:
                combined_results.update(json.load(f))
            continue

        full_video_path = os.path.join(video_path, video_name)
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        tensor = get_tensor_for_video(video_frame_tensor, video_name)
        if tensor is None:
            missing_tensor_count += 1
            continue

        clustered_indices = video_frame_clustering(tensor, num_keyframes, clustering_method)

        candidate_frame_indices = [
            get_original_frame_number(
                total_frames,
                index,
                fps=fps,
                max_frames_to_extract=max_frames_to_extract,
            )
            for index in clustered_indices
        ]

        if (
            len(candidate_frame_indices) != num_keyframes
            or len(set(candidate_frame_indices)) != num_keyframes
        ):
            candidate_frame_indices = build_uniform_keyframe_fallback(
                total_frames, num_keyframes
            )

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
        if not frames_cluster:
            skipped_empty_frame_count += 1
            continue

        image_batch = [preprocess_clip(frame) for frame in frames_cluster]

        prompt_words = prompt.split(" ")
        if len(prompt_words) > 40:
            prompt = " ".join(prompt_words[10:50])

        image_input = torch.stack(image_batch).to(device)
        with torch.no_grad():
            image_features = model_clip.encode_image(image_input)
            text_input = clip.tokenize([prompt]).to(device)
            text_features = model_clip.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * image_features @ text_features.T

        top_k_count = min(num_keyframes, similarity.numel())
        _, top_k_indices = torch.topk(similarity.squeeze(), top_k_count)

        top_keyframes = [candidate_frame_indices[int(index)] for index in top_k_indices]
        key_frame_order = [
            [frame_index, rank] for rank, frame_index in enumerate(top_keyframes)
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
        else Path(save_cluster_path) / "keyframe_ranking_summary.json"
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
        "skipped_existing_count": skipped_existing_count,
        "missing_tensor_count": missing_tensor_count,
        "skipped_empty_frame_count": skipped_empty_frame_count,
        "enable_query_aware_ranking": bool(enable_query_aware_ranking),
        "clustering_method": clustering_method,
        "num_keyframes": num_keyframes,
        "duration_seconds": time.time() - start_time,
    }
    summary["summary_path"] = write_summary_json(summary_path, summary)
    return summary
