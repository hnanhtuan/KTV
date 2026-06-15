import os
import re
import time
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm

import ktv.methods.clustering as base
from ktv.core.tracking import write_summary_json
from ktv.core.utils import resolve_path

QUERY_MODE_QUESTION_ONLY = "question_only"
QUERY_MODE_QUESTION_PLUS_OPTIONS = "question_plus_options"
VALID_QUERY_MODES = {
    QUERY_MODE_QUESTION_ONLY,
    QUERY_MODE_QUESTION_PLUS_OPTIONS,
}

OPTION_PREFIX_RE = re.compile(r"^\s*([A-Z])[\.\):]\s*")


def get_sample_id(sample):
    return str(sample.get("id", sample.get("question_id")))


def normalize_candidate_text(candidate, fallback_letter):
    text = str(candidate).strip()
    match = OPTION_PREFIX_RE.match(text)
    if match:
        return text
    return f"{fallback_letter}. {text}"


def build_query_text(sample, query_mode):
    if query_mode not in VALID_QUERY_MODES:
        raise ValueError(
            f"Unsupported query_mode={query_mode!r}. "
            f"Expected one of {sorted(VALID_QUERY_MODES)}."
        )

    question = str(sample.get("question", "")).strip()
    if query_mode == QUERY_MODE_QUESTION_ONLY:
        return question

    candidates = sample.get("candidates", []) or []
    option_lines = [
        normalize_candidate_text(candidate, chr(ord("A") + idx))
        for idx, candidate in enumerate(candidates)
    ]
    if not option_lines:
        return question
    return f"Question: {question}\nOptions:\n" + "\n".join(option_lines)


def truncate_query_for_clip(query_text, max_words=40):
    words = query_text.split()
    if len(words) <= max_words:
        return query_text
    return " ".join(words[10 : 10 + max_words])


def ensure_unique_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    unique_values = []
    for value in values:
        int_value = int(value)
        if int_value in seen:
            continue
        seen.add(int_value)
        unique_values.append(int_value)
    return unique_values


def resolve_variant_dir_name(selection_mode, dense_candidate_pool_size):
    if selection_mode == "clustered_12":
        return "query_aware_12_candidate"
    if selection_mode == "dense_uniform":
        return f"query_aware_dense_uniform_f{dense_candidate_pool_size}"
    if selection_mode == "uniform_12":
        return "query_aware_uniform_12_candidate"
    raise ValueError(
        f"Unsupported selection_mode={selection_mode!r}. "
        "Expected one of ['clustered_12', 'dense_uniform', 'uniform_12']."
    )


def default_output_paths(
    output_root,
    selection_mode,
    query_mode,
    dense_candidate_pool_size,
    output_top_k,
):
    variant_dir = resolve_variant_dir_name(selection_mode, dense_candidate_pool_size)
    base_dir = os.path.join(output_root, variant_dir, query_mode)
    return (
        os.path.join(base_dir, "keyframes"),
        os.path.join(base_dir, f"keyframe{output_top_k}_order.json"),
    )


def get_video_metadata(full_video_path):
    if os.path.isdir(full_video_path):
        total_frames = len(
            [
                name
                for name in os.listdir(full_video_path)
                if os.path.isfile(os.path.join(full_video_path, name))
            ]
        )
        return 1.0, total_frames

    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {full_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def build_dense_uniform_candidates(total_frames, pool_size):
    if pool_size <= 0:
        raise ValueError("dense_candidate_pool_size must be positive.")
    if total_frames <= 0:
        return []
    if total_frames <= pool_size:
        return list(range(total_frames))
    return ensure_unique_preserve_order(
        np.linspace(0, total_frames, pool_size, endpoint=False, dtype=int).tolist()
    )


def build_clustered_candidates(
    tensor,
    total_frames,
    fps,
    num_keyframes,
    max_frames_to_extract,
):
    clustered_indices = base.video_frame_clustering(tensor, num_keyframes)
    candidate_frame_indices = [
        base.get_original_frame_number(
            total_frames,
            index,
            fps=fps,
            max_frames_to_extract=max_frames_to_extract,
        )
        for index in clustered_indices
    ]
    candidate_frame_indices = ensure_unique_preserve_order(candidate_frame_indices)
    if len(candidate_frame_indices) != num_keyframes:
        candidate_frame_indices = build_dense_uniform_candidates(total_frames, num_keyframes)
    return candidate_frame_indices


def build_uniform_candidates(total_frames, num_keyframes):
    return build_dense_uniform_candidates(total_frames, num_keyframes)


def load_candidate_frames(video_path, candidate_frame_indices):
    candidate_frame_indices = ensure_unique_preserve_order(candidate_frame_indices)
    if not candidate_frame_indices:
        return [], []

    if os.path.isdir(video_path):
        frames = []
        valid_indices = []
        file_names = sorted(os.listdir(video_path))
        for frame_index in candidate_frame_indices:
            if frame_index < 0 or frame_index >= len(file_names):
                continue
            frame_name = file_names[frame_index]
            frame_path = os.path.join(video_path, frame_name)
            try:
                frames.append(Image.open(frame_path).convert("RGB"))
                valid_indices.append(frame_index)
            except Exception:
                print(f"Warning: failed to read frame {frame_index} from {video_path}")
        return frames, valid_indices

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    frames = []
    valid_indices = []
    for frame_index in candidate_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {frame_index} from {video_path}")
            continue
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            valid_indices.append(int(frame_index))
        except Exception:
            print(f"Warning: failed to convert frame {frame_index} from {video_path}")
    cap.release()
    return frames, valid_indices


def rank_candidate_frames(query_text, frames):
    if not frames:
        return torch.empty(0, device=base.device)

    image_batch = [base.preprocess_clip(frame) for frame in frames]
    image_input = torch.stack(image_batch).to(base.device)
    query_text = truncate_query_for_clip(query_text)

    with torch.no_grad():
        image_features = base.model_clip.encode_image(image_input)
        text_input = clip.tokenize([query_text]).to(base.device)
        text_features = base.model_clip.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).squeeze(-1)

    return similarity


def select_top_keyframes(candidate_frame_indices, similarity, output_top_k):
    if similarity.numel() == 0:
        return []
    top_k_count = min(output_top_k, similarity.numel())
    _, top_k_indices = torch.topk(similarity, top_k_count)
    top_keyframes = [candidate_frame_indices[int(index)] for index in top_k_indices]
    return [[int(frame_index), rank] for rank, frame_index in enumerate(top_keyframes)]


def run_query_aware_selection(
    json_path,
    video_path,
    video_frame_tensor_path,
    save_cluster_path,
    combined_output_path,
    selection_mode,
    query_mode,
    num_keyframes,
    dense_candidate_pool_size,
    output_top_k,
    max_frames_to_extract,
    skip_existing,
    sample_limit,
):
    start_time = time.time()
    if dense_candidate_pool_size < output_top_k:
        raise ValueError(
            "dense_candidate_pool_size must be greater than or equal to output_top_k."
        )

    os.makedirs(save_cluster_path, exist_ok=True)

    video_frame_tensor = None
    if selection_mode == "clustered_12":
        video_frame_tensor = base.load_video_frame_tensor(video_frame_tensor_path)

    with open(json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    if sample_limit is not None:
        qa_data = qa_data[:sample_limit]

    combined_results = {}
    missing_tensor_count = 0
    empty_result_count = 0
    skipped_existing_count = 0
    saved_count = 0

    desc_name = resolve_variant_dir_name(selection_mode, dense_candidate_pool_size)

    for sample in tqdm(qa_data, total=len(qa_data), desc=desc_name):
        sample_id = get_sample_id(sample)
        output_path = os.path.join(save_cluster_path, f"{sample_id}.json")
        if skip_existing and os.path.exists(output_path):
            skipped_existing_count += 1
            with open(output_path, "r", encoding="utf-8") as f:
                combined_results.update(json.load(f))
            continue

        full_video_path = os.path.join(video_path, sample["video_name"])
        fps, total_frames = get_video_metadata(full_video_path)

        if selection_mode == "uniform_12":
            candidate_frame_indices = build_uniform_candidates(
                total_frames,
                num_keyframes,
            )
        elif selection_mode == "clustered_12":
            tensor = base.get_tensor_for_video(video_frame_tensor, sample["video_name"])
            if tensor is None:
                missing_tensor_count += 1
                continue
            candidate_frame_indices = build_clustered_candidates(
                tensor,
                total_frames,
                fps,
                num_keyframes,
                max_frames_to_extract,
            )
        elif selection_mode == "dense_uniform":
            candidate_frame_indices = build_dense_uniform_candidates(
                total_frames,
                dense_candidate_pool_size,
            )
        else:
            raise ValueError(
                f"Unsupported selection_mode={selection_mode!r}. "
                "Expected one of ['uniform_12', 'clustered_12', 'dense_uniform']."
            )

        frames, valid_indices = load_candidate_frames(full_video_path, candidate_frame_indices)
        query_text = build_query_text(sample, query_mode)
        similarity = rank_candidate_frames(query_text, frames)
        key_frame_order = select_top_keyframes(valid_indices, similarity, output_top_k)
        if len(key_frame_order) != output_top_k:
            empty_result_count += 1

        result = {sample_id: key_frame_order}
        combined_results.update(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        saved_count += 1

    combined_output_dir = os.path.dirname(combined_output_path)
    if combined_output_dir:
        os.makedirs(combined_output_dir, exist_ok=True)
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)

    summary_path = Path(combined_output_path).with_name(
        f"{Path(combined_output_path).stem}_summary.json"
    )
    summary = {
        "json_path": str(Path(json_path).resolve()),
        "video_path": str(Path(video_path).resolve()),
        "video_frame_tensor_path": str(Path(video_frame_tensor_path).resolve()) if video_frame_tensor_path else None,
        "save_cluster_path": str(Path(save_cluster_path).resolve()),
        "combined_output_path": str(Path(combined_output_path).resolve()),
        "summary_path": str(summary_path.resolve()),
        "selection_mode": selection_mode,
        "query_mode": query_mode,
        "dense_candidate_pool_size": dense_candidate_pool_size,
        "output_top_k": output_top_k,
        "num_keyframes": num_keyframes,
        "sample_count": len(qa_data),
        "saved_count": saved_count,
        "skipped_existing_count": skipped_existing_count,
        "missing_tensor_count": missing_tensor_count,
        "non_full_output_count": empty_result_count,
        "duration_seconds": time.time() - start_time,
    }
    summary["summary_path"] = write_summary_json(summary_path, summary)
    return summary
