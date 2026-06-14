#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

# sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "ktv"))
import hydra
import torch  # for cuda device
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, PreTrainedTokenizerBase

# import torch_npu  for npu device
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from ktv.core.tracking import (
    default_shell_artifact_paths,
    track_run,
    write_summary_json,
)
from ktv.llava.constants import IMAGE_TOKEN_INDEX
from ktv.llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from ktv.llava.model.builder import load_pretrained_model
from ktv.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from ktv.llava.utils import disable_torch_init

from ktv.core.dataset import load_video
from eval.compute_accuracy import prediction_to_index
from ktv.core.prompt import get_multiple_choice_prompt
from ktv.core.utils import get_chunk


def resolve_path(path):
    """Resolve local config paths relative to the original launch directory."""
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_answered_records(output_path: str) -> dict[str, dict[str, Any]]:
    """Load existing predictions keyed by id for resumable running metrics."""
    answered_records: dict[str, dict[str, Any]] = {}
    if not os.path.exists(output_path):
        return answered_records

    with open(output_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping invalid JSON on line {line_number} in {output_path}"
                )
                continue

            answer_id = data.get("id", data.get("question_id"))
            if answer_id is None:
                continue
            answered_records[str(answer_id)] = data
    return answered_records


def is_correct_prediction(record: dict[str, Any]) -> bool:
    """Return whether a prediction matches its multiple-choice answer."""
    predicted_index = prediction_to_index(
        record.get("pred"), record.get("candidates", [])
    )
    return predicted_index is not None and predicted_index == record.get("answer_number")


def llava_inference(
    video_frames: Sequence[Image.Image],
    question: str,
    candidates: Sequence[str],
    conv_mode: str,
    model: LlavaLlamaForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: CLIPImageProcessor,
    image_sizes: Sequence[Tuple[int, int]],
    temperature: float,
    top_p: Optional[float],
    num_beams: int,
    temporal_aggregation: Optional[str],
    keyframe_order: Optional[Sequence[int]] = None,
    num_frames: Optional[int] = None,
    prune_mode: Optional[str] = None,
    global_rate: Optional[float] = None,
    tokens_num: Optional[int] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
) -> str:
    prompt = get_multiple_choice_prompt(model, conv_mode, question, candidates)
    input_ids = (
        tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(device)
    )

    image_tensor = process_images(video_frames, image_processor, model.config)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(
                dtype=dtype, device=device, non_blocking=(device == "cuda")
            ),
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=128,
            use_cache=True,
            temporal_aggregation=temporal_aggregation,
            keyframe_order=keyframe_order,
            num_frames=num_frames,
            prune_mode=prune_mode,
            global_rate=global_rate,
            tokens_num=tokens_num,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_inference(args):
    """Run inference on a multiple-choice video QA dataset."""
    start_time = time.time()
    disable_torch_init()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else {"": "cpu"}
    print(f"Using device: {device}")

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    output_dir = resolve_path(args.output_dir)
    key_frame_path = resolve_path(args.key_frame_path)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device=device,
        device_map=device_map,
        rope_scaling_factor=args.rope_scaling_factor,
    )
    del context_len

    keyframes_by_question = {}
    if key_frame_path:
        with open(key_frame_path, "r", encoding="utf-8") as f:
            keyframes_by_question = {
                str(question_id): keyframes
                for question_id, keyframes in json.load(f).items()
            }

    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    accuracy_path = os.path.join(output_dir, f"{args.output_name}_accuracy.txt")
    summary_path = os.path.join(output_dir, f"{args.output_name}_summary.json")
    answered_records = load_answered_records(output_path)
    chunk_ids = {
        str(sample.get("id", sample.get("question_id"))) for sample in gt_qa_pairs
    }
    answered_records = {
        answer_id: record
        for answer_id, record in answered_records.items()
        if answer_id in chunk_ids
    }
    generated_ids = set(answered_records)
    output_mode = "a" if os.path.exists(output_path) else "w"
    correct_count = sum(
        is_correct_prediction(record) for record in answered_records.values()
    )
    answered_count = len(answered_records)
    skipped_existing_count = answered_count
    missing_video_count = 0
    processed_now_count = 0

    with open(output_path, output_mode, encoding="utf-8") as ans_file:
        progress = tqdm(gt_qa_pairs)
        if answered_count:
            progress.set_postfix(
                acc=f"{correct_count / answered_count:.4f}",
                correct=f"{correct_count}/{answered_count}",
            )
        for sample in progress:
            video_name = sample["video_name"]
            answer_id = str(sample.get("id", sample.get("question_id")))
            if answer_id in generated_ids:
                continue

            question = sample["question"]
            candidates = sample["candidates"]
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                missing_video_count += 1
                continue

            sample_set = {
                "task_name": sample["task_name"],
                "question": question,
                "question_id": str(sample.get("question_id", answer_id)),
                "id": answer_id,
                "answer_number": sample["answer_number"],
                "candidates": candidates,
                "answer": sample["answer"],
            }

            keyframe = keyframes_by_question.get(answer_id)
            if keyframe:
                frame_to_rank = {frame_index: rank for frame_index, rank in keyframe}
                keyframe_order = [
                    frame_to_rank[frame_index] for frame_index in sorted(frame_to_rank)
                ]
            else:
                keyframe = None
                keyframe_order = None

            video_frames, sizes = load_video(video_path, keyframe, num_frms=args.num_frames)

            output = llava_inference(
                video_frames,
                question,
                candidates,
                args.conv_mode,
                model,
                tokenizer,
                image_processor,
                sizes,
                args.temperature,
                args.top_p,
                args.num_beams,
                args.temporal_aggregation,
                keyframe_order,
                args.num_frames,
                args.prune_mode,
                global_rate=args.rate,
                tokens_num=args.tokens_num,
                device=device,
                dtype=dtype,
            )
            output = output.replace("In the image", "In the video")
            sample_set["pred"] = output
            ans_file.write(json.dumps(sample_set) + "\n")
            generated_ids.add(answer_id)
            answered_count += 1
            processed_now_count += 1
            if is_correct_prediction(sample_set):
                correct_count += 1
            progress.set_postfix(
                acc=f"{correct_count / answered_count:.4f}",
                correct=f"{correct_count}/{answered_count}",
            )

    final_accuracy = correct_count / answered_count if answered_count else 0.0
    with open(accuracy_path, "w", encoding="utf-8") as accuracy_file:
        accuracy_file.write(f"{final_accuracy * 100:.4f}\n")

    summary = {
        "output_path": str(Path(output_path).resolve()),
        "accuracy_path": str(Path(accuracy_path).resolve()),
        "summary_path": str(Path(summary_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "dataset": getattr(args, "dataset", None),
        "device": str(device),
        "key_frame_path": str(Path(key_frame_path).resolve()) if key_frame_path else None,
        "num_samples_in_chunk": len(gt_qa_pairs),
        "answered_count": answered_count,
        "correct_count": correct_count,
        "accuracy": final_accuracy,
        "skipped_existing": skipped_existing_count,
        "processed_now": processed_now_count,
        "missing_video_count": missing_video_count,
        "duration_seconds": time.time() - start_time,
        "prune_mode": args.prune_mode,
        "tokens_num": args.tokens_num,
        "num_frames": args.num_frames,
    }
    summary["summary_path"] = write_summary_json(summary_path, summary)
    return summary


@hydra.main(config_path="configs/qa_inference", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = resolve_path(cfg.output_dir)
    with track_run(
        cfg,
        stage="qa_inference",
        script_path=__file__,
        output_dir=output_dir,
        extra_tags={
            "dataset": getattr(cfg, "dataset", None),
            "output_name": cfg.output_name,
        },
    ) as tracker:
        tracker.log_params_from_config(cfg)
        tracker.log_resolved_config(cfg)
        summary = run_inference(cfg)
        tracker.log_metrics(
            {
                "accuracy": summary["accuracy"],
                "answered_count": summary["answered_count"],
                "correct_count": summary["correct_count"],
                "skipped_existing": summary["skipped_existing"],
                "processed_now": summary["processed_now"],
                "missing_video_count": summary["missing_video_count"],
                "duration_seconds": summary["duration_seconds"],
            }
        )
        tracker.log_artifacts(
            [
                summary["output_path"],
                summary["accuracy_path"],
                summary["summary_path"],
                key_frame_path if (key_frame_path := summary.get("key_frame_path")) else None,
            ],
            artifact_path="outputs",
        )
        tracker.log_artifacts(default_shell_artifact_paths(), artifact_path="logs")


if __name__ == "__main__":
    main()
