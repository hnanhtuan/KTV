#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

# sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "ktv"))
import json
import hydra
from PIL import Image
from tqdm import tqdm
import torch  # for cuda device

# import torch_npu  for npu device
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from ktv.llava.constants import IMAGE_TOKEN_INDEX
from ktv.llava.model.builder import load_pretrained_model
from ktv.llava.utils import disable_torch_init
from ktv.llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

from dataset import load_video
from prompt import get_multiple_choice_prompt
from utils import get_chunk


def resolve_path(path):
    """Resolve local config paths relative to the original launch directory."""
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def llava_inference(
    video_frames: Sequence[Image.Image],
    question: str,
    candidates: Sequence[str],
    conv_mode: str,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
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
    # Get multiple choice prompt
    prompt = get_multiple_choice_prompt(model, conv_mode, question, candidates)
    # print(prompt)
    # Get text inputs
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

    # Get image inputs
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
    """
    Run inference on Video QA Dataset.

    Args:
        args: Hydra/OmegaConf config containing dataset, model, and decoding options.
    """
    # Keep model initialization lightweight and choose the runtime device once.
    disable_torch_init()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else {"": "cpu"}
    print(f"Using device: {device}")

    # Resolve paths that refer to local files or folders. Hydra can run from a
    # different working directory, so resolve_path keeps paths relative to launch dir.
    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    output_dir = resolve_path(args.output_dir)
    key_frame_path = resolve_path(args.key_frame_path)

    # Load tokenizer, model, and image preprocessor.
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

    # Load optional precomputed keyframe selections. The expected format is:
    # {question_id: [[frame_index, rank], ...]}.
    keyframes_by_question = {}
    if key_frame_path:
        with open(key_frame_path, "r") as f:
            keyframes_by_question = json.load(f)

    # Override image aspect ratio when the experiment config requests it.
    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    # Load QA samples and optionally keep only this process's shard.
    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    # Prepare output file. Existing predictions are appended to, and already
    # generated question IDs are skipped so interrupted runs can resume.
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    generated_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                generated_ids.add(data["id"])
    output_mode = "a" if os.path.exists(output_path) else "w"

    with open(output_path, output_mode) as ans_file:
        # Process each QA sample independently: load frames, run LLaVA, write result.
        for sample in tqdm(gt_qa_pairs):
            video_name = sample["video_name"]
            question_id = sample["question_id"]
            if question_id in generated_ids:
                continue

            question = sample["question"]
            candidates = sample["candidates"]
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                continue

            sample_set = {
                "task_name": sample["task_name"],
                "question": question,
                "id": question_id,
                "answer_number": sample["answer_number"],
                "candidates": candidates,
                "answer": sample["answer"],
            }

            # If keyframes exist for this question, pass the selected frame IDs into
            # video loading and build keyframe_order in sorted-frame order. The model
            # uses that order to assign token budgets to the loaded frames.
            keyframe = keyframes_by_question.get(question_id)
            if keyframe:
                frame_to_rank = {frame_index: rank for frame_index, rank in keyframe}
                keyframe_order = [
                    frame_to_rank[frame_index] for frame_index in sorted(frame_to_rank)
                ]
            else:
                keyframe = None
                keyframe_order = None

            # Load either the selected keyframes or uniformly sampled video frames.
            video_frames, sizes = load_video(
                video_path, keyframe, num_frms=args.num_frames
            )

            # Run one multiple-choice inference request and normalize image wording
            # in the generated answer back to video wording.
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
            print(output)
            sample_set["pred"] = output
            ans_file.write(json.dumps(sample_set) + "\n")


@hydra.main(config_path="configs/inference", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
