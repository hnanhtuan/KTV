# KTV

Code for [KTV: Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs](https://ojs.aaai.org/index.php/AAAI/article/view/37862), accepted by AAAI 2026.

## Setup With uv

This project uses `uv` for environment and package management. Do not mix this setup with conda or direct `pip` commands.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate the environment:

```bash
uv venv --python 3.10.12
source .venv/bin/activate
```

Install project dependencies and the local LLaVA package:

```bash
uv pip install -e ./ktv/llava
uv pip install -e .
uv pip install opencv-python numpy==1.26.2 protobuf transformers_stream_generator
```

If your environment needs a specific PyTorch build, install it with `uv pip install` using the command from the official PyTorch selector for your CUDA or CPU target.

## Download LLaVA-NeXT Weights

Download the pretrained model weights into the repository:

```bash
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.6-vicuna-7b
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b liuhaotian/llava-v1.6-34b
```

You can also download them manually:

- [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)
- [liuhaotian/llava-v1.6-34b](https://huggingface.co/liuhaotian/llava-v1.6-34b)

## Keyframe Preparation

First extract DINOv2 frame features:

```bash
uv run python keyframe_select_new.py
```

Then cluster and rank keyframes for each test sample:

```bash
uv run python cluster_keyframe_and_order.py
```

The resulting keyframe JSON can be passed to inference through `key_frame_path`.

## Inference

Inference is configured with Hydra. The base config is:

```text
configs/inference/config.yaml
```

Experiment configs live under:

```text
configs/inference/experiment/
```

Run an existing experiment:

```bash
uv run python run_inference_multiple_choice_qa.py experiment=nextqa_test_cpu
```

or:

```bash
uv run python run_inference_multiple_choice_qa.py experiment=videomme_6keyframe
```

You can also override values from the command line:

```bash
uv run python run_inference_multiple_choice_qa.py \
  device=cpu \
  video_dir=datasets/NExTQA/videos \
  gt_file=playground/gt_qa_files/NExTQA/val_qa.json \
  output_dir=outputs \
  output_name=nextqa_test_cpu \
  model_path=liuhaotian/llava-v1.6-vicuna-7b \
  conv_mode=multiple_choice_allvideo_34b_v4 \
  num_frames=6 \
  image_aspect_ratio=resize \
  rope_scaling_factor=2
```

Useful inference settings:

- `num_frames`: number of video frames or selected keyframes to load.
- `key_frame_path`: path to precomputed keyframe selections.
- `prune_mode`: visual token pruning mode, for example `cls_new_token_sim`.
- `rate`: alpha for balancing token importance and redundancy.
- `tokens_num`: number of visual tokens sent to the LLM.
- `device`: `auto`, `cuda`, or `cpu`.

For multi-GPU runs, set `CUDA_VISIBLE_DEVICES` before `uv run`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python run_inference_multiple_choice_qa.py experiment=videomme_6keyframe
```
