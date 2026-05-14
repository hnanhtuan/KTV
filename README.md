# KTV

Code for [KTV: Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs](https://ojs.aaai.org/index.php/AAAI/article/view/37862), accepted by AAAI 2026.

## 1) Environment Setup

This repo is configured around `uv` + Python 3.10.

### 1.1 Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 Create and activate the virtual environment

```bash
cd /path/to/KTV
uv venv --python 3.10.12
source .venv/bin/activate
```

### 1.3 Install dependencies

```bash
uv pip install -e ./ktv/llava
uv pip install -e .
uv pip install opencv-python numpy==1.26.2 protobuf transformers_stream_generator
```

If PyTorch is not already installed (or CUDA build does not match your machine), install the right build with `uv pip install ...` from the official PyTorch selector.

### 1.4 Optional: install Hugging Face CLI for dataset download scripts

The dataset scripts can auto-download from Hugging Face and require either `hf` or `huggingface-cli`.

```bash
uv tool install huggingface-hub
```

## 2) Download Model Weights

Download LLaVA-NeXT model weights into `liuhaotian/`:

```bash
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.6-vicuna-7b
# optional larger model
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b liuhaotian/llava-v1.6-34b
```

Manual pages:
- https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
- https://huggingface.co/liuhaotian/llava-v1.6-34b

## 3) Prepare Datasets

The default experiments in this repo use **NExTQA** and **Video-MME**.

### 3.1 NExTQA

Prepare videos (and optionally auto-download raw data) into `datasets/NExTQA/videos`:

```bash
bash scripts/data/prepare_nextqa_dataset.sh
```

Useful overrides:

```bash
# use an existing extracted NExTVideo directory
DOWNLOAD_RAW=never bash scripts/data/prepare_nextqa_dataset.sh \
  /abs/path/to/NExTVideo \
  datasets/NExTQA/videos \
  playground/gt_qa_files/NExTQA/val_qa.json
```

Expected paths after setup:
- videos: `datasets/NExTQA/videos`
- QA file: `playground/gt_qa_files/NExTQA/val_qa.json`

### 3.2 Video-MME

Prepare videos and build QA JSON:

```bash
bash scripts/data/prepare_videomme_dataset.sh
```

Useful overrides:

```bash
# use pre-downloaded Video-MME zip files
DOWNLOAD_RAW=never bash scripts/data/prepare_videomme_dataset.sh \
  /abs/path/to/Video-MME \
  datasets/Video-MME \
  playground/gt_qa_files/Videomme/val_qa.csv \
  playground/gt_qa_files/Videomme
```

Expected paths after setup:
- videos: `datasets/Video-MME/data`
- QA file: `playground/gt_qa_files/Videomme/val_qa.json`

## 4) Run Keyframe Preparation

KTV keyframe pipeline has two steps:
1. extract DINOv2 frame features
2. cluster + rank keyframes

### 4.1 NExTQA

```bash
uv run python keyframe_select_new.py experiment=nextqa
uv run python cluster_keyframe_and_order.py experiment=nextqa
```

Output keyframe JSON:
- `outputs/nextqa_keyframe6_order.json`

### 4.2 Video-MME

```bash
uv run python keyframe_select_new.py experiment=videomme
uv run python cluster_keyframe_and_order.py experiment=videomme
```

Output keyframe JSON:
- `outputs/videomme_keyframe6_order.json`

## 5) Run Inference / Experiments

Main entrypoint:

```bash
uv run python run_inference_multiple_choice_qa.py ...
```

### 5.1 Single run (NExTQA example)

```bash
uv run python run_inference_multiple_choice_qa.py \
  experiment=nextqa \
  output_name=nextqa_ktv_full_cls_new_token_sim_tokens1872 \
  key_frame_path=outputs/nextqa_keyframe6_order.json \
  prune_mode=cls_new_token_sim \
  rate=0.2 \
  tokens_num=1872
```

### 5.2 Evaluate predictions

```bash
uv run python eval/compute_accuracy.py outputs/nextqa_ktv_full_cls_new_token_sim_tokens1872.json
```

### 5.3 Run standard variant set with one command

`runcode.sh` runs these variants for one dataset setup:
- baseline uniform frames
- upper-bound dense uniform frames
- KTV keyframe-only
- KTV token-only (`cls_new_token_sim`, `uniform_token`)
- KTV full (`keyframe + token`)

```bash
bash runcode.sh
```

Common overrides:

```bash
EXPERIMENT=nextqa \
RUN_PREFIX=nextqa \
KEYFRAME_JSON=outputs/nextqa_keyframe6_order.json \
TOKENS_NUM=1872 \
RATE=0.2 \
UPPER_BOUND_NUM_FRAMES=48 \
bash runcode.sh
```

### 5.4 Run full token sweep for multiple datasets

`run_all_tokens_experiments.sh` runs key variants for all datasets in `DATASETS` and all token budgets in `TOKEN_LIST`, then computes accuracy.

```bash
bash run_all_tokens_experiments.sh
```

Common overrides:

```bash
DATASETS="nextqa videomme" \
TOKEN_LIST="504 936 1872" \
RATE=0.2 \
UPPER_BOUND_NUM_FRAMES_LIST="12 16 20 24 28 32 36 40 44 48" \
bash run_all_tokens_experiments.sh
```

## 6) Output Files

Inference predictions are written to:
- `outputs/<output_name>.json`

Accuracy reports are written to:
- `outputs/<output_name>_accuracy.txt`

Keyframe clustering outputs are written to:
- per-sample keyframes: `outputs/<dataset>_keyframes/`
- combined JSON: `outputs/<dataset>_keyframe6_order.json`

## 7) Notes and Troubleshooting

- If OOM happens, reduce `num_frames` and/or `tokens_num`.
- For CPU runs, override `device=cpu` (much slower).
- For multi-GPU runs, set:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python run_inference_multiple_choice_qa.py experiment=nextqa
```

- If `hf` command is missing, install Hugging Face CLI (`uv tool install huggingface-hub`).
- If dataset scripts cannot find files, check absolute paths and rerun with explicit arguments.
