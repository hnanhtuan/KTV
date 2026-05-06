#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_VIDEO_DIR="${1:-/media/anhxtuan/f4f67dc7-d85d-43b1-a0a7-e8da7f48f352/Documents/Datasets/NExTQA/NExTVideo}"
TARGET_VIDEO_DIR="${2:-datasets/NExTQA/videos}"
QA_JSON="${3:-playground/gt_qa_files/NExTQA/val_qa.json}"
HF_REPO_ID="${HF_REPO_ID:-rhymes-ai/NeXTVideo}"
DOWNLOAD_RAW="${DOWNLOAD_RAW:-auto}"
RAW_ZIP_NAME="${RAW_ZIP_NAME:-NExTVideo.zip}"
HF_INCLUDE_PATTERNS=(
  "$RAW_ZIP_NAME"
  "README.md"
  "train.jsonl"
  "val.jsonl"
)

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

SOURCE_VIDEO_DIR="$(resolve_path "$SOURCE_VIDEO_DIR")"
TARGET_VIDEO_DIR="$(resolve_path "$TARGET_VIDEO_DIR")"
QA_JSON="$(resolve_path "$QA_JSON")"
SOURCE_ROOT="$(dirname "$SOURCE_VIDEO_DIR")"
RAW_ZIP_PATH="${SOURCE_ROOT}/${RAW_ZIP_NAME}"

download_raw_dataset() {
  echo "Downloading NExTQA raw videos from Hugging Face"
  echo "  repo: $HF_REPO_ID"
  echo "  into: $SOURCE_ROOT"
  echo "This downloads a large zip file and can take a long time."

  local download_cmd=()
  if command -v hf >/dev/null 2>&1; then
    download_cmd=(hf download "$HF_REPO_ID")
  elif command -v huggingface-cli >/dev/null 2>&1; then
    download_cmd=(huggingface-cli download "$HF_REPO_ID")
  else
    echo "ERROR: Hugging Face CLI was not found." >&2
    echo "Install it first, for example:" >&2
    echo "  uv tool install huggingface-hub" >&2
    echo "or:" >&2
    echo "  pip install -U huggingface_hub[hf_xet]" >&2
    exit 1
  fi

  download_cmd+=(--include "${HF_INCLUDE_PATTERNS[@]}")

  download_cmd+=(
    --repo-type dataset
    --local-dir "$SOURCE_ROOT"
  )

  "${download_cmd[@]}"
}

extract_raw_zip() {
  if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: unzip is required but was not found in PATH." >&2
    exit 1
  fi

  echo "Extracting $RAW_ZIP_PATH"
  unzip -n -q "$RAW_ZIP_PATH" -d "$SOURCE_ROOT"
}

count_source_videos() {
  if [[ ! -d "$SOURCE_VIDEO_DIR" ]]; then
    printf '0\n'
    return
  fi
  find "$SOURCE_VIDEO_DIR" -mindepth 1 -type f -name '*.mp4' | wc -l
}

mkdir -p "$SOURCE_ROOT"

source_video_count="$(count_source_videos)"
case "$DOWNLOAD_RAW" in
  always)
    download_raw_dataset
    ;;
  auto)
    if (( source_video_count == 0 )) && [[ ! -f "$RAW_ZIP_PATH" ]]; then
      download_raw_dataset
    fi
    ;;
  never)
    ;;
  *)
    echo "ERROR: DOWNLOAD_RAW must be one of: auto, always, never" >&2
    echo "Current value: $DOWNLOAD_RAW" >&2
    exit 1
    ;;
esac

source_video_count="$(count_source_videos)"
if (( source_video_count == 0 )) && [[ -f "$RAW_ZIP_PATH" ]]; then
  extract_raw_zip
fi

source_video_count="$(count_source_videos)"
if (( source_video_count == 0 )); then
  echo "ERROR: no .mp4 files found under: $SOURCE_VIDEO_DIR" >&2
  echo "Set DOWNLOAD_RAW=always to download ${RAW_ZIP_NAME}, or point SOURCE_VIDEO_DIR at an extracted NExTVideo folder." >&2
  exit 1
fi

mkdir -p "$TARGET_VIDEO_DIR"

echo "Creating flat NExTQA video symlinks"
echo "  from nested source: $SOURCE_VIDEO_DIR"
echo "  into flat target:   $TARGET_VIDEO_DIR"

linked_count=0
skipped_count=0
while IFS= read -r source_video; do
  target_video="${TARGET_VIDEO_DIR}/$(basename "$source_video")"
  if [[ -L "$target_video" ]]; then
    ln -sfn "$source_video" "$target_video"
    linked_count=$((linked_count + 1))
  elif [[ -e "$target_video" ]]; then
    skipped_count=$((skipped_count + 1))
  else
    ln -s "$source_video" "$target_video"
    linked_count=$((linked_count + 1))
  fi
done < <(find "$SOURCE_VIDEO_DIR" -mindepth 1 -type f -name '*.mp4' | sort)

echo "Source videos found: $source_video_count"
echo "Symlinks created or refreshed: $linked_count"
if (( skipped_count > 0 )); then
  echo "Existing non-symlink target files left untouched: $skipped_count"
fi

if [[ -f "$QA_JSON" ]]; then
  echo "Validating videos referenced by $QA_JSON"
  python3 - "$QA_JSON" "$TARGET_VIDEO_DIR" <<'PY'
import json
import sys
from pathlib import Path

qa_json = Path(sys.argv[1])
video_dir = Path(sys.argv[2])

with qa_json.open() as f:
    rows = json.load(f)

video_names = sorted({row["video_name"] for row in rows})
missing = [name for name in video_names if not (video_dir / name).is_file()]

print(f"QA rows: {len(rows)}")
print(f"Unique videos referenced: {len(video_names)}")

if missing:
    print(f"Missing videos: {len(missing)}", file=sys.stderr)
    for name in missing[:20]:
        print(f"  {name}", file=sys.stderr)
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
    sys.exit(1)

print("All referenced videos are present.")
PY
else
  echo "Skipping video-reference validation because JSON was not found: $QA_JSON"
fi

echo "NExTQA dataset is ready:"
echo "  video_path=$TARGET_VIDEO_DIR"
echo "  gt_file=$QA_JSON"
