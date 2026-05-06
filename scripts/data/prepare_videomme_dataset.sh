#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_DIR="${1:-/media/anhxtuan/f4f67dc7-d85d-43b1-a0a7-e8da7f48f352/Documents/Datasets/Video-MME}"
TARGET_ROOT="${2:-datasets/Video-MME}"
QA_CSV="${3:-playground/gt_qa_files/Videomme/val_qa.csv}"
QA_OUTPUT_DIR="${4:-playground/gt_qa_files/Videomme}"
HF_REPO_ID="${HF_REPO_ID:-lmms-lab/Video-MME}"
DOWNLOAD_RAW="${DOWNLOAD_RAW:-auto}"
HF_INCLUDE_PATTERNS=(
  "videos_chunked_*.zip"
  "subtitle.zip"
  "README.md"
  "videomme"
  "videomme/*"
)

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

SOURCE_DIR="$(resolve_path "$SOURCE_DIR")"
TARGET_ROOT="$(resolve_path "$TARGET_ROOT")"
QA_CSV="$(resolve_path "$QA_CSV")"
QA_OUTPUT_DIR="$(resolve_path "$QA_OUTPUT_DIR")"
SOURCE_DATA_DIR="${SOURCE_DIR}/data"
TARGET_DATA_DIR="${TARGET_ROOT}/data"
QA_JSON="${QA_OUTPUT_DIR}/val_qa.json"

if ! command -v unzip >/dev/null 2>&1; then
  echo "ERROR: unzip is required but was not found in PATH." >&2
  exit 1
fi

mkdir -p "$SOURCE_DIR"

shopt -s nullglob
zip_files=("${SOURCE_DIR}"/*.zip)

download_raw_dataset() {
  echo "Downloading VideoMME raw dataset from Hugging Face"
  echo "  repo: $HF_REPO_ID"
  echo "  into: $SOURCE_DIR"
  echo "This downloads large video zip chunks and can take a long time."

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

  for pattern in "${HF_INCLUDE_PATTERNS[@]}"; do
    download_cmd+=(--include "$pattern")
  done

  download_cmd+=(
    --repo-type dataset
    --local-dir "$SOURCE_DIR"
  )

  "${download_cmd[@]}"
}

case "$DOWNLOAD_RAW" in
  always)
    download_raw_dataset
    ;;
  auto)
    if (( ${#zip_files[@]} == 0 )); then
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

zip_files=("${SOURCE_DIR}"/*.zip)
if (( ${#zip_files[@]} == 0 )); then
  echo "ERROR: no .zip files found in: $SOURCE_DIR" >&2
  echo "Set DOWNLOAD_RAW=always to download them, or place the VideoMME zip files there manually." >&2
  exit 1
fi

mkdir -p "$SOURCE_DATA_DIR"
mkdir -p "$TARGET_ROOT"

echo "Extracting ${#zip_files[@]} VideoMME zip files"
echo "  from: $SOURCE_DIR"
echo "  into: $SOURCE_DIR"

for zip_file in "${zip_files[@]}"; do
  echo "Extracting $(basename "$zip_file")"
  unzip -n -q "$zip_file" -d "$SOURCE_DIR"
done

if [[ -L "$TARGET_DATA_DIR" ]]; then
  ln -sfn "$SOURCE_DATA_DIR" "$TARGET_DATA_DIR"
elif [[ -e "$TARGET_DATA_DIR" ]]; then
  echo "ERROR: target data path already exists and is not a symlink: $TARGET_DATA_DIR" >&2
  echo "Move it aside, then rerun this script so it can create a symlink." >&2
  exit 1
else
  ln -s "$SOURCE_DATA_DIR" "$TARGET_DATA_DIR"
fi

video_count="$(find "$SOURCE_DATA_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
echo "Found ${video_count} mp4 files in $SOURCE_DATA_DIR"
echo "Linked $TARGET_DATA_DIR -> $SOURCE_DATA_DIR"

if [[ -f "$QA_CSV" ]]; then
  echo "Generating QA JSON from $QA_CSV"
  python3 "${REPO_ROOT}/scripts/data/prepare_videomme_qa_file.py" \
    --qa_file "$QA_CSV" \
    --output_dir "$QA_OUTPUT_DIR"
else
  echo "Skipping QA JSON generation because CSV was not found: $QA_CSV"
fi

if [[ -f "$QA_JSON" ]]; then
  echo "Validating videos referenced by $QA_JSON"
  python3 - "$QA_JSON" "$TARGET_DATA_DIR" <<'PY'
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

echo "VideoMME dataset is ready:"
echo "  video_dir=$TARGET_DATA_DIR"
echo "  gt_file=$QA_JSON"
