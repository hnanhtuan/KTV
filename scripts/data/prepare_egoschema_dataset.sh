#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_DIR="${1:-datasets_raw/EgoSchema}"
TARGET_VIDEO_DIR="${2:-datasets/EgoSchema/videos}"
QA_CSV="${3:-playground/gt_qa_files/EgoSchema/EgoSchema.csv}"
QA_OUTPUT_DIR="${4:-playground/gt_qa_files/EgoSchema}"
HF_REPO_ID="${HF_REPO_ID:-VLM2Vec/EgoSchema}"
DOWNLOAD_RAW="${DOWNLOAD_RAW:-auto}"
HF_INCLUDE_PATTERNS=(
  "*.zip"
  "*/*.zip"
  "*/*/*.zip"
  "*.csv"
  "*/*.csv"
  "*/*/*.csv"
  "*.parquet"
  "*/*.parquet"
  "*/*/*.parquet"
  "README.md"
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
TARGET_VIDEO_DIR="$(resolve_path "$TARGET_VIDEO_DIR")"
QA_CSV="$(resolve_path "$QA_CSV")"
QA_OUTPUT_DIR="$(resolve_path "$QA_OUTPUT_DIR")"
QA_JSON="${QA_OUTPUT_DIR}/val_qa.json"

mkdir -p "$SOURCE_DIR"

download_raw_dataset() {
  echo "Downloading EgoSchema raw files from Hugging Face"
  echo "  repo: $HF_REPO_ID"
  echo "  into: $SOURCE_DIR"

  local download_cmd=()
  if command -v hf >/dev/null 2>&1; then
    download_cmd=(hf download "$HF_REPO_ID")
  elif command -v huggingface-cli >/dev/null 2>&1; then
    download_cmd=(huggingface-cli download "$HF_REPO_ID")
  else
    echo "ERROR: Hugging Face CLI was not found." >&2
    echo "Install it first, for example:" >&2
    echo "  uv tool install huggingface-hub" >&2
    exit 1
  fi

  # Some hf CLI versions do not honor multiple --include patterns reliably.
  # Prefer enumerating files via HF API and downloading each explicit repo path.
  local tree_json
  tree_json="$(curl -sSfL "https://huggingface.co/api/datasets/${HF_REPO_ID}/tree/main?recursive=1")"

  mapfile -t selected_files < <(
    python3 - <<'PY' "$tree_json"
import json
import re
import sys

tree = json.loads(sys.argv[1])
patterns = [
    re.compile(r".*\.csv$", re.IGNORECASE),
    re.compile(r".*\.parquet$", re.IGNORECASE),
    re.compile(r".*test-\d+-of-\d+(\.\w+)?$", re.IGNORECASE),
    re.compile(r".*videos?.*chunk.*\.zip$", re.IGNORECASE),
    re.compile(r".*\.zip$", re.IGNORECASE),
]

for entry in tree:
    if entry.get("type") != "file":
        continue
    path = entry.get("path", "")
    if path == "README.md":
        print(path)
        continue
    if any(p.match(path) for p in patterns):
        print(path)
PY
  )

  if (( ${#selected_files[@]} == 0 )); then
    echo "ERROR: No candidate QA/video files discovered in HF dataset tree for $HF_REPO_ID" >&2
    exit 1
  fi

  local repo_file
  for repo_file in "${selected_files[@]}"; do
    echo "Downloading file: $repo_file"
    "${download_cmd[@]}" \
      "$repo_file" \
      --repo-type dataset \
      --local-dir "$SOURCE_DIR"
  done
}

count_source_videos() {
  find "$SOURCE_DIR" -type f -name '*.mp4' | wc -l
}

case "$DOWNLOAD_RAW" in
  always)
    download_raw_dataset
    ;;
  auto)
    if [[ ! -f "$QA_CSV" ]] || (( "$(count_source_videos)" == 0 )); then
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

if ! command -v unzip >/dev/null 2>&1; then
  echo "ERROR: unzip is required but was not found in PATH." >&2
  exit 1
fi

mapfile -t zip_files < <(find "$SOURCE_DIR" -type f -name '*.zip' | sort)
if (( ${#zip_files[@]} > 0 )); then
  echo "Extracting ${#zip_files[@]} archive(s) under $SOURCE_DIR"
  for zip_file in "${zip_files[@]}"; do
    echo "Extracting $(basename "$zip_file")"
    unzip -n -q "$zip_file" -d "$SOURCE_DIR"
  done
fi

if [[ ! -f "$QA_CSV" ]]; then
  mapfile -t qa_candidates < <(
    find "$SOURCE_DIR" -type f \( -iname '*egoschema*.csv' -o -iname '*qa*.csv' -o -iname '*.parquet' \) | sort
  )
  if (( ${#qa_candidates[@]} > 0 )); then
    # Prefer subsets/splits that usually contain released answers.
    mapfile -t prioritized_qa_candidates < <(
      printf "%s\n" "${qa_candidates[@]}" | awk '
      /\/Subset\// {print "0 " $0; next}
      /\/MC\// {print "1 " $0; next}
      /\/MC_PPL\// {print "2 " $0; next}
      /\/GENERATION\// {print "3 " $0; next}
      {print "4 " $0}
      ' | sort -n -k1,1 -k2,2 | cut -d' ' -f2-
    )
    QA_CSV="${prioritized_qa_candidates[0]}"
    echo "Using discovered QA file: $QA_CSV"
  else
    echo "ERROR: QA file not found at $QA_CSV and no matching csv/parquet discovered under $SOURCE_DIR" >&2
    exit 1
  fi
fi

video_count="$(count_source_videos)"
if (( video_count == 0 )); then
  echo "ERROR: no .mp4 files found under: $SOURCE_DIR" >&2
  echo "Set DOWNLOAD_RAW=always to download from Hugging Face, or place videos manually in SOURCE_DIR." >&2
  exit 1
fi

mkdir -p "$TARGET_VIDEO_DIR"

echo "Creating flat EgoSchema video symlinks"
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
done < <(find "$SOURCE_DIR" -type f -name '*.mp4' | sort)

echo "Source videos found: $video_count"
echo "Symlinks created or refreshed: $linked_count"
if (( skipped_count > 0 )); then
  echo "Existing non-symlink target files left untouched: $skipped_count"
fi

echo "Generating QA JSON from $QA_CSV"
python3 "${REPO_ROOT}/scripts/data/prepare_egoschema_qa_file.py" \
  --qa_file "$QA_CSV" \
  --output_dir "$QA_OUTPUT_DIR"

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
fi

echo "EgoSchema dataset is ready:"
echo "  video_path=$TARGET_VIDEO_DIR"
echo "  gt_file=$QA_JSON"
