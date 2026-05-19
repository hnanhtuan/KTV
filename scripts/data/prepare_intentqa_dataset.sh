#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_DIR="${1:-datasets_raw/IntentQA}"
TARGET_VIDEO_DIR="${2:-datasets/IntentQA/videos}"
QA_CSV="${3:-playground/gt_qa_files/IntentQA/IntentQA.csv}"
QA_OUTPUT_DIR="${4:-playground/gt_qa_files/IntentQA}"
HF_REPO_ID="${HF_REPO_ID:-hamedrahimi/IntentQA}"
DOWNLOAD_RAW="${DOWNLOAD_RAW:-auto}"
REQUIRE_VIDEOS="${REQUIRE_VIDEOS:-0}"
DOWNLOAD_YOUTUBE_VIDEOS="${DOWNLOAD_YOUTUBE_VIDEOS:-0}"
HF_INCLUDE_PATTERNS=(
  "*.zip"
  "*/*.zip"
  "*/*/*.zip"
  "*.mp4"
  "*/*.mp4"
  "*/*/*.mp4"
  "*/*/*/*.mp4"
  "*.webm"
  "*/*.webm"
  "*/*/*.webm"
  "*/*/*/*.webm"
  "*.csv"
  "*/*.csv"
  "*/*/*.csv"
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
  echo "Downloading IntentQA raw files from Hugging Face"
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

  local include_pattern
  for include_pattern in "${HF_INCLUDE_PATTERNS[@]}"; do
    download_cmd+=(--include "$include_pattern")
  done

  download_cmd+=(
    --repo-type dataset
    --local-dir "$SOURCE_DIR"
  )

  "${download_cmd[@]}"
}

count_source_videos() {
  find "$SOURCE_DIR" -type f -name '*.mp4' | wc -l
}

download_videos_from_youtube() {
  local qa_csv="$1"
  local download_dir="${SOURCE_DIR}/videos"

  if ! command -v yt-dlp >/dev/null 2>&1; then
    echo "ERROR: yt-dlp is required for YouTube video download but was not found." >&2
    echo "Install it first, for example:" >&2
    echo "  uv tool install yt-dlp" >&2
    exit 1
  fi

  mkdir -p "$download_dir"

  echo "Downloading videos from YouTube using video_id values in $qa_csv"
  echo "  output dir: $download_dir"
  echo "  this can take a long time and some videos may be unavailable."

  python3 - "$qa_csv" "$download_dir" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

qa_csv = Path(sys.argv[1])
download_dir = Path(sys.argv[2])

with qa_csv.open(newline="") as f:
    reader = csv.DictReader(f)
    ids = []
    for row in reader:
        v = (row.get("video_id") or row.get("video_name") or "").strip()
        if v:
            ids.append(v)

video_ids = sorted(set(ids))
print(f"Unique video ids to download: {len(video_ids)}")

ok = 0
failed = 0

for idx, vid in enumerate(video_ids, start=1):
    url = f"https://www.youtube.com/watch?v={vid}"
    cmd = [
        "yt-dlp",
        "--no-overwrites",
        "--ignore-errors",
        "--no-warnings",
        "--format",
        "mp4/bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--output",
        str(download_dir / "%(id)s.%(ext)s"),
        url,
    ]
    print(f"[{idx}/{len(video_ids)}] {vid}")
    ret = subprocess.run(cmd).returncode
    if ret == 0:
        ok += 1
    else:
        failed += 1

print(f"YouTube download finished: ok={ok}, failed={failed}")
PY
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
    find "$SOURCE_DIR" -type f \( -iname '*intentqa*.csv' -o -iname '*qa*.csv' -o -iname 'val.csv' -o -iname 'test.csv' -o -iname 'train.csv' \) | sort
  )
  if (( ${#qa_candidates[@]} > 0 )); then
    mapfile -t prioritized_qa_candidates < <(
      printf "%s\n" "${qa_candidates[@]}" | awk '
      /\/val\.csv$/ {print "0 " $0; next}
      /\/test\.csv$/ {print "1 " $0; next}
      /\/train\.csv$/ {print "2 " $0; next}
      {print "3 " $0}
      ' | sort -n -k1,1 -k2,2 | cut -d' ' -f2-
    )
    QA_CSV="${prioritized_qa_candidates[0]}"
    echo "Using discovered QA file: $QA_CSV"
  else
    echo "ERROR: QA file not found at $QA_CSV and no matching CSV discovered under $SOURCE_DIR" >&2
    exit 1
  fi
fi

video_count="$(count_source_videos)"
if (( video_count == 0 )); then
  echo "No local videos found after Hugging Face download."
  if [[ "$DOWNLOAD_YOUTUBE_VIDEOS" == "1" ]]; then
    download_videos_from_youtube "$QA_CSV"
    video_count="$(count_source_videos)"
  fi

  if [[ "$REQUIRE_VIDEOS" == "1" ]]; then
    echo "ERROR: no .mp4 files found under: $SOURCE_DIR" >&2
    echo "Set DOWNLOAD_YOUTUBE_VIDEOS=1 to attempt download by video_id, or place videos manually in SOURCE_DIR." >&2
    exit 1
  fi
  echo "WARNING: no .mp4 files found under $SOURCE_DIR"
  echo "Continuing to prepare QA file only."
  echo "Set DOWNLOAD_YOUTUBE_VIDEOS=1 to download from YouTube, or REQUIRE_VIDEOS=1 to enforce local video presence."
else
  mkdir -p "$TARGET_VIDEO_DIR"

  echo "Creating flat IntentQA video symlinks"
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
fi

echo "Generating QA JSON from $QA_CSV"
python3 "${REPO_ROOT}/scripts/data/prepare_intentqa_qa_file.py" \
  --qa_file "$QA_CSV" \
  --output_dir "$QA_OUTPUT_DIR"

if [[ -f "$QA_JSON" ]] && (( video_count > 0 )); then
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

echo "IntentQA dataset is ready:"
echo "  video_path=$TARGET_VIDEO_DIR"
echo "  gt_file=$QA_JSON"
