#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json
from pathlib import Path


def _normalize_candidates(options_raw):
    if isinstance(options_raw, list):
        return [str(x).strip() for x in options_raw]
    if isinstance(options_raw, tuple):
        return [str(x).strip() for x in options_raw]
    if hasattr(options_raw, "tolist"):
        converted = options_raw.tolist()
        if isinstance(converted, (list, tuple)):
            return [str(x).strip() for x in converted]
    if isinstance(options_raw, str):
        text = options_raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Unsupported options format: {type(options_raw)}")


def _resolve_answer(candidates, answer_raw):
    if isinstance(answer_raw, str):
        answer_clean = answer_raw.strip()
        if answer_clean in candidates:
            return answer_clean, candidates.index(answer_clean)
        if answer_clean in ("A", "B", "C", "D", "E"):
            idx = ord(answer_clean) - ord("A")
            return candidates[idx], idx
        if answer_clean.isdigit():
            idx = int(answer_clean)
            if 0 <= idx < len(candidates):
                return candidates[idx], idx
    if isinstance(answer_raw, (int, float)):
        idx = int(answer_raw)
        if 0 <= idx < len(candidates):
            return candidates[idx], idx
    raise ValueError(f"Cannot resolve answer: {answer_raw}")


def main(args, task_name="EgoSchema"):
    data_list_info = []
    qa_file = Path(args.qa_file)
    if qa_file.suffix.lower() == ".csv":
        with open(qa_file, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            # ,video_name,question_id,question,answer,a0,a1,a2,a3,a4
            for idx, row in enumerate(spamreader):
                if idx == 0:
                    continue
                _, video_name, question_id, question, answer, a0, a1, a2, a3, a4 = row
                candidates = [a0, a1, a2, a3, a4]
                assert answer in candidates
                data_list_info.append(
                    {
                        "task_name": task_name,
                        "video_name": f"{video_name}.mp4",
                        "question_id": question_id,
                        "question": question,
                        "answer_number": candidates.index(answer),
                        "candidates": candidates,
                        "answer": answer,
                    }
                )
    else:
        import pandas as pd

        df = pd.read_parquet(qa_file)
        required = {"question_idx", "question", "video_idx", "option", "answer"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Unsupported schema in {qa_file}. Missing columns: {sorted(missing)}"
            )
        for row in df.itertuples(index=False):
            if row.answer is None:
                continue
            candidates = _normalize_candidates(row.option)
            answer_text, answer_number = _resolve_answer(candidates, row.answer)
            data_list_info.append(
                {
                    "task_name": task_name,
                    "video_name": f"{row.video_idx}.mp4",
                    "question_id": str(row.question_idx),
                    "question": str(row.question),
                    "answer_number": answer_number,
                    "candidates": candidates,
                    "answer": answer_text,
                }
            )
        if len(data_list_info) == 0:
            raise ValueError(
                f"No answerable rows found in {qa_file}. "
                "Try using a split/file with released answers (e.g., Subset)."
            )

    folder = args.output_dir
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa_file",
        help="Path to EgoSchema QA file (csv or parquet)",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where val_qa.json will be written",
        default="playground/gt_qa_files/EgoSchema",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
