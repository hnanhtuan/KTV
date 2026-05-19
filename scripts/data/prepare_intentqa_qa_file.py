#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json


def main(args, task_name="IntentQA"):
    data_list_info = []
    with open(args.qa_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        field_names = set(reader.fieldnames or [])

        # Legacy format:
        # ,video_name,question,question_id,answer,video_id,question_type,a0,a1,a2,a3,a4
        is_legacy = {"video_name", "question_id", "question", "answer", "a0", "a1", "a2", "a3", "a4"}.issubset(field_names)
        # hamedrahimi/IntentQA format:
        # video_id,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4,...
        is_hf_split = {"video_id", "qid", "question", "answer", "a0", "a1", "a2", "a3", "a4"}.issubset(field_names)

        if not is_legacy and not is_hf_split:
            raise ValueError(
                f"Unsupported IntentQA CSV schema. Found fields: {sorted(field_names)}"
            )

        for row in reader:
            candidates = [row["a0"], row["a1"], row["a2"], row["a3"], row["a4"]]
            if is_legacy:
                answer_text = row["answer"]
                if answer_text not in candidates:
                    raise ValueError(f"Answer '{answer_text}' not in candidates for question_id={row.get('question_id')}")
                answer_number = candidates.index(answer_text)
                question_id = row["question_id"]
                video_name = row["video_name"]
            else:
                answer_idx = int(row["answer"])
                if answer_idx < 0 or answer_idx >= len(candidates):
                    raise ValueError(f"Answer index out of range: {answer_idx} for qid={row.get('qid')}")
                answer_number = answer_idx
                answer_text = candidates[answer_idx]
                question_id = row["qid"]
                video_name = row["video_id"]

            data_list_info.append(
                {
                    "task_name": task_name,
                    "video_name": f"{video_name}.mp4",
                    "question_id": question_id,
                    "question": row["question"],
                    "answer_number": answer_number,
                    "candidates": candidates,
                    "answer": answer_text,
                }
            )

    folder = args.output_dir or f"playground/gt_qa_files/{task_name}"
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "val_qa.json"), "w") as f:
        json.dump(data_list_info, f, indent=4)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", help="Path to IntentQA.csv", required=True)
    parser.add_argument("--output_dir", help="Output folder for val_qa.json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
