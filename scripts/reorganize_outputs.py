#!/usr/bin/env python3
import os
import re
import shutil
import argparse
from pathlib import Path

DATASETS = ["egoschema", "intentqa", "nextqa", "videomme"]

# Standard variants mapped to their folder names
VARIANTS = [
    "baseline_uniform_frames",
    "ktv_keyframe_only",
    "ktv_token_only_cls_new_token_sim",
    "ktv_token_only_uniform_token",
    "ktv_full_cls_new_token_sim",
    "ktv_full_uniform_token",
    "temporal_chain",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Reorganize KTV experiment output files.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually execute the reorganization (dry-run by default)."
    )
    return parser.parse_args()

def parse_file(filename, dataset):
    # Strip dataset prefix
    if not filename.startswith(dataset + "_"):
        # Special case: unprefixed files matching dataset name
        if filename == f"{dataset}.json":
            return {
                "dest_dir": f"outputs/{dataset}/legacy",
                "dest_name": "predictions.json"
            }
        if filename == f"{dataset}_accuracy.txt":
            return {
                "dest_dir": f"outputs/{dataset}/legacy",
                "dest_name": "predictions_accuracy.txt"
            }
        return None
    
    stem = filename[len(dataset) + 1:]
    
    # Check special cases
    if stem == "keyframe6_order.json":
        return {
            "dest_dir": f"outputs/{dataset}/keyframe_selection",
            "dest_name": "keyframe6_order.json"
        }
    if stem == "temporal_chain_keyframe6_order.json":
        return {
            "dest_dir": f"outputs/{dataset}/temporal_chain",
            "dest_name": "keyframe6_order.json"
        }
    if stem in ("temporal_chain_qa.json", "temporal_chain_qa_accuracy.txt", "temporal_chain_qa_accuracy.json", "temporal_chain_qa_summary.json"):
        ext = stem.split(".")[-1]
        name_map = {
            "json": "predictions.json",
            "txt": "predictions_accuracy.txt",
        }
        if "accuracy.json" in stem:
            dest_name = "predictions_accuracy.json"
        elif "summary.json" in stem:
            dest_name = "predictions_summary.json"
        else:
            dest_name = name_map.get(ext, stem)
            
        return {
            "dest_dir": f"outputs/{dataset}/temporal_chain",
            "dest_name": dest_name
        }
        
    # Extract token number if present (e.g. _tokens1872)
    token_match = re.search(r"_tokens(\d+)", stem)
    tokens_num = token_match.group(1) if token_match else None
    
    # Remove _tokens\d+ from the stem for folder classification
    if tokens_num:
        clean_stem = stem.replace(f"_tokens{tokens_num}", "")
    else:
        clean_stem = stem
        
    # Identify file type
    if clean_stem.endswith("_accuracy.txt"):
        file_type = "accuracy_txt"
        variant_name = clean_stem[:-13]
    elif clean_stem.endswith("_accuracy.json"):
        file_type = "accuracy_json"
        variant_name = clean_stem[:-14]
    elif clean_stem.endswith("_summary.json"):
        file_type = "summary_json"
        variant_name = clean_stem[:-13]
    elif clean_stem.endswith(".json"):
        file_type = "prediction"
        variant_name = clean_stem[:-5]
    else:
        return None
        
    # Classify variant folder
    dest_dir = f"outputs/{dataset}/{variant_name}"
    
    # Determine new filename
    if tokens_num:
        if file_type == "prediction":
            dest_name = f"predictions_tokens{tokens_num}.json"
        elif file_type == "accuracy_txt":
            dest_name = f"predictions_tokens{tokens_num}_accuracy.txt"
        elif file_type == "accuracy_json":
            dest_name = f"predictions_tokens{tokens_num}_accuracy.json"
        elif file_type == "summary_json":
            dest_name = f"predictions_tokens{tokens_num}_summary.json"
    else:
        if file_type == "prediction":
            dest_name = "predictions.json"
        elif file_type == "accuracy_txt":
            dest_name = "predictions_accuracy.txt"
        elif file_type == "accuracy_json":
            dest_name = "predictions_accuracy.json"
        elif file_type == "summary_json":
            dest_name = "predictions_summary.json"
            
    return {
        "dest_dir": dest_dir,
        "dest_name": dest_name
    }

def parse_directory(dir_name, dataset):
    if dir_name == f"{dataset}_keyframes":
        return f"outputs/{dataset}/keyframe_selection/keyframes"
    if dir_name == f"{dataset}_temporal_chain_keyframes":
        return f"outputs/{dataset}/temporal_chain/keyframes"
    return None

def main():
    args = parse_args()
    confirm = args.confirm
    
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "outputs"
    
    if not outputs_root.exists():
        print(f"Error: outputs directory not found at {outputs_root}")
        return
        
    moves = []
    
    for dataset in DATASETS:
        dataset_dir = outputs_root / dataset
        if not dataset_dir.exists():
            continue
            
        for child in sorted(dataset_dir.iterdir()):
            if child.is_dir():
                dest_rel = parse_directory(child.name, dataset)
                if dest_rel:
                    dest_path = repo_root / dest_rel
                    moves.append((child, dest_path, True)) # src, dest, is_dir
            elif child.is_file():
                mapping = parse_file(child.name, dataset)
                if mapping:
                    dest_dir = repo_root / mapping["dest_dir"]
                    dest_path = dest_dir / mapping["dest_name"]
                    moves.append((child, dest_path, False)) # src, dest, is_dir
                    
    if not moves:
        print("No files or directories found in the root of dataset output folders to reorganize.")
        return
        
    if not confirm:
        print("=== DRY RUN: Reorganizing Output Files (no changes made) ===")
        print("Run with '--confirm' to execute the changes.\n")
        for src, dest, is_dir in moves:
            src_rel = src.relative_to(repo_root)
            dest_rel = dest.relative_to(repo_root)
            type_str = "Folder" if is_dir else "File  "
            print(f"[{type_str}]  {src_rel}  ->  {dest_rel}")
        print(f"\nTotal items to move: {len(moves)}")
    else:
        print("=== EXECUTING REORGANIZATION ===")
        success_count = 0
        failure_count = 0
        for src, dest, is_dir in moves:
            try:
                # Ensure destination parent directory exists
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                # Move
                if dest.exists():
                    # If target exists, resolve conflict or warning
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                
                shutil.move(str(src), str(dest))
                print(f"Moved: {src.relative_to(repo_root)} -> {dest.relative_to(repo_root)}")
                success_count += 1
            except Exception as e:
                print(f"Failed to move {src.name}: {e}")
                failure_count += 1
                
        print(f"\nMigration complete: success={success_count}, failures={failure_count}")

if __name__ == "__main__":
    main()
