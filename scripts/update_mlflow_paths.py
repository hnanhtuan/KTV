#!/usr/bin/env python3
import os
import re
import hashlib
import argparse
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

DATASETS = ["egoschema", "intentqa", "nextqa", "videomme"]

def parse_args():
    parser = argparse.ArgumentParser(description="Update MLflow run tags to reflect reorganized output paths.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually update the run tags in MLflow (dry-run by default)."
    )
    return parser.parse_args()

def parse_file(filename, dataset):
    # Strip dataset prefix
    if not filename.startswith(dataset + "_"):
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
        clean_stem = clean_stem = stem
        
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

def map_path(old_path_str, repo_root):
    try:
        old_path = Path(old_path_str).resolve()
        relative_path = old_path.relative_to(repo_root.resolve())
    except ValueError:
        return None

    parts = relative_path.parts
    if len(parts) < 3 or parts[0] != "outputs":
        return None

    dataset = parts[1]
    if dataset not in DATASETS:
        return None

    # We only map paths that are directly inside outputs/<dataset>/
    if len(parts) == 3:
        filename = parts[2]
        mapping = parse_file(filename, dataset)
        if mapping:
            return repo_root / mapping["dest_dir"] / mapping["dest_name"]
    elif len(parts) > 3:
        # Check if the top-level directory needs mapping
        top_dir = parts[2]
        mapped_top_dir = parse_directory(top_dir, dataset)
        if mapped_top_dir:
            # Reconstruct the path with the mapped directory
            sub_parts = parts[3:]
            return repo_root / mapped_top_dir / Path(*sub_parts)
            
    return None

def compute_sha1(path):
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()

def main():
    # Allow filesystem tracking backend
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    args = parse_args()
    confirm = args.confirm

    repo_root = Path(__file__).resolve().parents[1]
    tracking_uri = (repo_root / "mlruns").resolve().as_uri()
    
    print(f"Connecting to MLflow Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name("ktv")
    if experiment is None:
        print("Error: Experiment 'ktv' not found.")
        return

    runs = client.search_runs(experiment.experiment_id, max_results=10000)
    print(f"Found {len(runs)} runs in experiment 'ktv'.")

    updates = []
    for run in runs:
        tags = run.data.tags
        params = run.data.params
        run_updates = {}

        # 1. Path mapping (only if backfill path exists)
        source_path_val = tags.get("ktv_backfill_source_path")
        if source_path_val:
            new_path = map_path(source_path_val, repo_root)
            if new_path:
                new_path_str = str(new_path.resolve())
                if new_path_str != source_path_val:
                    new_sha1 = compute_sha1(new_path)
                    run_updates["ktv_backfill_source_path"] = new_path_str
                    run_updates["ktv_backfill_source_sha1"] = new_sha1

        # 2. Dataset tag detection
        current_dataset = tags.get("dataset")
        detected_dataset = None
        
        # Determine from source path (either original or mapped)
        path_to_check = run_updates.get("ktv_backfill_source_path") or source_path_val
        if path_to_check:
            for ds in DATASETS:
                if f"/outputs/{ds}/" in path_to_check or f"outputs/{ds}/" in path_to_check:
                    detected_dataset = ds
                    break
        
        # Determine from run name or runName tag
        if not detected_dataset:
            run_name = run.info.run_name or ""
            run_name_tag = tags.get("mlflow.runName") or ""
            for name in (run_name, run_name_tag):
                for ds in DATASETS:
                    if ds in name.lower():
                        detected_dataset = ds
                        break
                if detected_dataset:
                    break
                    
        # Determine from params
        if not detected_dataset:
            dataset_param = params.get("dataset") or params.get("shell.dataset")
            if dataset_param and dataset_param.lower() in DATASETS:
                detected_dataset = dataset_param.lower()

        # If detected and different/missing, add to updates
        if detected_dataset and current_dataset != detected_dataset:
            run_updates["dataset"] = detected_dataset

        if run_updates:
            updates.append((run.info.run_id, run.info.run_name, run_updates))

    if not updates:
        print("No run tags need to be updated. All runs are in sync with current folder structure and have dataset tags.")
        return

    if not confirm:
        print(f"\n=== DRY RUN: Updating tags for {len(updates)} runs in MLflow ===")
        print("Run with '--confirm' to apply changes.\n")
        for run_id, run_name, run_updates in updates:
            print(f"Run: {run_name} ({run_id})")
            for tag_name, new_value in run_updates.items():
                print(f"  {tag_name} -> {new_value}")
            print("-" * 40)
        print(f"\nTotal runs to update: {len(updates)}")
    else:
        print(f"\n=== EXECUTING TAG UPDATE FOR {len(updates)} RUNS ===")
        success_count = 0
        for run_id, run_name, run_updates in updates:
            try:
                for tag_name, new_value in run_updates.items():
                    client.set_tag(run_id, tag_name, new_value)
                print(f"Updated: {run_name} ({run_id})")
                success_count += 1
            except Exception as e:
                print(f"Failed to update run {run_id}: {e}")
        print(f"\nUpdate complete: {success_count} runs updated successfully.")

if __name__ == "__main__":
    main()
