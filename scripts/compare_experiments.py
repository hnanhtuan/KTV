#!/usr/bin/env python3
"""MLflow experiment comparison and visualization script.

Fetches runs from the local MLflow store, formats comparative tables,
and plots Pareto frontiers (Accuracy vs. Tokens Budget).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Adjust path to find ktv core packages
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from ktv.core.tracking import repo_tracking_uri


def fetch_all_runs(experiment_name="ktv"):
    tracking_uri = repo_tracking_uri(REPO_ROOT)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found in tracking store: {tracking_uri}")
        return []
    
    runs = client.search_runs(experiment.experiment_id, max_results=1000)
    return runs


def compile_runs(runs):
    compiled = []
    for run in runs:
        params = run.data.params
        tags = run.data.tags
        metrics = run.data.metrics
        
        # Determine the selection method
        method = tags.get("stage") or tags.get("method_family") or params.get("prune_mode") or "unknown"
        if "temporal_chain" in method:
            method = "Temporal Chain"
        elif "keyframe_ranking" in method or "clustering" in method:
            method = "Clustering (CLIP)"
        elif "query_aware" in method:
            method = "Query-Aware"
        elif "inference" in method or "qa_inference" in method:
            prune_mode = params.get("prune_mode")
            if prune_mode == "cls_new_token_sim":
                method = "KTV (Token Pruning)"
            elif prune_mode == "uniform_token":
                method = "Uniform Token Pruning"
            else:
                method = "Baseline/Inference"
                
        dataset = tags.get("dataset") or params.get("dataset") or "unknown"
        tokens = params.get("tokens_num") or params.get("tokens") or "unknown"
        accuracy = metrics.get("overall.accuracy") or metrics.get("accuracy")
        
        if accuracy is None:
            continue
            
        compiled.append({
            "run_id": run.info.run_id,
            "run_name": tags.get("mlflow.runName") or run.info.run_name,
            "dataset": dataset,
            "method": method,
            "tokens_num": tokens,
            "accuracy": float(accuracy),
            "duration": metrics.get("duration_seconds")
        })
    return compiled


def print_comparison_tables(data):
    if not data:
        print("No metrics/runs discovered with overall accuracy.")
        return
        
    # Group by dataset
    datasets = set(item["dataset"] for item in data)
    for ds in sorted(datasets):
        ds_data = [item for item in data if item["dataset"] == ds]
        print(f"\n### Dataset: {ds.upper()}")
        print("| Method | Tokens Budget | Accuracy | Run Name |")
        print("| :--- | :---: | :---: | :--- |")
        
        # Sort by Method name, then token count (descending), then accuracy (descending)
        def sort_key(x):
            try:
                tokens = int(x["tokens_num"])
            except ValueError:
                tokens = 0
            return (x["method"], -tokens, -x["accuracy"])
            
        for item in sorted(ds_data, key=sort_key):
            acc_str = f"{item['accuracy'] * 100:.2f}%" if item['accuracy'] <= 1.0 else f"{item['accuracy']:.2f}%"
            print(f"| {item['method']} | {item['tokens_num']} | {acc_str} | {item['run_name']} |")


def plot_pareto_frontiers(data, output_dir="reports"):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("\n[Note] pandas, matplotlib, or seaborn not installed. Skipping Pareto frontier plotting.")
        return
        
    df = pd.DataFrame(data)
    # Filter out unknown/invalid tokens
    df = df[df["tokens_num"] != "unknown"]
    try:
        df["tokens_num"] = df["tokens_num"].astype(int)
    except Exception:
        pass
        
    os.makedirs(output_dir, exist_ok=True)
    datasets = df["dataset"].unique()
    
    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        if ds_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        # Group duplicates by taking the best accuracy for each (method, tokens)
        pivot_df = ds_df.groupby(["method", "tokens_num"], as_index=False)["accuracy"].max()
        
        sns.lineplot(
            data=pivot_df, 
            x="tokens_num", 
            y="accuracy", 
            hue="method", 
            marker="o",
            linewidth=2.5,
            markersize=8
        )
        
        plt.title(f"Pareto Frontier: Accuracy vs. Tokens Budget on {ds.upper()}", fontsize=14, pad=15)
        plt.xlabel("Tokens Budget", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(title="Method", frameon=True)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"pareto_frontier_{ds}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated Pareto Frontier plot: {plot_path}")


def main():
    print("Fetching runs from MLflow store...")
    runs = fetch_all_runs()
    data = compile_runs(runs)
    print(f"Successfully compiled {len(data)} runs.")
    print_comparison_tables(data)
    plot_pareto_frontiers(data)


if __name__ == "__main__":
    main()
