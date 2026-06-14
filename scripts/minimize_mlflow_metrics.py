#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Minimize MLflow metrics on disk to only keep overall accuracy.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete files (dry-run by default)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    confirm = args.confirm

    repo_root = Path(__file__).resolve().parents[1]
    mlruns_dir = repo_root / "mlruns"

    if not mlruns_dir.exists():
        print(f"Error: mlruns directory not found at {mlruns_dir}")
        return

    # Find all metrics folders
    metrics_dirs = list(mlruns_dir.glob("**/metrics"))
    print(f"Found {len(metrics_dirs)} runs with metrics.")

    deleted_count = 0
    kept_count = 0
    runs_modified = 0

    to_delete = []

    for metrics_dir in metrics_dirs:
        if not metrics_dir.is_dir():
            continue

        run_modified = False
        for child in metrics_dir.iterdir():
            if child.is_file():
                if child.name in ("overall.accuracy", "overall.accuracy_percent"):
                    kept_count += 1
                else:
                    to_delete.append(child)
                    deleted_count += 1
                    run_modified = True
        if run_modified:
            runs_modified += 1

    if not to_delete:
        print("No metrics need to be cleaned up. All runs already only show overall.accuracy.")
        return

    if not confirm:
        print(f"\n=== DRY RUN: Cleaning up {deleted_count} metric files in {runs_modified} runs ===")
        print("Run with '--confirm' to execute the deletions.\n")
        print("Sample of files that would be deleted:")
        for path in to_delete[:20]:
            print(f"  {path.relative_to(repo_root)}")
        if len(to_delete) > 20:
            print(f"  ... and {len(to_delete) - 20} more files.")
        print(f"\nTotal files to delete: {len(to_delete)}")
        print(f"Total files to keep: {kept_count}")
    else:
        print(f"\n=== EXECUTING CLEANUP OF {deleted_count} METRIC FILES ===")
        success_count = 0
        for path in to_delete:
            try:
                path.unlink()
                success_count += 1
            except Exception as e:
                print(f"Failed to delete {path.relative_to(repo_root)}: {e}")
        print(f"\nCleanup complete: deleted {success_count}/{len(to_delete)} files.")

if __name__ == "__main__":
    main()
