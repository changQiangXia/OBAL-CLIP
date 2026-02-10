#!/usr/bin/env python3
"""
Cleanup script for The Architect project
Removes debug/temporary files while preserving project structure
"""

import os
import shutil
import argparse
from pathlib import Path


def get_project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


def remove_pattern(directory, pattern, dry_run=False):
    """Remove files matching pattern"""
    removed = []
    directory = Path(directory)
    if directory.exists():
        for item in directory.rglob(pattern):
            if dry_run:
                print(f"[DRY RUN] Would remove: {item}")
            else:
                try:
                    if item.is_file():
                        item.unlink()
                        removed.append(item)
                    elif item.is_dir():
                        shutil.rmtree(item)
                        removed.append(item)
                except Exception as e:
                    print(f"[ERROR] Failed to remove {item}: {e}")
    return removed


def clean_models(dry_run=False, keep_models=False):
    """Clean downloaded models in models/ directory"""
    root = get_project_root()
    models_dir = root / "models"
    
    if not models_dir.exists():
        print("[INFO] models/ directory does not exist")
        return
    
    print(f"\n{'='*60}")
    print("Cleaning models/ directory")
    print(f"{'='*60}")
    
    if keep_models:
        print("[SKIP] Keeping models (--keep-models specified)")
        return
    
    removed = []
    for item in models_dir.rglob("*"):
        if item.is_file():
            if dry_run:
                print(f"[DRY RUN] Would remove: {item}")
            else:
                try:
                    item.unlink()
                    removed.append(item)
                except Exception as e:
                    print(f"[ERROR] {e}")
        elif item.is_dir() and item != models_dir:
            if dry_run:
                print(f"[DRY RUN] Would remove directory: {item}")
            else:
                try:
                    shutil.rmtree(item)
                    removed.append(item)
                except Exception as e:
                    print(f"[ERROR] {e}")
    
    if not dry_run:
        print(f"[REMOVED] {len(removed)} model files/directories")
        # Remove empty directories
        for item in sorted(models_dir.rglob("*"), reverse=True):
            if item.is_dir() and item != models_dir:
                try:
                    item.rmdir()
                except:
                    pass


def clean_outputs(dry_run=False):
    """Clean outputs directory (checkpoints, visualizations, logs)"""
    root = get_project_root()
    outputs_dir = root / "outputs"
    
    if not outputs_dir.exists():
        print("[INFO] outputs/ directory does not exist")
        return
    
    print(f"\n{'='*60}")
    print("Cleaning outputs/ directory")
    print(f"{'='*60}")
    
    # Remove checkpoint files (keep best model optionally)
    checkpoints_dir = outputs_dir / "checkpoints"
    if checkpoints_dir.exists():
        # Keep the best model, remove intermediate checkpoints
        for ckpt in checkpoints_dir.glob("*.pt"):
            if "best" not in ckpt.name and "epoch_" in ckpt.name:
                if dry_run:
                    print(f"[DRY RUN] Would remove: {ckpt}")
                else:
                    ckpt.unlink()
                    print(f"[REMOVED] {ckpt.name}")
    
    # Remove visualizations
    viz_dir = outputs_dir / "visualizations"
    if viz_dir.exists():
        removed = remove_pattern(viz_dir, "*.png", dry_run)
        removed += remove_pattern(viz_dir, "*.jpg", dry_run)
        if not dry_run:
            print(f"[REMOVED] {len(removed)} visualization files")
    
    # Remove error analysis
    error_dir = outputs_dir / "error_analysis"
    if error_dir.exists():
        removed = remove_pattern(error_dir, "*.json", dry_run)
        if not dry_run:
            print(f"[REMOVED] {len(removed)} error analysis files")
    
    # Remove evaluation results
    for json_file in outputs_dir.glob("*.json"):
        if dry_run:
            print(f"[DRY RUN] Would remove: {json_file}")
        else:
            json_file.unlink()
            print(f"[REMOVED] {json_file.name}")


def clean_logs(dry_run=False):
    """Clean logs directory"""
    root = get_project_root()
    logs_dir = root / "logs"
    
    if not logs_dir.exists():
        print("[INFO] logs/ directory does not exist")
        return
    
    print(f"\n{'='*60}")
    print("Cleaning logs/ directory")
    print(f"{'='*60}")
    
    removed = []
    for log_file in logs_dir.glob("*"):
        if log_file.is_file():
            if dry_run:
                print(f"[DRY RUN] Would remove: {log_file}")
            else:
                log_file.unlink()
                removed.append(log_file)
    
    if not dry_run:
        print(f"[REMOVED] {len(removed)} log files")


def clean_pycache(dry_run=False):
    """Clean Python cache files"""
    root = get_project_root()
    
    print(f"\n{'='*60}")
    print("Cleaning Python cache")
    print(f"{'='*60}")
    
    removed = []
    
    # Remove __pycache__ directories
    removed += remove_pattern(root, "__pycache__", dry_run)
    
    # Remove .pyc files
    removed += remove_pattern(root, "*.pyc", dry_run)
    
    # Remove .pyo files
    removed += remove_pattern(root, "*.pyo", dry_run)
    
    if not dry_run:
        print(f"[REMOVED] {len(removed)} cache files/directories")


def clean_notebook_checkpoints(dry_run=False):
    """Clean Jupyter notebook checkpoints"""
    root = get_project_root()
    
    print(f"\n{'='*60}")
    print("Cleaning notebook checkpoints")
    print(f"{'='*60}")
    
    removed = remove_pattern(root, ".ipynb_checkpoints", dry_run)
    
    if not dry_run:
        print(f"[REMOVED] {len(removed)} checkpoint directories")


def get_directory_size(path):
    """Get total size of directory in MB"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)  # Convert to MB


def main():
    parser = argparse.ArgumentParser(
        description="Clean up The Architect project debug/temporary files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove all debug files including best checkpoint"
    )
    parser.add_argument(
        "--keep-best",
        action="store_true",
        default=True,
        help="Keep the best checkpoint (default)"
    )
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help="Keep downloaded models in models/ (default: remove)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("The Architect - Project Cleanup")
    print(f"{'='*60}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] No files will be removed\n")
    
    # Show current size
    root = get_project_root()
    outputs_dir = root / "outputs"
    if outputs_dir.exists():
        size_mb = get_directory_size(outputs_dir)
        print(f"\nCurrent outputs/ size: {size_mb:.2f} MB")
    
    # Clean up
    clean_models(dry_run=args.dry_run, keep_models=args.keep_models)
    clean_outputs(dry_run=args.dry_run)
    clean_logs(dry_run=args.dry_run)
    clean_pycache(dry_run=args.dry_run)
    clean_notebook_checkpoints(dry_run=args.dry_run)
    
    # Show new size
    if outputs_dir.exists() and not args.dry_run:
        new_size_mb = get_directory_size(outputs_dir)
        saved_mb = size_mb - new_size_mb
        print(f"\n{'='*60}")
        print(f"Cleanup complete!")
        print(f"Space saved: {saved_mb:.2f} MB")
        print(f"Remaining outputs/: {new_size_mb:.2f} MB")
        print(f"{'='*60}")
    elif args.dry_run:
        print(f"\n{'='*60}")
        print("Dry run complete. Run without --dry-run to actually remove files.")
        print(f"{'='*60}")
    
    print("\nPreserved:")
    print("  [OK] Source code (src/)")
    print("  [OK] Configurations (configs/)")
    print("  [OK] Scripts (scripts/)")
    print("  [OK] Documentation (docs/)")
    print("  [OK] Notebooks (notebooks/)")
    if args.keep_best:
        print("  [OK] Best checkpoint (*best.pt)")
    if args.keep_models:
        print("  [OK] Downloaded models (models/)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
