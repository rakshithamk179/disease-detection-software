#!/usr/bin/env python3
"""Script to execute safe, group-aware dataset splitting with cryptographic test lock."""

import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.split_dataset import perform_split
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and locked test sets with group isolation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the split and generate reports without copying files.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Overwrite existing processed data directories if they already contain files.",
    )
    parser.add_argument(
        "--allow-fallback-groups",
        action="store_true",
        help="Acknowledge that source-group metadata is unverified/fallback and proceed with preliminary internal splitting.",
    )
    parser.add_argument(
        "--allow-small-test-set",
        action="store_true",
        help="Permit fewer than 20 images per test class with explicit preliminary caveat.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for deterministic cluster allocation (default: config seed).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["resolved_paths"]
    logger = setup_logger(log_dir=paths["logs_dir"])

    manifest_path = paths["manifest_path"]
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest file not found at: {manifest_path}")
        print("Please run `python scripts/create_manifest.py` before splitting.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    split_cfg = config.get("split", {})
    seed = args.seed if args.seed is not None else config.get("seed", 42)

    print("=" * 80)
    print("DATASET SPLITTING & TEST-SET CRYPTOGRAPHIC LOCKING")
    print(f"Manifest:          {manifest_path}")
    print(f"Target ratios:     Train={split_cfg.get('train_ratio', 0.70)}, Val={split_cfg.get('val_ratio', 0.15)}, Test={split_cfg.get('test_ratio', 0.15)}")
    print(f"Random seed:       {seed}")
    print(f"Dry-run mode:      {args.dry_run}")
    print(f"Force rebuild:     {args.force_rebuild}")
    print(f"Allow fallback:    {args.allow_fallback_groups}")
    print("=" * 80)

    try:
        summary = perform_split(
            manifest_df=df,
            raw_data_dir=paths["raw_data_dir"],
            processed_dir=paths["train_dir"].parent,
            reports_dir=paths["reports_dir"],
            metadata_dir=paths["test_lock_manifest"].parent,
            train_ratio=split_cfg.get("train_ratio", 0.70),
            val_ratio=split_cfg.get("val_ratio", 0.15),
            test_ratio=split_cfg.get("test_ratio", 0.15),
            min_test_images_per_class=split_cfg.get("minimum_test_images_per_class", 20),
            allow_fallback_groups=args.allow_fallback_groups,
            allow_small_test_set=args.allow_small_test_set,
            dry_run=args.dry_run,
            force_rebuild=args.force_rebuild,
            seed=seed,
        )

        print("\n[SUCCESS] Dataset split completed successfully!")
        print(f"Total valid images: {summary['total_valid_images']}")
        print("\nClass counts by split:")
        for s, counts in summary["class_count_by_split"].items():
            print(f"  - {s}: {counts}")

        if not args.dry_run:
            print(f"\nFiles copied to data/processed/")
            print(f"Test lock manifest generated: {paths['test_lock_manifest']}")
        else:
            print("\n[NOTE] Dry-run completed. No files were written to data/processed/.")

        print(f"Reports saved to {paths['reports_dir']}/")
    except Exception as e:
        print(f"\n[SPLIT HALTED]: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
