#!/usr/bin/env python3
"""Script to execute read-only audit of raw dataset and metadata manifest."""

import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.dataset_audit import audit_raw_dataset
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw tomato leaf dataset for duplicates, corruptions, and balance."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["resolved_paths"]
    logger = setup_logger(log_dir=paths["logs_dir"])

    manifest_path = paths["manifest_path"]
    reports_dir = paths["reports_dir"]

    if not manifest_path.is_file():
        print(f"[ERROR] Manifest not found at: {manifest_path}")
        print("Please generate the manifest first using:")
        print("    python scripts/create_manifest.py")
        sys.exit(1)

    print("=" * 80)
    print("AUDITING RAW DATASET & MANIFEST (READ-ONLY)")
    print(f"Reading manifest from: {manifest_path}")
    print(f"Saving reports to:    {reports_dir}")
    print("=" * 80)

    df = pd.read_csv(manifest_path)
    split_cfg = config.get("split", {})
    hamming_thresh = split_cfg.get("duplicate_hamming_threshold", 6)

    audit_result = audit_raw_dataset(
        manifest_df=df,
        reports_dir=reports_dir,
        hamming_threshold=hamming_thresh,
    )

    summary = audit_result["dataset_summary"]
    exact = audit_result["exact_duplicates"]
    near = audit_result["near_duplicates"]
    conflicts = audit_result["label_conflicts"]
    grp = audit_result["source_group_analysis"]

    print("\n--- AUDIT SUMMARY RESULTS ---")
    print(f"Total Images:          {summary['total_images']}")
    print(f"Valid Images:          {summary['valid_images']}")
    print(f"Invalid Images:        {summary['invalid_images']}")
    print(f"Class Balance Ratio:   {summary['class_imbalance_ratio']} ({'Imbalanced' if summary['is_imbalanced'] else 'Balanced'})")
    print(f"Exact Duplicate Pairs: {exact['total_exact_duplicate_pairs']} (Cross-class: {exact['cross_class_pairs']})")
    print(f"Near Duplicate Pairs:  {near['total_near_duplicate_pairs']} (Cross-class: {near['cross_class_near_duplicates']})")
    print(f"Total Label Conflicts: {conflicts['total_conflicts']}")
    print(f"Unique Source Groups:  {grp['total_unique_groups']} (Fallback: {grp['fallback_groups_count']}, Verified: {grp['verified_real_groups_count']})")

    if conflicts["total_conflicts"] > 0:
        print("\n[WARNING] Suspected label conflicts detected! Check reports/label_conflict_candidates.csv before splitting.")

    if not grp["is_sufficient_for_genuine_group_split"]:
        print("\n[NOTE] Source-group metadata is unverified or consists of fallback groups.")
        print("Splitting will require --allow-fallback-groups flag.")

    print(f"\nDetailed reports saved in: {reports_dir}/")
    print("  - reports/raw_dataset_audit.json")
    print("  - reports/raw_dataset_audit.md")
    print("  - reports/exact_duplicates.csv")
    print("  - reports/near_duplicates.csv")
    print("  - reports/label_conflict_candidates.csv")


if __name__ == "__main__":
    main()
