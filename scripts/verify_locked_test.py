#!/usr/bin/env python3
"""Script to cryptographically verify locked test-set integrity and absence of data leakage."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.split_dataset import verify_locked_test_set
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify cryptographic test-set lock and absence of hash/group leakage."
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

    test_lock_manifest_path = paths["test_lock_manifest"]
    processed_dir = paths["train_dir"].parent
    reports_dir = paths["reports_dir"]

    print("=" * 80)
    print("VERIFYING LOCKED TEST-SET INTEGRITY")
    print(f"Manifest path: {test_lock_manifest_path}")
    print(f"Processed dir: {processed_dir}")
    print("=" * 80)

    is_valid, report = verify_locked_test_set(
        test_lock_manifest_path=test_lock_manifest_path,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )

    print("\n--- TEST LOCK VERIFICATION RESULTS ---")
    print(f"Status:                      {report['status']}")
    print(f"Total Test Images Tracked:   {report['total_test_images']}")
    print(f"Manifest Checksum Valid:     {report['manifest_checksum_valid']}")
    print(f"Missing Files:               {len(report['missing_test_files'])}")
    print(f"Altered Files:               {len(report['altered_test_files'])}")
    print(f"Train/Val Hash Overlaps:     {len(report['train_val_hash_overlaps'])}")
    print(f"Source Group Overlaps:       {len(report['source_group_overlaps'])}")

    if not is_valid:
        print("\n[VERIFICATION FAILED]:")
        for err in report["errors"]:
            print(f"  ❌ {err}")
        print(f"\nDetailed report saved to: {reports_dir}/test_lock_verification.json")
        sys.exit(1)
    else:
        print("\n[VERIFICATION PASSED]: All locked test files are intact with zero leakages.")
        print(f"Reports saved to {reports_dir}/test_lock_verification.json and .md")


if __name__ == "__main__":
    main()
