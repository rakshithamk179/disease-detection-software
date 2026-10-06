#!/usr/bin/env python3
"""Script to scan raw tomato leaf images and generate or update the metadata manifest."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.manifest import scan_raw_dataset
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan data/raw and generate/update data/metadata/image_manifest.csv"
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

    raw_dir = paths["raw_data_dir"]
    manifest_path = paths["manifest_path"]
    supported_exts = config.get("image", {}).get(
        "supported_extensions", [".jpg", ".jpeg", ".png", ".webp"]
    )

    print("=" * 80)
    print("CREATING IMAGE MANIFEST")
    print(f"Scanning directory: {raw_dir}")
    print(f"Target manifest:   {manifest_path}")
    print("=" * 80)

    df = scan_raw_dataset(
        raw_data_dir=raw_dir,
        manifest_path=manifest_path,
        supported_extensions=supported_exts,
    )

    print(f"\nManifest successfully created/updated: {manifest_path}")
    print(f"Total images cataloged: {len(df)}")
    if not df.empty:
        valid_count = int(df["is_valid_image"].sum())
        invalid_count = len(df) - valid_count
        print(f"Valid images:           {valid_count}")
        print(f"Invalid/corrupt files:  {invalid_count}")
        print("\nClass distribution:")
        for c, count in df["class_label"].value_counts().items():
            print(f"  - {c}: {count}")

        fallback_groups = (df["source_type"] == "fallback_group").sum()
        if fallback_groups > 0:
            print(f"\n[NOTE] {fallback_groups} images have unverified fallback source groups.")
            print("To ensure scientifically valid splitting, edit source_group/plant_id columns in CSV if real metadata exists.")
    else:
        print("[WARNING] No images found. Please place images in data/raw/early_blight, data/raw/healthy, data/raw/late_blight.")


if __name__ == "__main__":
    main()
