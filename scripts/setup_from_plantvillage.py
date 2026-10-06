#!/usr/bin/env python3
"""Standalone script to set up data/raw directory from PlantVillage dataset.

Use this to populate data/raw/ from the existing PlantVillage-Dataset directory.
"""
import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy tomato leaf images from PlantVillage-Dataset into data/raw structure."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to PlantVillage-Dataset/raw/color directory. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="Max images to copy per class (default: all available).",
    )
    args = parser.parse_args()

    # Auto-detect PlantVillage location relative to project
    if args.source:
        color_dir = Path(args.source)
    else:
        candidates = [
            PROJECT_ROOT.parent / "PlantVillage-Dataset" / "raw" / "color",
            PROJECT_ROOT / "PlantVillage-Dataset" / "raw" / "color",
        ]
        color_dir = next((c for c in candidates if c.is_dir()), None)
        if color_dir is None:
            print("[ERROR] Could not auto-locate PlantVillage-Dataset/raw/color/")
            print("Please specify the path using --source /path/to/PlantVillage-Dataset/raw/color")
            sys.exit(1)

    data_raw_dir = PROJECT_ROOT / "data" / "raw"
    mapping = {
        "Tomato___Early_blight": data_raw_dir / "early_blight",
        "Tomato___healthy": data_raw_dir / "healthy",
        "Tomato___Late_blight": data_raw_dir / "late_blight",
    }

    print("=" * 80)
    print("POPULATING data/raw FROM PLANTVILLAGE DATASET")
    print(f"Source: {color_dir}")
    print("=" * 80)

    for src_folder, dst_folder in mapping.items():
        src_path = color_dir / src_folder
        if not src_path.is_dir():
            print(f"[WARNING] Source folder not found, skipping: {src_path}")
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)
        images = [
            f for f in sorted(src_path.iterdir())
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
        selected = images[:args.n_per_class] if args.n_per_class else images
        n_copied = 0
        for img in selected:
            dest_file = dst_folder / img.name
            if not dest_file.is_file():
                shutil.copy2(img, dest_file)
                n_copied += 1

        total = len(list(dst_folder.iterdir()))
        print(f"  {dst_folder.name}: {n_copied} new files copied ({total} total)")

    print("\n[DONE] Now run: python scripts/create_manifest.py")


if __name__ == "__main__":
    main()
