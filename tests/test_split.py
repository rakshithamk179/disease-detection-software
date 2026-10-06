"""Tests for group-aware dataset splitting, stratification, and leakage prevention."""

from pathlib import Path
import pandas as pd
from PIL import Image
import pytest

from src.split_dataset import build_connected_components, perform_split, verify_split_integrity


def create_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(50, 100, 50))
    img.save(path)


def make_sample_manifest(tmp_path: Path, n_per_class: int = 30) -> pd.DataFrame:
    raw_dir = tmp_path / "raw"
    records = []
    classes = ["early_blight", "healthy", "late_blight"]

    img_id = 1
    for c in classes:
        # Create 10 source groups per class (each group has 3 images)
        for g_idx in range(n_per_class // 3):
            group_name = f"plot_{c}_{g_idx:02d}"
            for i in range(3):
                fn = f"{group_name}_leaf_{i}.jpg"
                rel = f"{c}/{fn}"
                file_p = raw_dir / rel
                create_dummy_image(file_p)

                records.append({
                    "image_id": f"IMG_{img_id:05d}",
                    "relative_path": rel,
                    "filename": fn,
                    "class_label": c,
                    "file_extension": ".jpg",
                    "file_size_bytes": file_p.stat().st_size,
                    "image_width": 32,
                    "image_height": 32,
                    "image_mode": "RGB",
                    "is_valid_image": True,
                    "sha256_hash": f"hash_{c}_{g_idx}_{i}",
                    "perceptual_hash": "ffff0000ffff0000",
                    "source_group": group_name,
                    "source_type": "verified_plot",
                    "plant_id": f"plant_{g_idx}",
                    "capture_session": "",
                    "capture_date": "2026-05-01",
                    "camera_or_source": "field_rig_1",
                    "notes": "",
                })
                img_id += 1

    return pd.DataFrame(records)


def test_source_group_and_exact_hash_leakage_prevented(tmp_path: Path):
    manifest_df = make_sample_manifest(tmp_path, n_per_class=30)
    reports_dir = tmp_path / "reports"
    meta_dir = tmp_path / "meta"
    processed_dir = tmp_path / "processed"

    summary = perform_split(
        manifest_df=manifest_df,
        raw_data_dir=tmp_path / "raw",
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        metadata_dir=meta_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_test_images_per_class=3,
        allow_small_test_set=True,
        seed=42,
    )

    split_manifest = pd.read_csv(reports_dir / "split_manifest.csv")

    # 1. Verify zero source_group overlap across splits
    groups_train = set(split_manifest[split_manifest["split"] == "train"]["source_group"])
    groups_val = set(split_manifest[split_manifest["split"] == "val"]["source_group"])
    groups_test = set(split_manifest[split_manifest["split"] == "test_locked"]["source_group"])

    assert len(groups_train.intersection(groups_val)) == 0
    assert len(groups_train.intersection(groups_test)) == 0
    assert len(groups_val.intersection(groups_test)) == 0

    # 2. Verify zero SHA-256 overlap across splits
    hashes_train = set(split_manifest[split_manifest["split"] == "train"]["sha256_hash"])
    hashes_val = set(split_manifest[split_manifest["split"] == "val"]["sha256_hash"])
    hashes_test = set(split_manifest[split_manifest["split"] == "test_locked"]["sha256_hash"])

    assert len(hashes_train.intersection(hashes_val)) == 0
    assert len(hashes_train.intersection(hashes_test)) == 0
    assert len(hashes_val.intersection(hashes_test)) == 0


def test_split_fails_if_class_missing():
    # Construct dataframe where split is missing a class
    data = [
        {"image_id": "1", "sha256_hash": "h1", "source_group": "g1", "class_label": "early_blight", "split": "train"},
        {"image_id": "2", "sha256_hash": "h2", "source_group": "g2", "class_label": "healthy", "split": "train"},
        # late_blight is missing in train!
        {"image_id": "3", "sha256_hash": "h3", "source_group": "g3", "class_label": "early_blight", "split": "val"},
        {"image_id": "4", "sha256_hash": "h4", "source_group": "g4", "class_label": "healthy", "split": "val"},
        {"image_id": "5", "sha256_hash": "h5", "source_group": "g5", "class_label": "late_blight", "split": "val"},
        {"image_id": "6", "sha256_hash": "h6", "source_group": "g6", "class_label": "early_blight", "split": "test_locked"},
        {"image_id": "7", "sha256_hash": "h7", "source_group": "g7", "class_label": "healthy", "split": "test_locked"},
        {"image_id": "8", "sha256_hash": "h8", "source_group": "g8", "class_label": "late_blight", "split": "test_locked"},
    ]
    df = pd.DataFrame(data)
    is_valid, errors = verify_split_integrity(df, classes=["early_blight", "healthy", "late_blight"], min_test_per_class=1)
    assert is_valid is False
    assert any("missing from split 'train'" in e.lower() for e in errors)


def test_split_rejects_fallback_groups_without_flag(tmp_path: Path):
    manifest_df = make_sample_manifest(tmp_path, n_per_class=30)
    # Mark groups as fallback
    manifest_df["source_type"] = "fallback_group"

    with pytest.raises(ValueError, match="fallback groups"):
        perform_split(
            manifest_df=manifest_df,
            raw_data_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
            metadata_dir=tmp_path / "meta",
            allow_fallback_groups=False,  # Strict rejection
            allow_small_test_set=True,
        )
