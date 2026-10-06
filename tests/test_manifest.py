"""Tests for raw image scanning, metadata extraction, and manifest integrity."""

from pathlib import Path
import pandas as pd
from PIL import Image
import pytest

from src.manifest import scan_raw_dataset
from src.utils import validate_image_file


def create_dummy_image(path: Path, size=(64, 64), color=(0, 200, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def test_manifest_detects_label_from_folder(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    eb_dir = raw_dir / "early_blight"
    hl_dir = raw_dir / "healthy"
    lb_dir = raw_dir / "late_blight"

    create_dummy_image(eb_dir / "leaf1.jpg", color=(100, 50, 0))
    create_dummy_image(hl_dir / "leaf2.png", color=(0, 255, 0))
    create_dummy_image(lb_dir / "leaf3.jpeg", color=(50, 50, 50))

    manifest_csv = tmp_path / "manifest.csv"
    df = scan_raw_dataset(raw_dir, manifest_csv)

    assert len(df) == 3
    assert set(df["class_label"].unique()) == {"early_blight", "healthy", "late_blight"}

    eb_row = df[df["filename"] == "leaf1.jpg"].iloc[0]
    assert eb_row["class_label"] == "early_blight"
    assert eb_row["is_valid_image"] is True or eb_row["is_valid_image"] == 1


def test_corrupt_and_empty_images_flagged(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "healthy"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Valid image
    valid_file = raw_dir / "good.jpg"
    create_dummy_image(valid_file)

    # Empty 0-byte file
    empty_file = raw_dir / "empty.jpg"
    empty_file.write_bytes(b"")

    # Corrupt fake image file
    corrupt_file = raw_dir / "corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_A_VALID_IMAGE_FILE_HEADER_XYZ")

    # Unsupported extension
    txt_file = raw_dir / "notes.txt"
    txt_file.write_text("just some text", encoding="utf-8")

    manifest_csv = tmp_path / "manifest.csv"
    df = scan_raw_dataset(tmp_path / "raw", manifest_csv)

    good_row = df[df["filename"] == "good.jpg"].iloc[0]
    assert bool(good_row["is_valid_image"]) is True

    empty_row = df[df["filename"] == "empty.jpg"].iloc[0]
    assert bool(empty_row["is_valid_image"]) is False

    corrupt_row = df[df["filename"] == "corrupt.jpg"].iloc[0]
    assert bool(corrupt_row["is_valid_image"]) is False

    txt_row = df[df["filename"] == "notes.txt"].iloc[0]
    assert bool(txt_row["is_valid_image"]) is False


def test_manifest_preserves_manual_annotations(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "early_blight"
    img_file = raw_dir / "sample.jpg"
    create_dummy_image(img_file)

    manifest_csv = tmp_path / "manifest.csv"
    df1 = scan_raw_dataset(tmp_path / "raw", manifest_csv)
    assert len(df1) == 1

    # Simulate user manual edit
    df1.loc[0, "source_group"] = "greenhouse_plot_4"
    df1.loc[0, "plant_id"] = "PLANT_99"
    df1.loc[0, "notes"] = "Verified field sample"
    df1.to_csv(manifest_csv, index=False)

    # Rescan dataset
    df2 = scan_raw_dataset(tmp_path / "raw", manifest_csv)
    row = df2.iloc[0]
    assert row["source_group"] == "greenhouse_plot_4"
    assert row["plant_id"] == "PLANT_99"
    assert row["notes"] == "Verified field sample"
