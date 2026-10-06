"""Tests for cryptographic test-set lock verification, tamper detection, and isolation."""

import hashlib
import json
from pathlib import Path
from PIL import Image
import pytest

from src.dataset_loader import load_locked_test_dataset
from src.split_dataset import verify_locked_test_set
from src.utils import compute_sha256, save_json


def create_dummy_image(path: Path, color=(30, 150, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


def setup_mock_locked_environment(tmp_path: Path):
    processed = tmp_path / "processed"
    train_dir = processed / "train" / "early_blight"
    val_dir = processed / "val" / "early_blight"
    test_dir = processed / "test_locked" / "early_blight"

    create_dummy_image(train_dir / "tr1.jpg", color=(10, 10, 10))
    create_dummy_image(val_dir / "val1.jpg", color=(20, 20, 20))
    test_file = test_dir / "ts1.jpg"
    create_dummy_image(test_file, color=(30, 30, 30))

    test_sha = compute_sha256(test_file)
    test_records = [
        {
            "relative_path": "early_blight/ts1.jpg",
            "sha256": test_sha,
            "class_label": "early_blight",
            "source_group": "group_test_1",
        }
    ]
    canonical = json.dumps(test_records, sort_keys=True)
    manifest_checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    manifest_data = {
        "split_timestamp": "2026-05-01T12:00:00Z",
        "random_seed": 42,
        "test_image_count": 1,
        "class_counts": {"early_blight": 1},
        "source_groups": ["group_test_1"],
        "test_images": test_records,
        "manifest_checksum": manifest_checksum,
    }
    manifest_path = tmp_path / "test_lock_manifest.json"
    save_json(manifest_data, manifest_path)

    return manifest_path, processed, tmp_path / "reports", test_file


def test_lock_verification_passes_when_intact(tmp_path: Path):
    manifest_path, processed, reports, test_file = setup_mock_locked_environment(tmp_path)
    is_valid, report = verify_locked_test_set(
        test_lock_manifest_path=manifest_path,
        processed_dir=processed,
        reports_dir=reports,
    )
    assert is_valid is True
    assert report["status"] == "PASS"


def test_lock_verification_catches_tampered_image(tmp_path: Path):
    manifest_path, processed, reports, test_file = setup_mock_locked_environment(tmp_path)

    # Tamper with test image (modify pixels)
    create_dummy_image(test_file, color=(255, 0, 0))

    is_valid, report = verify_locked_test_set(
        test_lock_manifest_path=manifest_path,
        processed_dir=processed,
        reports_dir=reports,
    )
    assert is_valid is False
    assert report["status"] == "FAIL"
    assert len(report["altered_test_files"]) == 1
    assert "early_blight/ts1.jpg" in report["altered_test_files"]


def test_test_dataset_loader_denies_access_without_flag(tmp_path: Path):
    manifest_path, processed, reports, test_file = setup_mock_locked_environment(tmp_path)
    test_dir = processed / "test_locked"

    # Must raise PermissionError when final_evaluation is False
    with pytest.raises(PermissionError, match="Access to locked test dataset is DENIED"):
        load_locked_test_dataset(
            locked_test_dir=test_dir,
            final_evaluation=False,
        )


def test_training_code_never_loads_locked_test():
    # Source code analysis check: ensure train.py does not call load_locked_test_dataset
    import inspect
    from src import train
    source = inspect.getsource(train)

    assert "load_locked_test_dataset" not in source
    assert "locked_test_dir" not in source or "The locked test set will not be loaded" in source
