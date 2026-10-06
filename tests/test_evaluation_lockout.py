"""Tests for evaluation pipeline — test split lockout, eval reports, and reporting integrity."""

import json
from pathlib import Path
from PIL import Image
import pandas as pd
import pytest
import tensorflow as tf
import numpy as np

from src.utils import save_json


def create_dummy_image(path: Path, color=(100, 150, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path)


def test_evaluate_without_final_flag_denies_test_access():
    """Attempting to load test_locked without --final-evaluation must raise PermissionError."""
    from src.dataset_loader import load_locked_test_dataset
    with pytest.raises(PermissionError, match="Access to locked test dataset is DENIED"):
        load_locked_test_dataset(
            locked_test_dir=Path("data/processed/test_locked"),
            final_evaluation=False,
        )


def test_evaluate_script_argument_requires_final_flag(tmp_path: Path):
    """evaluate_model.py must refuse test_locked evaluation without --final-evaluation flag."""
    from src.evaluate import evaluate_model_pipeline
    from src.utils import save_json

    # Build mock config
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_json(["early_blight", "healthy", "late_blight"], models_dir / "class_names.json")

    config = {
        "image": {"width": 224, "height": 224},
        "training": {"batch_size": 4},
        "model": {"low_confidence_threshold": 0.5},
        "resolved_paths": {
            "models_dir": models_dir,
            "logs_dir": tmp_path / "logs",
            "reports_dir": tmp_path / "reports",
            "train_dir": tmp_path / "train",
            "val_dir": tmp_path / "val",
            "locked_test_dir": tmp_path / "test_locked",
            "test_lock_manifest": tmp_path / "test_lock_manifest.json",
            "manifest_path": tmp_path / "manifest.csv",
        },
    }

    with pytest.raises(PermissionError, match="ACCESS DENIED"):
        evaluate_model_pipeline(
            config=config,
            split="test_locked",
            final_evaluation=False,
        )
