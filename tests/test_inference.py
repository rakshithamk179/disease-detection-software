"""Tests for model inference, output schema, and TFLite export isolation."""

import inspect
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from src import export_tflite
from src.dataset_loader import load_training_dataset
from src.inference import MANDATORY_SEVERITY_DISCLAIMER, predict_single_image
from src.utils import load_json, save_json


def create_dummy_image(path: Path, color=(0, 200, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path)


def test_tensorflow_loader_saves_class_order(tmp_path: Path):
    train_dir = tmp_path / "train"
    c1 = train_dir / "late_blight"
    c2 = train_dir / "early_blight"
    c3 = train_dir / "healthy"

    create_dummy_image(c1 / "img1.jpg")
    create_dummy_image(c2 / "img2.jpg")
    create_dummy_image(c3 / "img3.jpg")

    class_names_path = tmp_path / "class_names.json"
    ds, names = load_training_dataset(
        train_dir=train_dir,
        image_size=(32, 32),
        batch_size=2,
        class_names_path=class_names_path,
    )

    saved_names = load_json(class_names_path)
    assert saved_names == names
    # Alphabetical order discovered by TensorFlow directory loader
    assert saved_names == ["early_blight", "healthy", "late_blight"]


def test_inference_json_serializable_and_no_severity_claim(tmp_path: Path):
    # Create lightweight mock model for test
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    mock_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_model.keras"
    mock_model.save(model_path)

    class_names = ["early_blight", "healthy", "late_blight"]
    save_json(class_names, models_dir / "class_names.json")

    test_img = tmp_path / "query_leaf.jpg"
    create_dummy_image(test_img)

    config = {
        "image": {"width": 224, "height": 224},
        "model": {"low_confidence_threshold": 0.50},
        "resolved_paths": {
            "models_dir": models_dir,
            "logs_dir": tmp_path / "logs",
            "reports_dir": tmp_path / "reports",
        },
    }

    result = predict_single_image(
        image_path=test_img,
        config=config,
        model=mock_model,
        class_names=class_names,
    )

    # 1. Output must be strictly JSON-serializable
    serialized = json.dumps(result)
    assert isinstance(serialized, str)

    # 2. Confidence must NOT claim to represent disease severity
    assert "scientific_integrity_note" in result
    assert result["scientific_integrity_note"] == MANDATORY_SEVERITY_DISCLAIMER
    assert "severity" not in result["predicted_class"].lower()


def test_tflite_export_never_references_locked_test():
    # Source code analysis check: export_tflite module must never reference test_locked or test dataset loader
    source = inspect.getsource(export_tflite)

    assert "load_locked_test_dataset" not in source
    assert "locked_test_dir" not in source
    assert "locked test images were NOT used" in source or "VALIDATION images only" in source
