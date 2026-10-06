"""TensorFlow tf.data dataset loading pipelines with locked-test enforcement."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf

from src.utils import load_json, save_json


def get_data_augmentation() -> tf.keras.Sequential:
    """Create realistic image data augmentation pipeline for training only."""
    layers = [
        tf.keras.layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
        tf.keras.layers.RandomRotation(0.08, name="aug_rotation"),
        tf.keras.layers.RandomZoom(0.10, name="aug_zoom"),
        tf.keras.layers.RandomContrast(0.10, name="aug_contrast"),
    ]
    # Check if RandomBrightness is available in the installed keras/tf version
    if hasattr(tf.keras.layers, "RandomBrightness"):
        layers.append(tf.keras.layers.RandomBrightness(0.08, name="aug_brightness"))

    return tf.keras.Sequential(layers, name="data_augmentation")


def load_training_dataset(
    train_dir: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    seed: int = 42,
    cache: bool = False,
    class_names_path: Optional[Path] = None,
) -> Tuple[tf.data.Dataset, List[str]]:
    """Load training dataset from data/processed/train with augmentation and shuffling.

    Saves the true class ordering discovered by TensorFlow to models/class_names.json.
    """
    train_dir = Path(train_dir).resolve()
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=None,  # Discovers and sorts subdirectories
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )

    class_names = list(raw_ds.class_names)
    if class_names_path:
        class_names_path = Path(class_names_path)
        class_names_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(class_names, class_names_path)

    aug = get_data_augmentation()
    # Apply data augmentation only to training data
    train_ds = raw_ds.map(
        lambda x, y: (aug(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        train_ds = train_ds.cache()

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, class_names


def load_validation_dataset(
    val_dir: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    class_names: Optional[List[str]] = None,
) -> tf.data.Dataset:
    """Load validation dataset from data/processed/val with no augmentation and no shuffling."""
    val_dir = Path(val_dir).resolve()
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found at: {val_dir}")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,  # Ensures identical index mapping to training
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return val_ds


def load_locked_test_dataset(
    locked_test_dir: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    class_names: Optional[List[str]] = None,
    final_evaluation: bool = False,
) -> tf.data.Dataset:
    """Load locked test dataset strictly for final frozen evaluation.

    Raises PermissionError if final_evaluation is False.
    """
    if not final_evaluation:
        raise PermissionError(
            "Access to locked test dataset is DENIED. "
            "The locked test set is strictly reserved for final frozen evaluation. "
            "You must supply the explicit '--final-evaluation' flag."
        )

    locked_test_dir = Path(locked_test_dir).resolve()
    if not locked_test_dir.is_dir():
        raise FileNotFoundError(f"Locked test directory not found at: {locked_test_dir}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        locked_test_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return test_ds
