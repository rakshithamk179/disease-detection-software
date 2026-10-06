"""Utility functions for logging, hashing, image validation, and file I/O."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError


def setup_logger(
    name: str = "tomato_disease_classifier",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_image_file(file_path: Path, min_dimension: int = 16) -> Dict[str, Any]:
    """Validate image file integrity, readability, dimensions, and color mode.

    Returns a dictionary with validation status, dimensions, mode, and reason if invalid.
    """
    file_path = Path(file_path)
    result: Dict[str, Any] = {
        "is_valid": False,
        "reason": None,
        "width": 0,
        "height": 0,
        "mode": "UNKNOWN",
        "file_size_bytes": 0,
    }

    if not file_path.is_file():
        result["reason"] = "File does not exist"
        return result

    try:
        size_bytes = file_path.stat().st_size
        result["file_size_bytes"] = size_bytes
        if size_bytes == 0:
            result["reason"] = "Zero-byte empty file"
            return result

        with Image.open(file_path) as img:
            img.verify()

        # Reopen to read dimensions and mode after verify()
        with Image.open(file_path) as img:
            w, h = img.size
            mode = img.mode
            result["width"] = w
            result["height"] = h
            result["mode"] = mode

            if w < min_dimension or h < min_dimension:
                result["reason"] = f"Image dimensions ({w}x{h}) smaller than minimum ({min_dimension}x{min_dimension})"
                return result

            # Verify that converting to RGB works
            _ = img.convert("RGB")

        result["is_valid"] = True
        return result
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as err:
        result["reason"] = f"Corrupted or invalid image: {str(err)}"
        return result


class NumpyJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder to serialize NumPy numbers, arrays, and Path objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (Path, set)):
            return str(obj) if isinstance(obj, Path) else list(obj)
        return super().default(obj)


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """Save data to JSON with custom serialization for NumPy types and Path objects."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyJsonEncoder)


def load_json(file_path: Path) -> Any:
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_markdown(content: str, file_path: Path) -> None:
    """Write markdown text to a file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
