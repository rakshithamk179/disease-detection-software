"""Manifest generation and metadata management for raw tomato leaf images."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from PIL import Image

from src.utils import compute_sha256, validate_image_file


MANIFEST_COLUMNS = [
    "image_id",
    "relative_path",
    "filename",
    "class_label",
    "file_extension",
    "file_size_bytes",
    "image_width",
    "image_height",
    "image_mode",
    "is_valid_image",
    "sha256_hash",
    "perceptual_hash",
    "source_group",
    "source_type",
    "plant_id",
    "capture_session",
    "capture_date",
    "camera_or_source",
    "notes",
]


def compute_perceptual_hash(image_path: Path) -> str:
    """Compute perceptual hash (pHash) for image duplicate analysis."""
    try:
        import imagehash
    except ImportError as e:
        raise ImportError(
            "ImageHash is required for perceptual hashing. "
            "Please install it using: pip install ImageHash Pillow"
        ) from e

    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img.convert("RGB")))
    except Exception:
        return ""


def infer_initial_metadata(
    rel_path: Path,
    filename: str,
    class_name: str,
    sha256_str: str,
) -> Dict[str, str]:
    """Infer initial source group, source type, and metadata from path/filename."""
    parts = rel_path.parts
    # If there's an intermediate subfolder inside class_name:
    # e.g., raw/early_blight/plant_01/leaf_1.jpg -> parts = ('early_blight', 'plant_01', 'leaf_1.jpg')
    if len(parts) > 2:
        inferred_group = parts[1]
        source_type = "folder_group"
        return {
            "source_group": inferred_group,
            "source_type": source_type,
            "plant_id": inferred_group if "plant" in inferred_group.lower() else "",
            "capture_session": inferred_group if "session" in inferred_group.lower() else "",
            "capture_date": "",
            "camera_or_source": "",
            "notes": "Inferred from directory structure",
        }

    # If filename has prefix separated by '__' or similar delimiter indicating group/plant
    # e.g. plant4_001.jpg
    stem = Path(filename).stem
    if "_" in stem:
        prefix = stem.split("_")[0].strip()
        # If prefix resembles an ID (e.g. plant1, plotB, cameraA)
        if len(prefix) >= 3 and any(c.isdigit() for c in prefix) and len(prefix) <= 12:
            return {
                "source_group": prefix,
                "source_type": "filename_prefix",
                "plant_id": prefix if "plant" in prefix.lower() else "",
                "capture_session": "",
                "capture_date": "",
                "camera_or_source": "",
                "notes": f"Inferred from filename prefix '{prefix}'",
            }

    # If filename contains PlantVillage '___' delimiter
    if "___" in filename:
        suffix = stem.split("___")[-1].strip().lower()
        leaf_map_paths = [
            Path("PlantVillage-Dataset/leaf-map.json"),
            Path("../PlantVillage-Dataset/leaf-map.json"),
            Path("leaf-map.json"),
        ]
        leaf_id = None
        for lmp in leaf_map_paths:
            if lmp.is_file():
                try:
                    import json
                    with open(lmp, "r", encoding="utf-8") as f:
                        lmap = json.load(f)
                    mapped = lmap.get(suffix)
                    if mapped and len(mapped) > 0:
                        entry = mapped[0]
                        if ":::" in entry:
                            leaf_id = entry.split(":::")[-1].replace(".0", "")
                            break
                except Exception:
                    pass
        if leaf_id:
            return {
                "source_group": f"plant_leaf_{leaf_id}",
                "source_type": "verified_plant_leaf",
                "plant_id": f"plant_{leaf_id}",
                "capture_session": "",
                "capture_date": "",
                "camera_or_source": stem.split("___")[-1].split()[0] if " " in stem.split("___")[-1] else "",
                "notes": f"Ground-truth plant leaf ID from leaf mapping ({suffix})",
            }
        tokens = stem.split("___")[-1].strip().split()
        if tokens:
            grp = tokens[0].replace(".", "_")
            return {
                "source_group": grp,
                "source_type": "filename_token",
                "plant_id": "",
                "capture_session": grp,
                "capture_date": "",
                "camera_or_source": grp,
                "notes": f"Inferred from dataset delimiter pattern '{grp}'",
            }

    # Deterministic fallback group: We assign a deterministic fallback cluster
    # but explicitly mark it as fallback_group so it is NOT mistaken for genuine plant-level isolation.
    fallback_id = f"fallback_group_{sha256_str[:6]}" if sha256_str else f"fallback_group_{hash(filename) % 1000:03d}"
    return {
        "source_group": fallback_id,
        "source_type": "fallback_group",
        "plant_id": "",
        "capture_session": "",
        "capture_date": "",
        "camera_or_source": "",
        "notes": "Deterministic fallback group (unverified source independence)",
    }


def scan_raw_dataset(
    raw_data_dir: Path,
    manifest_path: Path,
    supported_extensions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Scan raw images, extract metadata and hashes, preserving existing manual annotations.

    - Class label comes strictly from raw folder name.
    - Unsupported files, corrupt images, and empty files are flagged.
    - Preserves user edits in existing manifest CSV.
    """
    raw_data_dir = Path(raw_data_dir).resolve()
    manifest_path = Path(manifest_path).resolve()

    if supported_extensions is None:
        supported_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    supported_ext_set = {ext.lower() for ext in supported_extensions}

    existing_metadata: Dict[str, Dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            old_df = pd.read_csv(manifest_path)
            for _, row in old_df.iterrows():
                key = str(row.get("relative_path", "")).replace("\\", "/")
                existing_metadata[key] = row.to_dict()
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []

    if not raw_data_dir.is_dir():
        # Return empty dataframe with specified columns
        df = pd.DataFrame(columns=MANIFEST_COLUMNS)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, index=False)
        return df

    # Find all class folders (direct subdirectories in raw_data_dir)
    class_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]

    idx = 1
    for c_dir in sorted(class_dirs, key=lambda x: x.name):
        class_label = c_dir.name
        # Find all files recursively in this class folder
        for file_path in sorted(c_dir.rglob("*")):
            if not file_path.is_file():
                continue

            rel_path = file_path.relative_to(raw_data_dir)
            rel_path_str = str(rel_path).replace("\\", "/")
            ext = file_path.suffix.lower()

            is_supported = ext in supported_ext_set
            file_size = file_path.stat().st_size

            val_result = validate_image_file(file_path) if is_supported and file_size > 0 else {
                "is_valid": False,
                "reason": "Unsupported file extension" if not is_supported else "Zero-byte file",
                "width": 0,
                "height": 0,
                "mode": "UNKNOWN",
                "file_size_bytes": file_size,
            }

            sha256_str = compute_sha256(file_path) if file_size > 0 else ""
            p_hash_str = compute_perceptual_hash(file_path) if val_result["is_valid"] else ""

            # Check if this file already has manual metadata in existing manifest
            existing = existing_metadata.get(rel_path_str)
            if existing:
                image_id = existing.get("image_id", f"IMG_{idx:06d}")
                source_group = existing.get("source_group") or ""
                source_type = existing.get("source_type") or "fallback_group"
                plant_id = existing.get("plant_id") or ""
                capture_session = existing.get("capture_session") or ""
                capture_date = existing.get("capture_date") or ""
                camera_or_source = existing.get("camera_or_source") or ""
                notes = existing.get("notes") or ""
            else:
                image_id = f"IMG_{idx:06d}"
                inferred = infer_initial_metadata(rel_path, file_path.name, class_label, sha256_str)
                source_group = inferred["source_group"]
                source_type = inferred["source_type"]
                plant_id = inferred["plant_id"]
                capture_session = inferred["capture_session"]
                capture_date = inferred["capture_date"]
                camera_or_source = inferred["camera_or_source"]
                notes = inferred["notes"]

            row_dict = {
                "image_id": image_id,
                "relative_path": rel_path_str,
                "filename": file_path.name,
                "class_label": class_label,
                "file_extension": ext,
                "file_size_bytes": file_size,
                "image_width": val_result["width"],
                "image_height": val_result["height"],
                "image_mode": val_result["mode"],
                "is_valid_image": bool(val_result["is_valid"]),
                "sha256_hash": sha256_str,
                "perceptual_hash": p_hash_str,
                "source_group": str(source_group),
                "source_type": str(source_type),
                "plant_id": str(plant_id) if pd.notna(plant_id) else "",
                "capture_session": str(capture_session) if pd.notna(capture_session) else "",
                "capture_date": str(capture_date) if pd.notna(capture_date) else "",
                "camera_or_source": str(camera_or_source) if pd.notna(camera_or_source) else "",
                "notes": str(notes) if pd.notna(notes) else "",
            }
            rows.append(row_dict)
            idx += 1

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df
