"""Duplicate detection module for exact (SHA-256) and near-duplicate (pHash) analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def compute_hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex-encoded perceptual hashes."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 9999
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        return (val1 ^ val2).bit_count()
    except (ValueError, TypeError):
        return 9999


def find_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Identify exact duplicates by SHA-256 hash.

    Analyzes duplicate distribution within classes, across classes (critical label conflict),
    and across source groups.
    """
    valid_df = df[df["is_valid_image"] & (df["sha256_hash"].str.len() > 0)].copy()

    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "total_exact_duplicate_pairs": 0,
        "within_class_pairs": 0,
        "cross_class_pairs": 0,
        "cross_group_pairs": 0,
    }

    # Group by SHA256
    hash_groups = valid_df.groupby("sha256_hash")
    for sha, group in hash_groups:
        if len(group) > 1:
            group_rows = group.to_dict("records")
            for i in range(len(group_rows)):
                for j in range(i + 1, len(group_rows)):
                    r1 = group_rows[i]
                    r2 = group_rows[j]
                    is_cross_class = r1["class_label"] != r2["class_label"]
                    is_cross_group = str(r1["source_group"]) != str(r2["source_group"])

                    conflict_type = "same_class"
                    if is_cross_class:
                        conflict_type = "cross_class_conflict"
                        summary["cross_class_pairs"] += 1
                    else:
                        summary["within_class_pairs"] += 1

                    if is_cross_group:
                        summary["cross_group_pairs"] += 1

                    summary["total_exact_duplicate_pairs"] += 1

                    records.append({
                        "sha256_hash": sha,
                        "image_id_1": r1["image_id"],
                        "relative_path_1": r1["relative_path"],
                        "class_1": r1["class_label"],
                        "source_group_1": r1["source_group"],
                        "image_id_2": r2["image_id"],
                        "relative_path_2": r2["relative_path"],
                        "class_2": r2["class_label"],
                        "source_group_2": r2["source_group"],
                        "conflict_type": conflict_type,
                    })

    cols = [
        "sha256_hash",
        "image_id_1",
        "relative_path_1",
        "class_1",
        "source_group_1",
        "image_id_2",
        "relative_path_2",
        "class_2",
        "source_group_2",
        "conflict_type",
    ]
    return pd.DataFrame(records, columns=cols), summary


def find_near_duplicates(
    df: pd.DataFrame,
    hamming_threshold: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Identify near-duplicate image pairs using perceptual hash Hamming distance.

    Near-duplicate pairs are flagged as 'manual review required'.
    """
    valid_df = df[df["is_valid_image"] & (df["perceptual_hash"].str.len() > 0)].copy()
    rows = valid_df.to_dict("records")
    n = len(rows)

    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "hamming_threshold": hamming_threshold,
        "total_near_duplicate_pairs": 0,
        "within_class_near_duplicates": 0,
        "cross_class_near_duplicates": 0,
    }

    # Pre-parse integers for pHash for high performance
    parsed_hashes = []
    for r in rows:
        ph = str(r["perceptual_hash"])
        try:
            val = int(ph, 16)
        except Exception:
            val = None
        parsed_hashes.append(val)

    for i in range(n):
        v1 = parsed_hashes[i]
        if v1 is None:
            continue
        r1 = rows[i]
        for j in range(i + 1, n):
            v2 = parsed_hashes[j]
            if v2 is None:
                continue

            dist = (v1 ^ v2).bit_count()
            # If distance is 0, that's either exact match or identical perceptual appearance
            if dist <= hamming_threshold:
                r2 = rows[j]
                # If exact sha256 is the same, it is already an exact duplicate
                if r1["sha256_hash"] == r2["sha256_hash"] and r1["sha256_hash"]:
                    continue

                is_cross_class = bool(r1["class_label"] != r2["class_label"])
                if is_cross_class:
                    summary["cross_class_near_duplicates"] += 1
                else:
                    summary["within_class_near_duplicates"] += 1
                summary["total_near_duplicate_pairs"] += 1

                records.append({
                    "image_id_1": r1["image_id"],
                    "relative_path_1": r1["relative_path"],
                    "class_1": r1["class_label"],
                    "source_group_1": r1["source_group"],
                    "image_id_2": r2["image_id"],
                    "relative_path_2": r2["relative_path"],
                    "class_2": r2["class_label"],
                    "source_group_2": r2["source_group"],
                    "hamming_distance": int(dist),
                    "is_cross_class_conflict": is_cross_class,
                    "status": "manual review required",
                })

    cols = [
        "image_id_1",
        "relative_path_1",
        "class_1",
        "source_group_1",
        "image_id_2",
        "relative_path_2",
        "class_2",
        "source_group_2",
        "hamming_distance",
        "is_cross_class_conflict",
        "status",
    ]
    return pd.DataFrame(records, columns=cols), summary


def find_label_conflicts(
    exact_duplicates_df: pd.DataFrame,
    near_duplicates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Consolidate suspected label conflicts (exact or near duplicates across differing classes)."""
    conflict_rows: List[Dict[str, Any]] = []

    # Exact duplicates across classes
    if not exact_duplicates_df.empty:
        cross_exact = exact_duplicates_df[exact_duplicates_df["conflict_type"] == "cross_class_conflict"]
        for _, row in cross_exact.iterrows():
            conflict_rows.append({
                "conflict_type": "exact_duplicate_cross_class",
                "image_id_1": row["image_id_1"],
                "relative_path_1": row["relative_path_1"],
                "class_1": row["class_1"],
                "source_group_1": row["source_group_1"],
                "image_id_2": row["image_id_2"],
                "relative_path_2": row["relative_path_2"],
                "class_2": row["class_2"],
                "source_group_2": row["source_group_2"],
                "metric_value": "SHA-256 match",
                "recommendation": "CRITICAL: Same image labeled with different classes. Must be resolved before training.",
            })

    # Near duplicates across classes
    if not near_duplicates_df.empty:
        cross_near = near_duplicates_df[near_duplicates_df["is_cross_class_conflict"] == True]
        for _, row in cross_near.iterrows():
            conflict_rows.append({
                "conflict_type": "near_duplicate_cross_class",
                "image_id_1": row["image_id_1"],
                "relative_path_1": row["relative_path_1"],
                "class_1": row["class_1"],
                "source_group_1": row["source_group_1"],
                "image_id_2": row["image_id_2"],
                "relative_path_2": row["relative_path_2"],
                "class_2": row["class_2"],
                "source_group_2": row["source_group_2"],
                "metric_value": f"Hamming distance {row['hamming_distance']}",
                "recommendation": "WARNING: Perceptually near-identical images in different classes. Manual expert review required.",
            })

    cols = [
        "conflict_type",
        "image_id_1",
        "relative_path_1",
        "class_1",
        "source_group_1",
        "image_id_2",
        "relative_path_2",
        "class_2",
        "source_group_2",
        "metric_value",
        "recommendation",
    ]
    return pd.DataFrame(conflict_rows, columns=cols)
