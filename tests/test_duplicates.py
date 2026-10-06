"""Tests for exact duplicate, near-duplicate, and label conflict detection."""

from pathlib import Path
import pandas as pd
from PIL import Image
import pytest

from src.duplicate_detection import (
    compute_hamming_distance,
    find_exact_duplicates,
    find_label_conflicts,
    find_near_duplicates,
)


def test_compute_hamming_distance():
    # Identical hex hashes -> distance 0
    h1 = "ffff0000ffff0000"
    assert compute_hamming_distance(h1, h1) == 0

    # 1 bit difference: ffff0000ffff0000 vs ffff0000ffff0001 (0x0 vs 0x1 -> 1 bit)
    h2 = "ffff0000ffff0001"
    assert compute_hamming_distance(h1, h2) == 1

    # Completely inverted 64 bits -> 64
    h_zeros = "0000000000000000"
    h_ones = "ffffffffffffffff"
    assert compute_hamming_distance(h_zeros, h_ones) == 64

    # Invalid input
    assert compute_hamming_distance("", "abc") == 9999


def test_exact_duplicate_detection():
    data = [
        {
            "image_id": "IMG_1",
            "relative_path": "early_blight/img1.jpg",
            "class_label": "early_blight",
            "source_group": "group_a",
            "is_valid_image": True,
            "sha256_hash": "aaa111",
            "perceptual_hash": "f0f0f0f0",
        },
        {
            "image_id": "IMG_2",
            "relative_path": "early_blight/img2.jpg",
            "class_label": "early_blight",
            "source_group": "group_a",
            "is_valid_image": True,
            "sha256_hash": "aaa111",  # Same hash, same class
            "perceptual_hash": "f0f0f0f0",
        },
        {
            "image_id": "IMG_3",
            "relative_path": "healthy/img3.jpg",
            "class_label": "healthy",
            "source_group": "group_b",
            "is_valid_image": True,
            "sha256_hash": "aaa111",  # Same hash, different class (cross-class conflict!)
            "perceptual_hash": "f0f0f0f0",
        },
        {
            "image_id": "IMG_4",
            "relative_path": "healthy/img4.jpg",
            "class_label": "healthy",
            "source_group": "group_c",
            "is_valid_image": True,
            "sha256_hash": "bbb222",  # Unique
            "perceptual_hash": "00000000",
        },
    ]
    df = pd.DataFrame(data)

    dup_df, summary = find_exact_duplicates(df)
    assert summary["total_exact_duplicate_pairs"] == 3  # (IMG1, IMG2), (IMG1, IMG3), (IMG2, IMG3)
    assert summary["within_class_pairs"] == 1
    assert summary["cross_class_pairs"] == 2

    conflicts = find_label_conflicts(dup_df, pd.DataFrame())
    assert len(conflicts) == 2
    assert (conflicts["conflict_type"] == "exact_duplicate_cross_class").all()


def test_near_duplicate_report_format():
    # Two distinct images with near pHash (Hamming distance 2)
    data = [
        {
            "image_id": "IMG_A",
            "relative_path": "early_blight/a.jpg",
            "class_label": "early_blight",
            "source_group": "group_1",
            "is_valid_image": True,
            "sha256_hash": "hash_a",
            "perceptual_hash": "ffff0000ffff0000",
        },
        {
            "image_id": "IMG_B",
            "relative_path": "healthy/b.jpg",
            "class_label": "healthy",
            "source_group": "group_2",
            "is_valid_image": True,
            "sha256_hash": "hash_b",
            "perceptual_hash": "ffff0000ffff0003",  # 2 bit difference from IMG_A
        },
    ]
    df = pd.DataFrame(data)

    near_df, summary = find_near_duplicates(df, hamming_threshold=6)
    assert len(near_df) == 1
    row = near_df.iloc[0]
    assert row["image_id_1"] == "IMG_A"
    assert row["image_id_2"] == "IMG_B"
    assert row["hamming_distance"] == 2
    assert row["status"] == "manual review required"
    assert bool(row["is_cross_class_conflict"]) is True
