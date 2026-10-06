"""Read-only dataset audit module for raw tomato leaf images and manifest."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.duplicate_detection import (
    find_exact_duplicates,
    find_near_duplicates,
    find_label_conflicts,
)
from src.utils import save_json, save_markdown


def audit_raw_dataset(
    manifest_df: pd.DataFrame,
    reports_dir: Path,
    hamming_threshold: int = 6,
    small_image_threshold: int = 128,
) -> Dict[str, Any]:
    """Perform comprehensive read-only audit of raw dataset manifest.

    Outputs JSON, Markdown, and CSV reports under reports_dir.
    """
    reports_dir = Path(reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(manifest_df)
    if total_images == 0:
        empty_audit = {
            "total_images": 0,
            "status": "EMPTY_DATASET",
            "message": "Manifest is empty. No images found in raw directory.",
        }
        save_json(empty_audit, reports_dir / "raw_dataset_audit.json")
        save_markdown("# Raw Dataset Audit Report\n\n**Manifest is empty.**", reports_dir / "raw_dataset_audit.md")
        return empty_audit

    # 1. Valid vs Invalid
    valid_df = manifest_df[manifest_df["is_valid_image"] == True]
    invalid_df = manifest_df[manifest_df["is_valid_image"] == False]
    valid_count = len(valid_df)
    invalid_count = len(invalid_df)

    # 2. Images per class
    class_counts = manifest_df["class_label"].value_counts().to_dict()
    valid_class_counts = valid_df["class_label"].value_counts().to_dict()

    # 3. Class balance ratio
    if valid_class_counts:
        counts = list(valid_class_counts.values())
        max_c = max(counts)
        min_c = min(counts)
        balance_ratio = float(max_c / min_c) if min_c > 0 else 0.0
    else:
        balance_ratio = 0.0

    # 4. Dimensions and Aspect Ratio Distribution
    dim_stats: Dict[str, Any] = {}
    small_images: List[Dict[str, Any]] = []
    if valid_count > 0:
        widths = valid_df["image_width"].astype(int)
        heights = valid_df["image_height"].astype(int)
        aspect_ratios = widths / heights.replace(0, 1)

        dim_stats = {
            "min_width": int(widths.min()),
            "max_width": int(widths.max()),
            "mean_width": float(round(widths.mean(), 2)),
            "median_width": float(widths.median()),
            "min_height": int(heights.min()),
            "max_height": int(heights.max()),
            "mean_height": float(round(heights.mean(), 2)),
            "median_height": float(heights.median()),
            "mean_aspect_ratio": float(round(aspect_ratios.mean(), 3)),
            "median_aspect_ratio": float(round(aspect_ratios.median(), 3)),
        }

        # Small image warning check
        small_df = valid_df[(widths < small_image_threshold) | (heights < small_image_threshold)]
        for _, row in small_df.iterrows():
            small_images.append({
                "image_id": row["image_id"],
                "relative_path": row["relative_path"],
                "width": int(row["image_width"]),
                "height": int(row["image_height"]),
            })

    # 5. Duplicates and Conflicts
    exact_dup_df, exact_summary = find_exact_duplicates(manifest_df)
    near_dup_df, near_summary = find_near_duplicates(manifest_df, hamming_threshold=hamming_threshold)
    label_conflicts_df = find_label_conflicts(exact_dup_df, near_dup_df)

    # 6. Source group distribution by class
    group_counts_by_class: Dict[str, int] = {}
    fallback_group_count = 0
    real_group_count = 0

    if not manifest_df.empty:
        for c, grp in manifest_df.groupby("class_label"):
            group_counts_by_class[str(c)] = int(grp["source_group"].nunique())

        total_unique_groups = int(manifest_df["source_group"].nunique())
        fallback_df = manifest_df[manifest_df["source_type"] == "fallback_group"]
        fallback_group_count = int(fallback_df["source_group"].nunique())
        real_group_count = total_unique_groups - fallback_group_count
    else:
        total_unique_groups = 0

    # 7. Metadata sufficiency check for group-wise splitting
    is_group_metadata_sufficient = (
        fallback_group_count == 0 and
        total_unique_groups >= 6 and
        all(count >= 2 for count in group_counts_by_class.values())
    )

    # Save CSV tables
    exact_dup_df.to_csv(reports_dir / "exact_duplicates.csv", index=False)
    near_dup_df.to_csv(reports_dir / "near_duplicates.csv", index=False)
    label_conflicts_df.to_csv(reports_dir / "label_conflict_candidates.csv", index=False)

    # Consolidate audit data
    audit_data: Dict[str, Any] = {
        "dataset_summary": {
            "total_images": total_images,
            "valid_images": valid_count,
            "invalid_images": invalid_count,
            "class_counts_total": class_counts,
            "class_counts_valid": valid_class_counts,
            "class_imbalance_ratio": round(balance_ratio, 3),
            "is_imbalanced": balance_ratio > 1.2,
        },
        "dimension_statistics": dim_stats,
        "small_image_warnings": {
            "threshold": small_image_threshold,
            "count": len(small_images),
            "samples": small_images[:10],
        },
        "exact_duplicates": exact_summary,
        "near_duplicates": near_summary,
        "label_conflicts": {
            "total_conflicts": len(label_conflicts_df),
            "critical_exact_conflicts": int((label_conflicts_df["conflict_type"] == "exact_duplicate_cross_class").sum()) if not label_conflicts_df.empty else 0,
            "warning_near_conflicts": int((label_conflicts_df["conflict_type"] == "near_duplicate_cross_class").sum()) if not label_conflicts_df.empty else 0,
        },
        "source_group_analysis": {
            "total_unique_groups": total_unique_groups,
            "fallback_groups_count": fallback_group_count,
            "verified_real_groups_count": real_group_count,
            "group_counts_by_class": group_counts_by_class,
            "is_sufficient_for_genuine_group_split": is_group_metadata_sufficient,
        },
    }

    # Save JSON report
    save_json(audit_data, reports_dir / "raw_dataset_audit.json")

    # Generate Markdown report
    md_content = generate_markdown_audit_report(audit_data)
    save_markdown(md_content, reports_dir / "raw_dataset_audit.md")

    return audit_data


def generate_markdown_audit_report(data: Dict[str, Any]) -> str:
    """Generate user-facing Markdown report from audit data."""
    summary = data["dataset_summary"]
    dims = data.get("dimension_statistics", {})
    exact = data["exact_duplicates"]
    near = data["near_duplicates"]
    conflicts = data["label_conflicts"]
    grp = data["source_group_analysis"]

    md = []
    md.append("# Raw Dataset Audit Report")
    md.append("")
    md.append("## 1. Summary of Dataset Integrity")
    md.append(f"- **Total images scanned**: {summary['total_images']}")
    md.append(f"- **Valid images**: {summary['valid_images']}")
    md.append(f"- **Invalid / corrupt images**: {summary['invalid_images']}")
    md.append(f"- **Class distribution (valid)**: `{summary['class_counts_valid']}`")
    md.append(f"- **Class balance ratio**: {summary['class_imbalance_ratio']:.2f} ({'Imbalanced (>1.2) - class weights recommended' if summary['is_imbalanced'] else 'Balanced'})")
    md.append("")

    md.append("## 2. Spatial Dimension & Aspect Ratio Statistics")
    if dims:
        md.append(f"- **Width Range**: {dims.get('min_width')} px to {dims.get('max_width')} px (Mean: {dims.get('mean_width')} px, Median: {dims.get('median_width')} px)")
        md.append(f"- **Height Range**: {dims.get('min_height')} px to {dims.get('max_height')} px (Mean: {dims.get('mean_height')} px, Median: {dims.get('median_height')} px)")
        md.append(f"- **Aspect Ratio**: Mean: {dims.get('mean_aspect_ratio')}, Median: {dims.get('median_aspect_ratio')}")
    else:
        md.append("No valid image spatial statistics available.")
    md.append("")

    md.append("## 3. Duplicate and Label Conflict Analysis")
    md.append(f"- **Exact duplicate pairs (SHA-256)**: {exact.get('total_exact_duplicate_pairs', 0)}")
    md.append(f"  - Within same class: {exact.get('within_class_pairs', 0)}")
    md.append(f"  - Cross-class conflicts: {exact.get('cross_class_pairs', 0)}")
    md.append(f"  - Across source groups: {exact.get('cross_group_pairs', 0)}")
    md.append(f"- **Near-duplicate pairs (pHash Hamming <= {near.get('hamming_threshold', 6)})**: {near.get('total_near_duplicate_pairs', 0)}")
    md.append(f"  - Within same class: {near.get('within_class_near_duplicates', 0)}")
    md.append(f"  - Cross-class conflicts: {near.get('cross_class_near_duplicates', 0)}")
    md.append(f"- **Total suspected label conflicts**: {conflicts.get('total_conflicts', 0)}")
    md.append("")
    if conflicts.get("critical_exact_conflicts", 0) > 0:
        md.append("> [!CAUTION] **CRITICAL LABEL CONFLICTS FOUND**: Identical images are tagged with different classes. These must be reviewed in `reports/label_conflict_candidates.csv`.")
    md.append("")

    md.append("## 4. Source Group Independence Evaluation")
    md.append(f"- **Total unique source groups**: {grp.get('total_unique_groups', 0)}")
    md.append(f"- **Verified independent groups**: {grp.get('verified_real_groups_count', 0)}")
    md.append(f"- **Fallback groups**: {grp.get('fallback_groups_count', 0)}")
    md.append(f"- **Group counts per class**: `{grp.get('group_counts_by_class', {})}`")
    md.append("")
    if grp.get("is_sufficient_for_genuine_group_split"):
        md.append("> [!NOTE] Verified source-group metadata is present and sufficient for genuine group-aware splitting.")
    else:
        md.append("> [!WARNING] **Source-group metadata is insufficient or consists of fallback groups.** Group-wise splitting cannot guarantee independence by plant, farm plot, or capture session without manual metadata verification.")
    md.append("")

    return "\n".join(md)
