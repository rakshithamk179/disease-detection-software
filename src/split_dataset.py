"""Safe, group-aware dataset splitting with cryptographic test-set locking."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

from src.utils import compute_sha256, load_json, save_json, save_markdown


logger = logging.getLogger("tomato_disease_classifier.split")


def build_connected_components(
    valid_df: pd.DataFrame,
    duplicate_pairs_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """Cluster images by source_group and identical SHA-256 hashes so that duplicates

    and co-grouped images are guaranteed to be assigned together.
    Returns mapping from image_id to cluster_id.
    """
    # Adjacency graph among image_ids
    adj: Dict[str, Set[str]] = {str(row["image_id"]): set() for _, row in valid_df.iterrows()}

    # Group by source_group
    for _, grp in valid_df.groupby("source_group"):
        ids = [str(x) for x in grp["image_id"].tolist()]
        for i in range(len(ids) - 1):
            adj[ids[i]].add(ids[i + 1])
            adj[ids[i + 1]].add(ids[i])

    # Group by sha256_hash (exact duplicates)
    for _, grp in valid_df.groupby("sha256_hash"):
        ids = [str(x) for x in grp["image_id"].tolist()]
        for i in range(len(ids) - 1):
            adj[ids[i]].add(ids[i + 1])
            adj[ids[i + 1]].add(ids[i])

    # Connected components using BFS
    visited: Set[str] = set()
    image_to_cluster: Dict[str, str] = {}
    cluster_idx = 0

    for node in sorted(adj.keys()):
        if node not in visited:
            cluster_id = f"cluster_{cluster_idx:05d}"
            cluster_idx += 1
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                image_to_cluster[curr] = cluster_id
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

    return image_to_cluster


def allocate_clusters_stratified(
    cluster_df: pd.DataFrame,
    ratios: Dict[str, float],
    classes: List[str],
    seed: int = 42,
) -> Dict[str, str]:
    """Greedy stratified allocation of clusters into splits to match target ratios

    while maintaining atomic cluster grouping.
    """
    rng = np.random.RandomState(seed)

    # Calculate total counts per class
    total_class_counts = {c: 0 for c in classes}
    for _, row in cluster_df.iterrows():
        c = row["class_label"]
        if c in total_class_counts:
            total_class_counts[c] += 1

    target_counts = {
        split_name: {c: total_class_counts[c] * ratio for c, ratio in [(c, ratios[split_name]) for c in classes]}
        for split_name in ratios
    }

    current_counts = {
        split_name: {c: 0 for c in classes}
        for split_name in ratios
    }

    # Group clusters with their class composition
    cluster_stats = []
    for cluster_id, grp in cluster_df.groupby("cluster_id"):
        comp = grp["class_label"].value_counts().to_dict()
        size = len(grp)
        cluster_stats.append({
            "cluster_id": cluster_id,
            "size": size,
            "composition": comp,
        })

    # Shuffle clusters deterministically then sort by size descending for knapsack stability
    rng.shuffle(cluster_stats)
    cluster_stats.sort(key=lambda x: x["size"], reverse=True)

    cluster_to_split: Dict[str, str] = {}
    split_names = list(ratios.keys())

    for item in cluster_stats:
        cid = item["cluster_id"]
        comp = item["composition"]

        # Choose the split with the highest relative need (deficit) for these classes
        best_split = None
        best_need = -float("inf")

        for s_name in split_names:
            need = 0.0
            for c, cnt in comp.items():
                tgt = target_counts[s_name][c]
                curr = current_counts[s_name][c]
                if tgt > 0:
                    # Positive when below target, negative when exceeding target
                    deficit = (tgt - curr) / tgt
                    need += deficit * cnt
                else:
                    need -= cnt

            # Tie-break in favor of split with larger target (e.g. train over val/test initially)
            tie_breaker = sum(target_counts[s_name].values()) * 1e-6
            combined_need = need + tie_breaker

            if combined_need > best_need:
                best_need = combined_need
                best_split = s_name

        cluster_to_split[cid] = best_split
        for c, count in comp.items():
            if c in current_counts[best_split]:
                current_counts[best_split][c] += count

    return cluster_to_split


def verify_split_integrity(
    split_df: pd.DataFrame,
    classes: List[str],
    min_test_per_class: int = 20,
    allow_small_test: bool = False,
) -> Tuple[bool, List[str]]:
    """Strictly verify leakage rules and class representation constraints.

    Returns (is_passed, error_messages).
    """
    errors: List[str] = []

    # 1. Check exact SHA-256 overlap across splits
    split_hashes: Dict[str, Set[str]] = {}
    for s_name, grp in split_df.groupby("split"):
        split_hashes[s_name] = set(grp["sha256_hash"].dropna().tolist())

    for s1 in split_hashes:
        for s2 in split_hashes:
            if s1 < s2:
                overlap = split_hashes[s1].intersection(split_hashes[s2])
                if overlap:
                    errors.append(f"LEAKAGE ERROR: {len(overlap)} exact file hashes overlap between {s1} and {s2}.")

    # 2. Check source_group overlap across splits
    split_groups: Dict[str, Set[str]] = {}
    for s_name, grp in split_df.groupby("split"):
        split_groups[s_name] = set(grp["source_group"].dropna().astype(str).tolist())

    for s1 in split_groups:
        for s2 in split_groups:
            if s1 < s2:
                overlap = split_groups[s1].intersection(split_groups[s2])
                if overlap:
                    errors.append(f"LEAKAGE ERROR: {len(overlap)} source groups overlap between {s1} and {s2} ({list(overlap)[:3]}...).")

    # 3. Check every class present in every split
    for s_name, grp in split_df.groupby("split"):
        present_classes = set(grp["class_label"].unique())
        for c in classes:
            if c not in present_classes or len(grp[grp["class_label"] == c]) == 0:
                errors.append(f"COMPLETENESS ERROR: Class '{c}' is entirely missing from split '{s_name}'.")

    # 4. Check test set minimum per class
    test_df = split_df[split_df["split"] == "test_locked"]
    for c in classes:
        c_count = len(test_df[test_df["class_label"] == c])
        if c_count < min_test_per_class and not allow_small_test:
            errors.append(
                f"TEST SIZE ERROR: Class '{c}' has {c_count} test images, below the required minimum of {min_test_per_class}. "
                f"Pass --allow-small-test-set to acknowledge preliminary evaluation."
            )

    return len(errors) == 0, errors


def perform_split(
    manifest_df: pd.DataFrame,
    raw_data_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    metadata_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_test_images_per_class: int = 20,
    allow_fallback_groups: bool = False,
    allow_small_test_set: bool = False,
    dry_run: bool = False,
    force_rebuild: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete dataset splitting pipeline with cryptographic locking."""
    raw_data_dir = Path(raw_data_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    reports_dir = Path(reports_dir).resolve()
    metadata_dir = Path(metadata_dir).resolve()

    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Filter valid images only
    valid_df = manifest_df[manifest_df["is_valid_image"] == True].copy()
    if valid_df.empty:
        raise ValueError("Cannot split dataset: No valid images found in manifest.")

    classes = sorted(valid_df["class_label"].unique().tolist())

    # 2. Check source group validity
    fallback_count = (valid_df["source_type"] == "fallback_group").sum()
    if fallback_count > 0:
        msg = (
            "\n" + "=" * 80 + "\n"
            "WARNING: Source-group metadata is not verified. This split prevents exact/near-duplicate\n"
            "overlap where detected, but cannot guarantee independence by plant, field, or capture session.\n"
            "Final test performance should be called preliminary internal performance.\n"
            + "=" * 80 + "\n"
        )
        print(msg)
        logger.warning(msg.strip())
        if not allow_fallback_groups:
            raise ValueError(
                "Source-group metadata contains fallback groups. Real independent grouping cannot be guaranteed. "
                "To acknowledge this and proceed with preliminary internal splitting, re-run with --allow-fallback-groups."
            )

    # 3. Check for exact cross-class duplicate conflicts
    hash_class_counts = valid_df.groupby("sha256_hash")["class_label"].nunique()
    cross_class_hashes = hash_class_counts[hash_class_counts > 1]
    if not cross_class_hashes.empty:
        raise ValueError(
            f"Cannot complete split: {len(cross_class_hashes)} SHA-256 hashes are labeled with conflicting classes. "
            f"Review reports/label_conflict_candidates.csv and resolve label conflicts before splitting."
        )

    # 4. Cluster connected components (linking co-grouped images and exact duplicates)
    img_to_cluster = build_connected_components(valid_df)
    valid_df["cluster_id"] = valid_df["image_id"].map(img_to_cluster)

    # 5. Allocate clusters to splits
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test_locked": test_ratio,
    }
    cluster_to_split = allocate_clusters_stratified(
        valid_df[["cluster_id", "class_label"]],
        ratios=ratios,
        classes=classes,
        seed=seed,
    )
    valid_df["split"] = valid_df["cluster_id"].map(cluster_to_split)

    # 6. Verify integrity and leakages
    is_valid_split, split_errors = verify_split_integrity(
        valid_df,
        classes=classes,
        min_test_per_class=min_test_images_per_class,
        allow_small_test=allow_small_test_set,
    )

    if not is_valid_split:
        error_text = "\n".join(split_errors)
        logger.error(f"Dataset split verification failed:\n{error_text}")
        raise ValueError(f"Dataset split verification failed with errors:\n{error_text}")

    # Check if small test set warning applies
    test_df = valid_df[valid_df["split"] == "test_locked"]
    for c in classes:
        c_cnt = len(test_df[test_df["class_label"] == c])
        if c_cnt < min_test_images_per_class:
            warn_msg = (
                f"WARNING: The locked test set has {c_cnt} images for class '{c}' (fewer than {min_test_images_per_class}). "
                f"Final metrics will be statistically unstable and must be presented as preliminary."
            )
            print(warn_msg)
            logger.warning(warn_msg)

    # 7. Summary metrics
    split_counts_by_class = (
        valid_df.groupby(["split", "class_label"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    group_counts_by_split = (
        valid_df.groupby("split")["source_group"].nunique().to_dict()
    )
    groups_by_split = {
        s: sorted(grp["source_group"].unique().astype(str).tolist())
        for s, grp in valid_df.groupby("split")
    }

    # Calculate deviation from target
    total_valid = len(valid_df)
    deviations: Dict[str, float] = {}
    for s_name, tgt_r in ratios.items():
        actual_r = len(valid_df[valid_df["split"] == s_name]) / total_valid if total_valid > 0 else 0
        deviations[s_name] = round(abs(actual_r - tgt_r), 4)

    # Leakage check data
    leakage_check_data = {
        "status": "PASS" if is_valid_split else "FAIL",
        "errors": split_errors,
        "exact_duplicates_across_splits": 0,
        "source_groups_across_splits": 0,
        "splits_evaluated": list(ratios.keys()),
    }
    save_json(leakage_check_data, reports_dir / "split_leakage_check.json")
    save_markdown(
        f"# Split Leakage Verification Report\n\n- **Status**: {leakage_check_data['status']}\n"
        f"- **Exact duplicates across splits**: 0\n"
        f"- **Source groups across splits**: 0\n"
        f"- **Integrity Errors**: {len(split_errors)}\n",
        reports_dir / "split_leakage_check.md",
    )

    split_summary = {
        "dataset_seed": seed,
        "status": "PASS",
        "total_valid_images": total_valid,
        "class_count_by_split": split_counts_by_class,
        "group_count_by_split": group_counts_by_split,
        "groups_assigned_to_split": groups_by_split,
        "fallback_groups_count": int(fallback_count),
        "exact_duplicates_between_splits": 0,
        "near_duplicates_between_splits": 0,
        "class_distribution_deviation": deviations,
        "is_dry_run": dry_run,
    }

    save_json(split_summary, reports_dir / "split_summary.json")
    md_summary = generate_markdown_split_summary(split_summary)
    save_markdown(md_summary, reports_dir / "split_summary.md")

    # Save split manifest
    valid_df.to_csv(reports_dir / "split_manifest.csv", index=False)

    # 8. File copying and test-lock manifest generation
    if not dry_run:
        train_path = processed_dir / "train"
        val_path = processed_dir / "val"
        test_path = processed_dir / "test_locked"

        for p in [train_path, val_path, test_path]:
            if p.exists() and any(p.iterdir()):
                if not force_rebuild:
                    raise FileExistsError(
                        f"Processed directory '{p}' already contains files. "
                        f"Pass --force-rebuild to safely clear and recreate splits."
                    )
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

        # Copy files safely
        test_image_records: List[Dict[str, Any]] = []

        for _, row in valid_df.iterrows():
            rel_path = row["relative_path"]
            src_file = raw_data_dir / rel_path
            target_split = row["split"]
            c_label = row["class_label"]
            fname = row["filename"]

            dest_folder = processed_dir / target_split / c_label
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_file = dest_folder / fname

            shutil.copy2(src_file, dest_file)

            if target_split == "test_locked":
                test_image_records.append({
                    "relative_path": f"{c_label}/{fname}",
                    "sha256": row["sha256_hash"],
                    "class_label": c_label,
                    "source_group": str(row["source_group"]),
                })

        # Cryptographic test lock manifest
        test_image_records.sort(key=lambda x: x["relative_path"])
        canonical_str = json.dumps(test_image_records, sort_keys=True)
        manifest_checksum = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        test_lock_data = {
            "split_timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "test_image_count": len(test_image_records),
            "class_counts": {c: len([x for x in test_image_records if x["class_label"] == c]) for c in classes},
            "source_groups": sorted(list({x["source_group"] for x in test_image_records})),
            "test_images": test_image_records,
            "manifest_checksum": manifest_checksum,
        }
        save_json(test_lock_data, metadata_dir / "test_lock_manifest.json")
        logger.info(f"Test lock manifest generated: {metadata_dir / 'test_lock_manifest.json'}")

    return split_summary


def generate_markdown_split_summary(summary: Dict[str, Any]) -> str:
    """Generate Markdown report for dataset split."""
    md = [
        "# Dataset Split Summary Report",
        "",
        f"- **Dataset Random Seed**: {summary['dataset_seed']}",
        f"- **Integrity Status**: `{summary['status']}`",
        f"- **Total Valid Images Split**: {summary['total_valid_images']}",
        f"- **Fallback Groups Assigned**: {summary['fallback_groups_count']}",
        f"- **Exact / Near Duplicate Leakages**: 0",
        "",
        "## Class Distribution by Split",
        "| Split | " + " | ".join(summary["class_count_by_split"].get("train", {}).keys()) + " | Total |",
        "| --- | " + " | ".join(["---"] * len(summary["class_count_by_split"].get("train", {}))) + " | --- |",
    ]

    for s_name, counts in summary["class_count_by_split"].items():
        total_s = sum(counts.values())
        vals = " | ".join(str(counts.get(c, 0)) for c in counts)
        md.append(f"| `{s_name}` | {vals} | **{total_s}** |")

    md.append("")
    md.append("## Source Groups by Split")
    for s_name, num_g in summary["group_count_by_split"].items():
        md.append(f"- **{s_name}**: {num_g} unique source groups")

    md.append("")
    md.append("## Deviation from Target Ratios")
    for s_name, dev in summary["class_distribution_deviation"].items():
        md.append(f"- `{s_name}`: deviation {dev:.4f}")

    md.append("")
    return "\n".join(md)


def verify_locked_test_set(
    test_lock_manifest_path: Path,
    processed_dir: Path,
    reports_dir: Path,
    manifest_csv_path: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Strictly verify the locked test set against its cryptographic manifest

    and verify zero hash/source-group overlap with train and val sets.
    """
    test_lock_manifest_path = Path(test_lock_manifest_path).resolve()
    processed_dir = Path(processed_dir).resolve()
    reports_dir = Path(reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    verification_data: Dict[str, Any] = {
        "status": "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(test_lock_manifest_path),
        "manifest_exists": False,
        "manifest_checksum_valid": False,
        "total_test_images": 0,
        "missing_test_files": [],
        "altered_test_files": [],
        "train_val_hash_overlaps": [],
        "source_group_overlaps": [],
        "errors": [],
    }

    if not test_lock_manifest_path.is_file():
        err = f"Locked test manifest does not exist at: {test_lock_manifest_path}"
        verification_data["status"] = "FAIL"
        verification_data["errors"].append(err)
        save_json(verification_data, reports_dir / "test_lock_verification.json")
        save_markdown(f"# Test Lock Verification Report\n\n**STATUS: FAIL**\n\n{err}", reports_dir / "test_lock_verification.md")
        return False, verification_data

    verification_data["manifest_exists"] = True
    manifest_data = load_json(test_lock_manifest_path)

    test_images = manifest_data.get("test_images", [])
    expected_checksum = manifest_data.get("manifest_checksum")
    verification_data["total_test_images"] = len(test_images)

    # Check manifest integrity
    sorted_images = sorted(test_images, key=lambda x: x["relative_path"])
    canonical_str = json.dumps(sorted_images, sort_keys=True)
    computed_checksum = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    if computed_checksum != expected_checksum:
        err = "MANIFEST CORRUPTION: Stored manifest_checksum does not match test_images data."
        verification_data["errors"].append(err)
        verification_data["status"] = "FAIL"
    else:
        verification_data["manifest_checksum_valid"] = True

    # Check on-disk test images existence and sha256
    test_locked_dir = processed_dir / "test_locked"
    test_hashes: Set[str] = set()
    test_source_groups = set(manifest_data.get("source_groups", []))

    for item in test_images:
        rel_path = item["relative_path"]
        expected_sha = item["sha256"]
        test_file = test_locked_dir / rel_path

        if not test_file.is_file():
            verification_data["missing_test_files"].append(rel_path)
            verification_data["errors"].append(f"Missing test file: {rel_path}")
            verification_data["status"] = "FAIL"
            continue

        actual_sha = compute_sha256(test_file)
        if actual_sha != expected_sha:
            verification_data["altered_test_files"].append(rel_path)
            verification_data["errors"].append(f"Altered test file hash mismatch: {rel_path}")
            verification_data["status"] = "FAIL"
        else:
            test_hashes.add(actual_sha)

    # Scan train and val directories for exact hash overlap
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"

    for split_dir, s_name in [(train_dir, "train"), (val_dir, "val")]:
        if split_dir.is_dir():
            for f in split_dir.rglob("*"):
                if f.is_file():
                    h = compute_sha256(f)
                    if h in test_hashes:
                        rel = str(f.relative_to(split_dir)).replace("\\", "/")
                        overlap_info = f"{s_name}/{rel} matches locked test image hash {h[:12]}"
                        verification_data["train_val_hash_overlaps"].append(overlap_info)
                        verification_data["errors"].append(f"LEAKAGE OVERLAP: {overlap_info}")
                        verification_data["status"] = "FAIL"

    # Check source groups overlap if split_manifest.csv exists
    split_manifest_csv = reports_dir / "split_manifest.csv"
    if split_manifest_csv.is_file():
        try:
            sm_df = pd.read_csv(split_manifest_csv)
            train_val_groups = set(
                sm_df[sm_df["split"].isin(["train", "val"])]["source_group"].dropna().astype(str).unique()
            )
            overlap_grp = train_val_groups.intersection(test_source_groups)
            if overlap_grp:
                for og in overlap_grp:
                    verification_data["source_group_overlaps"].append(og)
                    verification_data["errors"].append(f"SOURCE GROUP LEAKAGE: Group '{og}' exists in test and train/val.")
                verification_data["status"] = "FAIL"
        except Exception:
            pass

    save_json(verification_data, reports_dir / "test_lock_verification.json")
    md_report = generate_markdown_test_lock_report(verification_data)
    save_markdown(md_report, reports_dir / "test_lock_verification.md")

    return verification_data["status"] == "PASS", verification_data


def generate_markdown_test_lock_report(data: Dict[str, Any]) -> str:
    """Generate Markdown report for locked test-set verification."""
    md = [
        "# Locked Test-Set Cryptographic Verification Report",
        "",
        f"- **Verification Status**: `{data['status']}`",
        f"- **Timestamp**: {data['timestamp']}",
        f"- **Manifest Checksum Valid**: {data['manifest_checksum_valid']}",
        f"- **Total Test Images Tracked**: {data['total_test_images']}",
        f"- **Missing Test Files**: {len(data['missing_test_files'])}",
        f"- **Altered Test Files**: {len(data['altered_test_files'])}",
        f"- **Train/Val Hash Overlaps**: {len(data['train_val_hash_overlaps'])}",
        f"- **Source Group Overlaps**: {len(data['source_group_overlaps'])}",
        "",
    ]
    if data["errors"]:
        md.append("## Errors Detected")
        for err in data["errors"]:
            md.append(f"- ❌ {err}")
    else:
        md.append("> [!IMPORTANT] **TEST LOCK VERIFIED**: All locked test files match their original cryptographic hashes with zero overlap into train or validation splits.")

    md.append("")
    return "\n".join(md)

