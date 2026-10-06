# Raw Dataset Audit Report

## 1. Summary of Dataset Integrity
- **Total images scanned**: 4500
- **Valid images**: 4500
- **Invalid / corrupt images**: 0
- **Class distribution (valid)**: `{'late_blight': 1909, 'healthy': 1591, 'early_blight': 1000}`
- **Class balance ratio**: 1.91 (Imbalanced (>1.2) - class weights recommended)

## 2. Spatial Dimension & Aspect Ratio Statistics
- **Width Range**: 256 px to 256 px (Mean: 256.0 px, Median: 256.0 px)
- **Height Range**: 256 px to 256 px (Mean: 256.0 px, Median: 256.0 px)
- **Aspect Ratio**: Mean: 1.0, Median: 1.0

## 3. Duplicate and Label Conflict Analysis
- **Exact duplicate pairs (SHA-256)**: 14
  - Within same class: 14
  - Cross-class conflicts: 0
  - Across source groups: 0
- **Near-duplicate pairs (pHash Hamming <= 6)**: 51
  - Within same class: 48
  - Cross-class conflicts: 3
- **Total suspected label conflicts**: 3


## 4. Source Group Independence Evaluation
- **Total unique source groups**: 258
- **Verified independent groups**: 258
- **Fallback groups**: 0
- **Group counts per class**: `{'early_blight': 251, 'healthy': 174, 'late_blight': 235}`

> [!NOTE] Verified source-group metadata is present and sufficient for genuine group-aware splitting.
