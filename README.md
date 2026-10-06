# Tomato Leaf Disease Classifier

A **production-quality, scientifically honest** tomato leaf disease classification system using MobileNetV3Small, two-stage transfer learning, and cryptographic locked-test-set protection.

---

## 1. Project Purpose

Classify a tomato leaf image into exactly one of three classes:

| Label | Description |
|-------|-------------|
| `early_blight` | Early blight disease (*Alternaria solani*) |
| `healthy` | Visually healthy tomato leaf |
| `late_blight` | Late blight disease (*Phytophthora infestans*) |

The system prioritizes **scientifically honest evaluation**. A credible 80–90% accuracy on a properly separated test set is far more meaningful than 100% on a contaminated or trivially easy test set.

---

## 2. Pipeline Architecture

```
Raw Images
    ↓
scripts/create_manifest.py       (scan, hash, infer source groups)
    ↓
scripts/audit_dataset.py         (duplicate check, label conflict detection)
    ↓
scripts/split_dataset.py         (group-aware split + cryptographic test lock)
    ↓
scripts/verify_locked_test.py    (SHA-256 integrity verification)
    ↓
scripts/train_model.py           (Stage 1 head + Stage 2 fine-tune, val only)
    ↓
scripts/evaluate_model.py --split val        (validation performance)
    ↓
scripts/evaluate_model.py --split test_locked --final-evaluation   (final only)
    ↓
scripts/export_model.py          (TFLite float32 + dynamic-range quantized)
    ↓
scripts/predict_image.py         (single-image inference)
```

---

## 3. Supported Classes

- `early_blight`
- `healthy`
- `late_blight`

> [!IMPORTANT]
> These folder names must match exactly inside `data/raw/`. Any other subfolders are silently skipped.

---

## 4. Installation

### Requirements

- Python 3.10 or higher
- Windows / Linux / macOS

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd tomato_disease_classifier

# 2. Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify TensorFlow and core libraries
python -c "import tensorflow as tf, sklearn, imagehash, cv2; print('TF:', tf.__version__)"
```

---

## 5. Required Dataset Layout

Place your real tomato leaf images in:

```
data/raw/
  early_blight/      ← images of early blight disease
  healthy/           ← images of healthy tomato leaves
  late_blight/       ← images of late blight disease
```

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.webp`

> [!NOTE]
> The PlantVillage dataset (`Tomato___Early_blight`, `Tomato___healthy`, `Tomato___Late_blight` folders) from the `PlantVillage-Dataset/raw/color/` directory will work directly. The manifest scanner auto-reads the leaf-map.json if found alongside the dataset for real plant-level source groups.

---

## 6. How to Add Meaningful Source Metadata

After running `create_manifest.py`, open `data/metadata/image_manifest.csv` and populate these columns for your images if known:

| Column | What to fill |
|--------|-------------|
| `source_group` | A unique group ID per independent source (plant, field plot, camera session) |
| `source_type` | `verified_plant_leaf`, `field_plot`, `camera_session`, etc. |
| `plant_id` | Physical plant identifier if available |
| `capture_session` | Session / date / camera identifier |
| `capture_date` | ISO date of capture (e.g. `2026-05-01`) |
| `camera_or_source` | Camera model or dataset origin |
| `notes` | Free text notes |

> [!WARNING]
> **Group-wise splitting is only scientifically meaningful if `source_group` represents a true independent source**, such as a plant ID, farm plot, capture session, date/camera session, or original dataset source. File names alone may not be enough. If only fallback groups are available, use `--allow-fallback-groups` and treat final metrics as **preliminary internal performance**.

---

## 7. Why Group-Wise Splitting Matters

Standard random splitting risks placing images of the same physical leaf or plant in both training and test sets. When the same plant appears in both sets, the test set is no longer truly "unseen" data — this artificially inflates accuracy. Group-aware splitting guarantees that every image from a given source group is confined to exactly one split.

---

## 8. Step-by-Step Commands

### Step 1 – Scan raw images and create manifest

```bash
python scripts/create_manifest.py
```

- Reads: `data/raw/`
- Creates: `data/metadata/image_manifest.csv`
- Sets: class labels from folder names, source groups from leaf-map or filename patterns

---

### Step 2 – Audit dataset for duplicates and conflicts

```bash
python scripts/audit_dataset.py
```

- Reads: `data/metadata/image_manifest.csv` (read-only, never modifies raw data)
- Saves: `reports/raw_dataset_audit.json`, `reports/raw_dataset_audit.md`
- Saves: `reports/exact_duplicates.csv`, `reports/near_duplicates.csv`, `reports/label_conflict_candidates.csv`

---

### Step 3 – Dry-run the split

```bash
python scripts/split_dataset.py --dry-run
```

Shows what the split would produce without writing any files.

### Step 4 – Execute the split and lock test set

```bash
python scripts/split_dataset.py
```

Additional flags:

```bash
python scripts/split_dataset.py --allow-fallback-groups   # if no real group metadata
python scripts/split_dataset.py --force-rebuild           # overwrite existing splits
python scripts/split_dataset.py --seed 42                 # set random seed
python scripts/split_dataset.py --allow-small-test-set    # if fewer than 20 test images/class
```

- Creates: `data/processed/train/`, `data/processed/val/`, `data/processed/test_locked/`
- Creates: `data/metadata/test_lock_manifest.json` (cryptographic SHA-256 manifest)

---

### Step 5 – Verify locked test set integrity

```bash
python scripts/verify_locked_test.py
```

- Verifies every test file's SHA-256 matches the locked manifest
- Detects tampering, deletion, or accidental overlap with train/val
- **Must pass before training begins**

---

### Step 6 – Train the model (train + validation only)

```bash
python scripts/train_model.py
```

Optional flags:

```bash
python scripts/train_model.py --stage stage1         # head training only
python scripts/train_model.py --skip-fine-tune       # skip Stage 2
python scripts/train_model.py --stage1-epochs 20
python scripts/train_model.py --fine-tune-epochs 10
python scripts/train_model.py --batch-size 16
python scripts/train_model.py --resume               # continue from checkpoint
```

> [!IMPORTANT]
> **The locked test set (`data/processed/test_locked/`) is NEVER loaded during training.**

Saves:
- `models/stage1_best_model.keras`
- `models/best_model.keras`
- `models/class_names.json`
- `logs/training_history_stage1.csv`
- `logs/training_history_finetune.csv`
- `reports/training_summary.md`
- `reports/training_accuracy.png`
- `reports/training_loss.png`

---

### Step 7 – Evaluate on validation set

```bash
python scripts/evaluate_model.py --split val
```

- Uses only `data/processed/val/`
- Saves: `reports/validation_metrics.json`, confusion matrix PNGs
- Prints performance warnings if accuracy < 75% or macro F1 < 0.70

---

### Step 8 – Final locked test evaluation

```bash
python scripts/evaluate_model.py --split test_locked --final-evaluation
```

> [!CAUTION]
> Only run this command **once**, after all model development is finalized. Do not use test results to further tune the model.

Saves:
- `reports/final_test_metrics.json`
- `reports/final_test_confusion_matrix.png`
- `reports/final_test_predictions.csv`
- `reports/final_test_summary.md`

---

### Step 9 – Export TensorFlow Lite models

```bash
python scripts/export_model.py
```

Produces:
- `models/tomato_disease_mobilenetv3_float32.tflite`
- `models/tomato_disease_mobilenetv3_dynamic_range.tflite`
- `reports/tflite_export_report.md`

---

### Step 10 – Run inference on a single image

```bash
python scripts/predict_image.py --image path/to/leaf.jpg
python scripts/predict_image.py --image path/to/leaf.jpg --save-annotated
```

Outputs: predicted class, confidence %, class probabilities, inference latency

---

## 9. How to Interpret Metrics

| Metric | What it Means |
|--------|--------------|
| **Accuracy** | Fraction of images correctly classified |
| **Macro F1** | Average F1 per class; penalizes poor performance on any single class |
| **Weighted F1** | F1 weighted by class support; reflects overall dataset balance |
| **Recall (per class)** | Fraction of true disease cases correctly detected |
| **Precision (per class)** | Fraction of positive predictions that are actually correct |
| **Confusion Matrix** | Shows which classes get confused with each other |
| **Normalized Confusion Matrix** | Row-normalized: each row sums to 1.0 (shows recall per class) |

---

## 10. Honest Reporting Template

Use this exact template when reporting final results:

> **"On a locked internal test set containing [N] images across three classes, the model achieved [accuracy]% accuracy and [macro_f1] macro F1-score. The test split was separated using [grouping method]. These results estimate internal generalization only and do not by themselves demonstrate field or drone deployment performance."**

---

## 11. Limitations

- **Curated images differ from field/drone images.** PlantVillage images are clean, controlled lab shots. Real-world performance under variable field lighting, angles, and camera quality may be lower.
- **Group splitting requires meaningful source metadata.** Without real plant/session-level grouping, the split cannot guarantee plant-level independence.
- **Classifier confidence is not disease severity.** The confidence score indicates classifier certainty for this image; it does not measure infected leaf area or disease spread extent.
- **Pesticide/spray actuation is out of scope.** Classifying disease is not the same as prescribing treatment.
- **Real agricultural deployment needs agronomist, regulatory, and safety validation.** This system is for research and educational use.

---

## 12. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` |
| `ModuleNotFoundError: imagehash` | Run `pip install ImageHash Pillow` |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python` |
| Split fails with "fallback groups" | Add `--allow-fallback-groups` flag |
| Split fails with "small test set" | Add `--allow-small-test-set` flag |
| Test evaluation denied | Add `--final-evaluation` flag to evaluate_model.py |
| Model not found | Run `train_model.py` before `evaluate_model.py` |
| Corrupt image errors | Check `reports/raw_dataset_audit.md` for list of invalid files |
| Training crashes mid-run | Use `--resume` flag to continue from checkpoint |

---

## 13. Git Ignore Policy

The following are excluded from version control:
- Raw data and processed splits (`data/raw/`, `data/processed/`)
- Trained model weights (`models/*.keras`, `models/*.tflite`)
- Training logs and reports with large binary content
- Virtual environment (`.venv/`)

The following **are** tracked:
- All source code (`src/`, `scripts/`, `tests/`)
- Configuration (`config/config.yaml`)
- `models/class_names.json`
- `data/metadata/image_manifest.csv` (except `test_lock_manifest.json`)
- `reports/*.md`, `reports/*.json` (summary reports only, not binary PNGs/CSVs)

---

## 14. Run All Tests

```bash
pytest -q
```

All tests run without a GPU. Tests cover:
- Manifest label detection from folder names
- Invalid image flagging
- Exact and near-duplicate detection
- Group isolation across train/val/test
- Cryptographic test lock tamper detection
- Training code never accesses locked test data
- TFLite export never uses locked test images
- Inference output JSON-serializable and no severity claim
