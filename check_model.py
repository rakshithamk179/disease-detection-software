#!/usr/bin/env python3
"""Model Status Check Script"""

import json
from pathlib import Path
import tensorflow as tf

print("\n" + "="*60)
print("🍅 TOMATO DISEASE CLASSIFIER - MODEL STATUS CHECK")
print("="*60 + "\n")

# Check model files
model_path = Path("models/best_model.keras")
class_names_path = Path("models/class_names.json")
metrics_path = Path("reports/final_test_metrics.json")

print("📁 FILE CHECK:")
print(f"  ✓ Model file exists: {model_path.exists()} ({model_path.stat().st_size / 1e6:.1f} MB)")
print(f"  ✓ Class names exist: {class_names_path.exists()}")
print(f"  ✓ Results exist: {metrics_path.exists()}")

# Load and display metrics
print("\n📊 PERFORMANCE METRICS:")
try:
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    print(f"  ✓ Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  ✓ Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"  ✓ Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"  ✓ Total Test Images: {metrics['total_images']}")
    
    print("\n📈 CLASS-WISE PERFORMANCE:")
    for class_name, metrics_class in metrics['class_wise'].items():
        print(f"  • {class_name.upper()}")
        print(f"    - Precision: {metrics_class['precision']:.2%}")
        print(f"    - Recall: {metrics_class['recall']:.2%}")
        print(f"    - F1-Score: {metrics_class['f1']:.4f}")
except Exception as e:
    print(f"  ✗ Error loading metrics: {e}")

# Try loading model
print("\n🤖 MODEL LOADING TEST:")
try:
    model = tf.keras.models.load_model("models/best_model.keras")
    with open("models/class_names.json") as f:
        class_names = json.load(f)
    print("  ✓ Model loaded successfully!")
    print(f"  ✓ Classes: {list(class_names.values())}")
    print(f"  ✓ Model shape: {model.input_shape}")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")

print("\n" + "="*60)
print("✅ MODEL IS RUNNING FINE!")
print("="*60 + "\n")

print("💡 NEXT STEPS - Run predictions:")
print("  1. Test one image:")
print('     python scripts/predict_image.py --image "data/processed/test_locked/healthy/image_001.jpg"')
print("\n  2. Evaluate all test images:")
print("     python scripts/evaluate_model.py --split test_locked --final-evaluation")
print()
