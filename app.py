"""
Tomato Disease Classifier - Dashboard Server
Runs on http://localhost:5000
"""

from flask import Flask, render_template, jsonify
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Load metrics data
def load_metrics():
    """Load final test metrics from JSON"""
    metrics_file = REPORTS_DIR / "final_test_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}

def load_predictions():
    """Load per-image predictions"""
    pred_file = REPORTS_DIR / "final_test_predictions.csv"
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        return df.to_dict('records')
    return []

def load_training_summary():
    """Load training summary"""
    summary_file = REPORTS_DIR / "training_summary.md"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return f.read()
    return ""

def load_split_summary():
    """Load split summary"""
    summary_file = REPORTS_DIR / "split_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return {}

# Routes
@app.route('/')
def index():
    """Serve main dashboard"""
    metrics = load_metrics()
    split_info = load_split_summary()
    
    return render_template('dashboard.html', 
                         metrics=metrics,
                         split_info=split_info)

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics"""
    return jsonify(load_metrics())

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for predictions"""
    predictions = load_predictions()
    return jsonify({
        'total': len(predictions),
        'predictions': predictions[:50]  # Return first 50
    })

@app.route('/api/stats')
def api_stats():
    """API endpoint for quick stats"""
    metrics = load_metrics()
    split_info = load_split_summary()
    
    return jsonify({
        'accuracy': metrics.get('accuracy', 0),
        'macro_f1': metrics.get('macro_f1', 0),
        'weighted_f1': metrics.get('weighted_f1', 0),
        'inference_latency': metrics.get('inference_latency', {}),
        'class_counts': split_info.get('class_counts', {}),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/class-performance')
def api_class_performance():
    """API endpoint for class-wise performance"""
    metrics = load_metrics()
    class_wise = metrics.get('class_wise', {})
    
    return jsonify({
        'classes': list(class_wise.keys()),
        'performance': class_wise
    })

@app.route('/api/confusion-matrix')
def api_confusion_matrix():
    """API endpoint for confusion matrix"""
    metrics = load_metrics()
    
    return jsonify({
        'matrix': metrics.get('confusion_matrix', []),
        'matrix_normalized': metrics.get('confusion_matrix_normalized', []),
        'classes': metrics.get('class_names', ['early_blight', 'healthy', 'late_blight'])
    })

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    metrics = load_metrics()
    return jsonify({
        'status': 'ok',
        'model_exists': (MODELS_DIR / 'best_model.keras').exists(),
        'metrics_loaded': bool(metrics),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🍅 TOMATO DISEASE CLASSIFIER - DASHBOARD SERVER")
    print("=" * 70)
    print(f"\n✓ Project root: {PROJECT_ROOT}")
    print(f"✓ Reports loaded from: {REPORTS_DIR}")
    print(f"✓ Models path: {MODELS_DIR}")
    
    print("\n" + "=" * 70)
    print("🌐 STARTING FLASK SERVER")
    print("=" * 70)
    print("\n📊 Dashboard URL: http://localhost:5000")
    print("📊 API Endpoint: http://localhost:5000/api/metrics")
    print("\n🔗 API Routes Available:")
    print("   • http://localhost:5000/api/metrics          - All metrics")
    print("   • http://localhost:5000/api/predictions      - Sample predictions")
    print("   • http://localhost:5000/api/stats            - Quick stats")
    print("   • http://localhost:5000/api/class-performance- Class-wise metrics")
    print("   • http://localhost:5000/api/confusion-matrix - Confusion matrix data")
    print("   • http://localhost:5000/api/health           - Server health check")
    
    print("\n💡 Press CTRL+C to stop the server\n")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
