"""
Flask Web Application for Food Image Fraud Detection.

Provides a web UI for uploading food images and getting
fraud analysis results with heatmaps and reports.

Usage:
    python -m src.app
    # Then open http://127.0.0.1:5000
"""

import os
import sys
from pathlib import Path
import random
import string
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, render_template, jsonify, redirect, url_for
import config

app = Flask(__name__,
            template_folder=str(config.BASE_DIR / "templates"),
            static_folder=str(config.BASE_DIR / "static"))
app.config["SECRET_KEY"] = config.SECRET_KEY

# Create uploads directory
config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Global pipeline (lazy-loaded on first request)
_pipeline = None


def get_pipeline():
    """Get or create the fraud detection pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        model_path = config.MODEL_PATH
        if model_path.exists():
            print(f"Loading trained model from {model_path}")
            from src.pipeline import FraudDetectionPipeline
            _pipeline = FraudDetectionPipeline(str(model_path))
        else:
            print("No trained model found. Please run `python -m src.train_model` first.")
            print("Running in demo mode with placeholder predictions.")
    return _pipeline


def allowed_file(filename: str) -> bool:
    """Check if the filename has a valid extension."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Home page with upload form."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handle image upload and return analysis result."""
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, WebP, or BMP."}), 400

    # Save uploaded file
    upload_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_path = config.UPLOAD_FOLDER / f"{upload_id}.{ext}"
    file.save(str(upload_path))

    # Check file size
    if upload_path.stat().st_size > config.MAX_FILE_SIZE:
        upload_path.unlink()
        return jsonify({"error": "File too large. Max 10MB."}), 400

    pipeline = get_pipeline()

    if pipeline is not None:
        # Run full analysis with unique heatmap filename to avoid caching issues
        try:
            # Clean up old heatmap files to avoid clutter
            static_dir = config.BASE_DIR / "static"
            for old_file in static_dir.glob("heatmap_*.png"):
                try:
                    old_file.unlink(missing_ok=True)
                except Exception:
                    pass

            # Generate unique heatmap filename using upload_id and timestamp
            import time
            heatmap_filename = f"heatmap_{upload_id}_{int(time.time())}.png"
            heatmap_path = static_dir / heatmap_filename

            result = pipeline.analyze(str(upload_path), heatmap_output_path=str(heatmap_path))

            # Clean up upload
            upload_path.unlink(missing_ok=True)

            # Build heatmap URL with cache-busting timestamp
            heatmap_url = f"/static/{heatmap_filename}?t={int(time.time())}"

            return jsonify({
                "success": True,
                "report": result["report"],
                "heatmap_url": heatmap_url,
                "fraud_probability": result["classification"]["fraud_probability"],
                "risk_level": result["classification"]["risk_level"],
                "is_fraud": result["classification"]["is_fraud"],
            })
        except Exception as e:
            # Clean up on error
            upload_path.unlink(missing_ok=True)
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    else:
        # Demo mode — return placeholder result
        upload_path.unlink(missing_ok=True)
        return jsonify({
            "success": True,
            "demo_mode": True,
            "message": "No trained model found. Showing placeholder result.",
            "report": {
                "verdict": "PLACEHOLDER — No model trained",
                "risk_level": "MEDIUM",
                "risk_color": "#f39c12",
                "fraud_probability": 50.0,
                "classification": "—",
                "top_suspicious_features": [],
            },
            "heatmap_url": "",
            "fraud_probability": 0.5,
            "risk_level": "medium",
            "is_fraud": False,
        })


@app.route("/gallery")
def gallery():
    """Gallery page showing real and fake images from the dataset."""
    static_images_dir = config.BASE_DIR / "static" / "images"

    real_images = []
    fake_images = []

    if config.REAL_DIR.exists():
        real_images = sorted([f.name for f in config.REAL_DIR.iterdir() if f.is_file()])

    if config.FAKE_DIR.exists():
        fake_images = sorted([f.name for f in config.FAKE_DIR.iterdir() if f.is_file()])

    # Copy images to static folder for serving
    real_static = static_images_dir / "real"
    fake_static = static_images_dir / "fake"

    for folder, images, src_dir in [
        (real_static, real_images, config.REAL_DIR),
        (fake_static, fake_images, config.FAKE_DIR),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        for img_name in images:
            dst = folder / img_name
            if not dst.exists():
                src = src_dir / img_name
                if src.exists():
                    shutil.copy2(str(src), str(dst))

    return render_template(
        "gallery.html",
        real_images=real_images,
        fake_images=fake_images,
        real_count=len(real_images),
        fake_count=len(fake_images),
    )


@app.route("/about")
def about():
    """About page explaining the project and pipeline."""
    real_count = len(list(config.REAL_DIR.iterdir())) if config.REAL_DIR.exists() else 0
    fake_count = len(list(config.FAKE_DIR.iterdir())) if config.FAKE_DIR.exists() else 0
    return render_template("about.html", real_count=real_count, fake_count=fake_count)


@app.route("/api/health")
def health():
    """Health check endpoint."""
    model_status = "loaded" if config.MODEL_PATH.exists() else "not found"
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "pipeline_available": _pipeline is not None,
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  Food Image Fraud Detector — Web Server")
    print("=" * 50)
    print("  Starting at: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
