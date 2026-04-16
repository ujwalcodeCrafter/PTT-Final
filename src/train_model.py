"""
Training script for the Food Image Fraud Detector.

Usage:
    # Train on default data directories (data/real and data/fake)
    python -m src.train_model

    # Train with custom paths
    python -m src.train_model --real data/my_real --fake data/my_fake --output models/my_model.joblib
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.pipeline import FraudDetectionPipeline
from src.stage6_ml_classification.classifier import FoodFraudClassifier


def collect_dataset(real_dir: str, fake_dir: str):
    """
    Load all images from real/fake directories and extract features.

    Returns:
        X: np.ndarray of shape (n_samples, 40)
        y: np.ndarray of shape (n_samples,) — 0 for real, 1 for fake
        file_paths: list of file paths (for debugging)
    """
    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)

    real_images = [f for f in real_dir.iterdir()
                   if f.suffix.lower()[1:] in config.ALLOWED_EXTENSIONS]
    fake_images = [f for f in fake_dir.iterdir()
                   if f.suffix.lower()[1:] in config.ALLOWED_EXTENSIONS]

    print(f"Found {len(real_images)} real images, {len(fake_images)} fake images")

    # Use pipeline for feature extraction but no classifier
    from src.stage1_metadata.extractor import extract_stage1_features
    from src.stage2_preprocessing.preprocessor import load_and_preprocess
    from src.stage3_forensic_spatial.extractor import extract_stage3_features
    from src.stage4_frequency_analysis.extractor import extract_stage4_features
    from src.stage5_region_analysis.extractor import extract_stage5_features

    X_real = []
    X_fake = []
    paths = []

    # Process real images
    for img_path in real_images:
        try:
            print(f"  Processing real: {img_path.name}")
            meta = extract_stage1_features(str(img_path))
            proc = load_and_preprocess(str(img_path))
            spatial = extract_stage3_features(proc)
            freq = extract_stage4_features(proc)
            region = extract_stage5_features(proc, str(img_path))

            features = np.concatenate([meta, spatial, freq, region])
            X_real.append(features)
            paths.append(str(img_path))
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")

    # Process fake images
    for img_path in fake_images:
        try:
            print(f"  Processing fake: {img_path.name}")
            meta = extract_stage1_features(str(img_path))
            proc = load_and_preprocess(str(img_path))
            spatial = extract_stage3_features(proc)
            freq = extract_stage4_features(proc)
            region = extract_stage5_features(proc, str(img_path))

            features = np.concatenate([meta, spatial, freq, region])
            X_fake.append(features)
            paths.append(str(img_path))
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")

    if not X_real and not X_fake:
        raise ValueError("No images found. Please add images to data/real and data/fake.")

    X = np.array(X_real + X_fake, dtype=np.float64)
    y = np.array([0] * len(X_real) + [1] * len(X_fake), dtype=np.int64)

    print(f"\nDataset: {len(X_real)} real + {len(X_fake)} fake = {len(X)} total")
    return X, y, paths


def main():
    parser = argparse.ArgumentParser(description="Train the Food Fraud Detector")
    parser.add_argument("--real", default=str(config.REAL_DIR),
                        help="Path to real images directory")
    parser.add_argument("--fake", default=str(config.FAKE_DIR),
                        help="Path to fake images directory")
    parser.add_argument("--output", default=str(config.MODEL_PATH),
                        help="Path to save the trained model")
    args = parser.parse_args()

    print("=" * 60)
    print("  FOOD IMAGE FRAUD DETECTOR — Training")
    print("=" * 60)

    # Step 1: Collect dataset
    print("\n[1/3] Loading images and extracting features...")
    X, y, paths = collect_dataset(args.real, args.fake)

    # Step 2: Train model
    print("\n[2/3] Training Random Forest classifier...")
    clf = FoodFraudClassifier()
    metrics = clf.train(X, y, save_path=args.output)

    # Step 3: Print results
    print("\n[3/3] Training Results")
    print("-" * 40)
    print(f"  CV Accuracy:  {metrics['cv_accuracy_mean']:.2%} (+/- {metrics['cv_accuracy_std']:.2%})")
    print(f"  CV ROC-AUC:   {metrics['cv_auc_mean']:.2%} (+/- {metrics['cv_auc_std']:.2%})")
    print(f"  Train Acc:    {metrics['train_accuracy']:.2%}")
    print(f"  Train AUC:    {metrics['train_auc']:.2%}")
    print(f"  Threshold:    {metrics['calibrated_threshold']:.4f}")
    print(f"  Confusion:    {metrics['confusion_matrix']}")
    print("-" * 40)

    # Top 10 features
    print("\nTop 10 Most Important Features:")
    fi = metrics["feature_importances"]
    for i in np.argsort(fi)[::-1][:10]:
        print(f"  {clf.FEATURE_NAMES[i]:30s}: {fi[i]:.4f}")

    print("\n" + "=" * 60)
    print(f"  Model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
