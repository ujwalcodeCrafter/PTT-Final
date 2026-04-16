"""
Complete 6-Stage Fraud Detection Pipeline

Imports from all stage modules and chains them into a single inference pipeline.
This is the main entry point for analyzing images.

Usage:
    from src.pipeline import FraudDetectionPipeline

    pipeline = FraudDetectionPipeline(clf_path="models/food_fraud_detector.joblib")
    result = pipeline.analyze("path/to/image.jpg")
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1_metadata.extractor import extract_stage1_features
from src.stage2_preprocessing.preprocessor import load_and_preprocess
from src.stage3_forensic_spatial.extractor import extract_stage3_features
from src.stage4_frequency_analysis.extractor import extract_stage4_features
from src.stage5_region_analysis.extractor import extract_stage5_features
from src.stage6_ml_classification.classifier import FoodFraudClassifier
from src.visualization.report import (
    generate_heatmap,
    generate_feature_importance_plot,
    generate_fraud_report,
)


class FraudDetectionPipeline:
    """
    Complete 6-stage forensic analysis pipeline.

    Stages:
      1. Metadata extraction (4 features)
      2. Preprocessing (resize, normalize, color conversion)
      3. Forensic spatial features (18 features)
      4. Frequency analysis (10 features)
      5. Region-based analysis (8 features)
      6. ML classification & report generation
    """

    def __init__(self, clf_path: str = None):
        """
        Initialize the pipeline with a trained classifier.

        Args:
            clf_path: Path to .joblib model file. If None, tries config.MODEL_PATH.
        """
        if clf_path:
            self.classifier = FoodFraudClassifier.load_from_disk(clf_path)
        else:
            self.classifier = FoodFraudClassifier()

    def extract_features(self, file_path: str) -> np.ndarray:
        """
        Run Stages 1-5 to extract the full 40-feature vector.

        Args:
            file_path: Path to the input image.

        Returns:
            numpy array of shape (40,) — concatenated features from all stages.
        """
        # Stage 1: Metadata (needs file path, not preprocessed image)
        meta_features = extract_stage1_features(file_path)  # (4,)

        # Stage 2: Preprocessing
        proc_dict = load_and_preprocess(file_path)

        # Stage 3: Spatial features
        spatial_features = extract_stage3_features(proc_dict)  # (18,)

        # Stage 4: Frequency features
        freq_features = extract_stage4_features(proc_dict)  # (10,)

        # Stage 5: Region features
        region_features = extract_stage5_features(proc_dict, file_path)  # (8,)

        # Concatenate all features: 4 + 18 + 10 + 8 = 40
        full_vector = np.concatenate([
            meta_features,
            spatial_features,
            freq_features,
            region_features,
        ])

        return full_vector, proc_dict

    def analyze(self, file_path: str, heatmap_output_path: str = None) -> dict:
        """
        Full pipeline: extract features → classify → generate report.

        Args:
            file_path: Path to the input image.
            heatmap_output_path: Optional custom path for saving the heatmap.
                If None, uses default static/heatmap.png.

        Returns:
            dict with:
                'classification': result from Stage 6
                'report': formatted fraud report
                'heatmap_path': path to generated heatmap (if enabled)
                'feature_vector': the raw 40-feature vector
                'proc_dict': preprocessed image data (for further visualization)
        """
        # Stages 1-5: feature extraction
        feature_vector, proc_dict = self.extract_features(file_path)

        # Stage 6: ML classification
        clf_result = self.classifier.predict(feature_vector)

        # Visualization
        heatmap_path = ""
        try:
            heatmap_path = generate_heatmap(proc_dict, output_path=heatmap_output_path)
        except Exception as e:
            print(f"[Visualization Warning] Heatmap generation failed: {e}")

        # Generate fraud report
        report = generate_fraud_report(clf_result, feature_vector)

        return {
            "classification": clf_result,
            "report": report,
            "heatmap_path": heatmap_path,
            "feature_vector": feature_vector,
            "proc_dict": proc_dict,
        }

    def analyze_features(self, feature_vector: np.ndarray) -> dict:
        """
        Analyze using a pre-extracted feature vector (skip Stages 1-5).
        Useful for batch analysis.

        Args:
            feature_vector: numpy array of shape (40,).

        Returns:
            dict with classification and report.
        """
        clf_result = self.classifier.predict(feature_vector)
        report = generate_fraud_report(clf_result, feature_vector)

        return {
            "classification": clf_result,
            "report": report,
            "feature_vector": feature_vector,
        }
