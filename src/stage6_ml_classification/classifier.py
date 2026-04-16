"""
STAGE 6: ML Classification (Random Forest)

Train and apply a Random Forest classifier on the 40-feature vector
extracted from Stages 1-5. Includes training, cross-validation,
threshold calibration, and prediction.

Input: 40-feature vector (4 + 18 + 10 + 8)
Output: Fraud probability, risk level, feature importance
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import config


class FoodFraudClassifier:
    """
    Wrapper around Random Forest classifier for food image fraud detection.
    Handles training, evaluation, threshold calibration, and prediction.
    """

    FEATURE_NAMES = [
        # Stage 1: Metadata (4)
        "exif_camera", "timestamp_consistency", "editing_software", "jpeg_quality",
        # Stage 3: Spatial (18)
        "lbp_uniform", "noise_R", "noise_G", "noise_B",
        "chroma_lum_ratio", "laplacian_energy", "edge_coherence",
        "rgb_mean_R", "rgb_mean_G", "rgb_mean_B",
        "rgb_std_R", "rgb_std_G", "rgb_std_B",
        "rgb_skew_R", "rgb_skew_G", "rgb_skew_B",
        "ycbcr_mean_Cb", "ycbcr_mean_Cr",
        "ycbcr_std_Cb", "ycbcr_std_Cr",
        "noise_corr_RG", "noise_corr_GB",
        # Stage 4: Frequency (10)
        "fft_hf_ratio", "spectral_entropy",
        "dct_coeff_std", "texture_smoothness", "noise_residual_energy",
        "freq_band_1", "freq_band_2", "freq_band_3", "freq_band_4",
        "dct_double_comp",
        # Stage 5: Region (8)
        "patch_anomaly_mean", "patch_anomaly_std", "patch_anomaly_max",
        "sharpness_variance", "color_variance", "anomaly_variance",
        "semantic_text_flag", "semantic_geometry_flag",
    ]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray, save_path: str = None) -> dict:
        """
        Train the Random Forest classifier.

        Args:
            X: Feature matrix of shape (n_samples, 40).
            y: Labels of shape (n_samples,) — 0=real, 1=fraud.
            save_path: Optional path to save trained model.

        Returns:
            dict with training metrics.
        """
        # Handle missing sklearn-scaler in path
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        # Replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Cross-validation
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
        cv_accuracy = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="roc_auc")

        # Best threshold calibration
        self.threshold = _calibrate_threshold(self.model, X_scaled, y)

        # Evaluation on training data
        y_pred = (self.model.predict_proba(X_scaled)[:, 1] >= self.threshold).astype(int)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        metrics = {
            "cv_accuracy_mean": float(np.mean(cv_accuracy)),
            "cv_accuracy_std": float(np.std(cv_accuracy)),
            "cv_auc_mean": float(np.mean(cv_auc)),
            "cv_auc_std": float(np.std(cv_auc)),
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "train_auc": float(roc_auc_score(y, y_proba)),
            "calibrated_threshold": float(self.threshold),
            "feature_importances": list(self.model.feature_importances_),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }

        # Save model
        if save_path:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, save_path)
            joblib.dump(self.threshold, str(config.MODELS_DIR / "threshold.joblib"))
            joblib.dump(self.scaler, str(config.MODELS_DIR / "scaler.joblib"))
            print(f"Model saved to {save_path}")
            print(f"Threshold saved to {config.MODELS_DIR / 'threshold.joblib'}")

        return metrics

    def predict(self, feature_vector: np.ndarray) -> dict:
        """
        Predict fraud probability for a single feature vector.

        Args:
            feature_vector: numpy array of shape (40,).

        Returns:
            dict with:
                'fraud_probability': float [0, 1]
                'risk_level': 'low' | 'medium' | 'high'
                'feature_contributions': dict mapping feature names to importance
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.array(feature_vector).reshape(1, -1).astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0, 1]
        is_fraud = proba >= self.threshold
        risk = _get_risk_level(proba)

        # Feature contributions (importance-weighted)
        contributions = dict(zip(
            self.FEATURE_NAMES,
            self.model.feature_importances_ * X_scaled[0]
        ))

        return {
            "fraud_probability": float(proba),
            "is_fraud": bool(is_fraud),
            "risk_level": risk,
            "feature_contributions": contributions,
            "feature_names": self.FEATURE_NAMES,
        }

    def predict_batch(self, X: np.ndarray) -> list:
        """Predict for multiple feature vectors. Returns list of dicts."""
        return [self.predict(x) for x in X]

    @staticmethod
    def load_from_disk(model_path: str = None) -> "FoodFraudClassifier":
        """
        Load a trained model from disk.

        Args:
            model_path: Path to .joblib model file. Defaults to config.MODEL_PATH.
        """
        if model_path is None:
            model_path = str(config.MODEL_PATH)

        clf = FoodFraudClassifier()
        clf.model = joblib.load(model_path)

        scaler_path = str(config.MODELS_DIR / "scaler.joblib")
        if Path(scaler_path).exists():
            clf.scaler = joblib.load(scaler_path)

        thresh_path = str(config.MODELS_DIR / "threshold.joblib")
        if Path(thresh_path).exists():
            clf.threshold = joblib.load(thresh_path)

        clf.is_fitted = True
        return clf


def _calibrate_threshold(model, X_scaled, y):
    """
    Find the optimal classification threshold by maximizing
    F1 score across candidate thresholds.
    """
    proba = model.predict_proba(X_scaled)[:, 1]
    best_f1 = 0
    best_thresh = 0.5

    for t in np.linspace(0.1, 0.9, config.THRESHOLD_MODELS):
        preds = (proba >= t).astype(int)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return best_thresh


def _get_risk_level(probability: float) -> str:
    """Convert fraud probability to risk level."""
    if probability < config.RISK_THRESHOLDS["low"]:
        return "low"
    elif probability < config.RISK_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "high"
