"""
Global configuration for the Food Image Fraud Detector.
All tunable parameters centralized here.
"""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "food_fraud_detector.joblib"
STATS_PATH = MODELS_DIR / "scaler_stats.joblib"

# --- Image Processing ---
TARGET_SIZE = (256, 256)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "JPG", "JPEG", "PNG", "WEBP", "BMP"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# --- Stage 5: Region Analysis ---
PATCH_GRID_SIZE = 4  # 4x4 grid = 16 patches
PATCH_SIZE = 64      # 256/4 = 64 pixels per patch
ANOMALY_THRESHOLD = 2.0  # Z-score threshold for patch anomalies

# --- Stage 6: Classification ---
CLASSIFIER_TYPE = "random_forest"
TEST_SPLIT = 0.2
CV_FOLDS = 5
THRESHOLD_MODELS = 100  # Number of candidate thresholds to evaluate

# --- Flask Server ---
UPLOAD_FOLDER = BASE_DIR / "uploads"
SECRET_KEY = "food-fraud-detector-secret-key-change-in-production"

# --- Risk Level Thresholds ---
RISK_THRESHOLDS = {
    "low": 0.35,
    "medium": 0.60,
}
