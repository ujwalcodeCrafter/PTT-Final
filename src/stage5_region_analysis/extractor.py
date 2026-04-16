"""
STAGE 5: Region-Based Analysis (8 features)

Divide image into a 4x4 grid of 64x64 patches and analyze
each region for anomalies using z-scores, sharpness variance,
color consistency, and semantic consistency checks.

Features (8):
  0. patch_anomaly_mean       — Mean z-score anomaly across patches
  1. patch_anomaly_std        — Std of patch anomaly scores
  2. patch_anomaly_max        — Max z-score anomaly (worst patch)
  3. sharpness_variance       — Variance of patch-level sharpness
  4. color_variance           — Variance of patch-level color std
  5. anomaly_variance         — Variance across patch anomalies
  6. semantic_text_flag       — Binary: gibberish/random text detected [NEW]
  7. semantic_geometry_flag   — Binary: impossible food geometry flag [NEW]
"""

import numpy as np
import cv2
from scipy import stats
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import config


def extract_stage5_features(proc_dict: dict, file_path: str = None) -> np.ndarray:
    """
    Extract 8 region-based features from preprocessed image.

    Args:
        proc_dict: Output of stage2 preprocessing.
        file_path: Optional path to original image file (for semantic checks).

    Returns:
        numpy array of shape (8,).
    """
    rgb = proc_dict["rgb"]
    gray = proc_dict["gray"]
    h, w = gray.shape

    patch_h = h // config.PATCH_GRID_SIZE
    patch_w = w // config.PATCH_GRID_SIZE

    patch_anomalies = []
    patch_sharpness = []
    patch_colors = []

    # Compute per-patch metrics
    for row in range(config.PATCH_GRID_SIZE):
        for col in range(config.PATCH_GRID_SIZE):
            y1, y2 = row * patch_h, (row + 1) * patch_h
            x1, x2 = col * patch_w, (col + 1) * patch_w

            patch_gray = gray[y1:y2, x1:x2]
            patch_rgb = rgb[y1:y2, x1:x2]

            # Anomaly score: z-score of pixel intensity distribution
            patch_mean = np.mean(patch_gray)
            patch_std = np.std(patch_gray)
            patch_skew = float(stats.skew(patch_gray.flatten()))

            anomaly_score = abs(patch_skew) + (patch_mean - 0.5) ** 2
            patch_anomalies.append(anomaly_score)

            # Sharpness: Laplacian variance in the patch
            if patch_gray.min() >= 0 and patch_gray.max() <= 1.0:
                patch_8u = (patch_gray * 255).astype(np.uint8)
            else:
                patch_8u = np.clip(patch_gray, 0, 1).astype(np.float64)
                patch_8u = (patch_8u * 255).astype(np.uint8)
            sharpness = cv2.Laplacian(patch_8u, cv2.CV_64F).var()
            patch_sharpness.append(sharpness)

            # Color consistency: mean per-channel std
            color_std = np.mean([np.std(patch_rgb[:, :, c]) for c in range(3)])
            patch_colors.append(color_std)

    patch_anomalies = np.array(patch_anomalies)
    patch_sharpness = np.array(patch_sharpness)
    patch_colors = np.array(patch_colors)

    # Global z-score based anomaly detection
    z_scores = stats.zscore(patch_anomalies)
    anomaly_count = np.sum(np.abs(z_scores) > config.ANOMALY_THRESHOLD)

    features = [
        float(np.mean(patch_anomalies)),       # 0
        float(np.std(patch_anomalies)),         # 1
        float(np.max(patch_anomalies)),         # 2
        float(np.var(patch_sharpness)),         # 3
        float(np.var(patch_colors)),            # 4
        float(np.var(patch_anomalies)),         # 5
    ]

    # Feature 6: Semantic text consistency flag [NEW]
    if file_path is not None:
        features.append(float(_detect_gibberish_text(file_path)))
    else:
        features.append(0.0)

    # Feature 7: Semantic geometry flag [NEW]
    if file_path is not None:
        features.append(float(_detect_impossible_geometry(file_path)))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float64)


def _detect_gibberish_text(file_path: str) -> bool:
    """
    Detect gibberish/random text in image using EasyOCR.
    Natural food packaging has readable text; AI images may produce gibberish.

    Returns True if gibberish text is detected.
    """
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(file_path)

        if len(results) == 0:
            return False

        # Analyze entropy of detected text
        gibberish_count = 0
        for bbox, text, confidence in results:
            if confidence < 0.3:
                continue
            # High entropy + short text = likely gibberish
            if len(text) > 2 and len(text) < 30:
                entropy = _shannon_entropy(text)
                # Pure random ASCII string has high entropy
                ascii_ratio = sum(1 for c in text if c.isascii()) / len(text)
                consonant_ratio = sum(1 for c in text.lower() if c in "bcdfghjklmnpqrstvwxyz") / max(1, sum(1 for c in text if c.isalpha()))

                # Gibberish markers: very high entropy AND high consonant ratio with no vowels
                if entropy > 3.5 and consonant_ratio > 0.8 and sum(1 for c in text.lower() if c in "aeiou") == 0 and len(text) > 5:
                    gibberish_count += 1
                # Also check for repeated character patterns typical of AI gibberish
                elif len(set(text)) < 3 and len(text) > 6:
                    gibberish_count += 1

        return gibberish_count > 0

    except Exception:
        # EasyOCR not available or failed
        return False


def _detect_impossible_geometry(file_path: str) -> bool:
    """
    Detect impossible food geometry using YOLOv8 object detection.
    Checks for unrealistic proportions (objects too large/small).

    Returns True if impossible geometry is detected.
    """
    try:
        from ultralytics import YOLO

        # Use YOLOv8 nano for speed
        model = YOLO("yolov8n.pt")
        results = model(file_path, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return False

        boxes = results[0].boxes
        img_h, img_w = results[0].orig_shape

        # Check for unrealistic object proportions
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            obj_w = x2 - x1
            obj_h = y2 - y1
            area_ratio = (obj_w * obj_h) / (img_w * img_h)

            # An object occupying >80% of the image is suspicious in food context
            if area_ratio > 0.8:
                return True

            # Very thin objects spanning most of the image (impossible proportions)
            aspect = max(obj_w / img_w, obj_h / img_h)
            thinness = min(obj_w / img_w, obj_h / img_h)
            if aspect > 0.9 and thinness < 0.02:
                return True

        return False

    except Exception:
        # YOLOv8 not available or failed
        return False


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string."""
    if len(text) == 0:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    return -sum((f / length) * np.log2(f / length) for f in freq.values())
