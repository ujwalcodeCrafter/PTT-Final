"""
STAGE 1: Metadata Extraction (4 features)

Extract EXIF and compression metadata from food images.
AI-generated and edited images typically lack natural camera metadata
or show signs of software manipulation.

Features (4):
  0. exif_camera_present    — 1.0 if camera EXIF exists, 0.0 otherwise
  1. timestamp_consistency  — 1.0 if EXIF timestamps are self-consistent, 0.5 if partial, 0.0 if conflicting
  2. editing_software       — 1.0 if editing software detected in EXIF, 0.0 otherwise
  3.jpeg_quality_estimate   — Estimated JPEG compression quality (0-1), 0.5 for non-JPEG
"""

import os
import exifread
import numpy as np
from PIL import Image
from io import BytesIO


def extract_stage1_features(file_path: str) -> np.ndarray:
    """
    Extract 4 metadata features from an image file.

    Args:
        file_path: Path to the image file.

    Returns:
        numpy array of shape (4,) with metadata features.
    """
    features = _extract_exif_features(file_path)
    features.append(_estimate_jpeg_quality(file_path))
    return np.array(features, dtype=np.float64)


def _extract_exif_features(file_path: str) -> list:
    """Extract EXIF-based features: camera, timestamp, editing software."""
    exif_camera_pres = 0.0
    timestamp_consist = 0.0
    editing_soft = 0.0

    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        # Feature 0: Camera presence
        camera_tags = [
            "Image Make", "Image Model", "EXIF LensModel",
            "Image SerialNumber", "Image BodySerialNumber"
        ]
        if any(key in tags for key in camera_tags):
            exif_camera_pres = 1.0

        # Feature 1: Timestamp consistency
        ts_tags = ["Image DateTime", "EXIF DateTimeOriginal",
                   "EXIF DateTimeDigitized", "Image DateTime"]
        found_ts = {key: str(tags[key]) for key in ts_tags if key in tags}

        if len(found_ts) == 0:
            timestamp_consist = 0.0  # No timestamps at all (common in AI images)
        elif len(found_ts) == 1:
            timestamp_consist = 0.7  # Single timestamp is somewhat suspicious
        else:
            # Check if all timestamps are in a reasonable range (within 60 seconds of each other)
            values = list(found_ts.values())
            try:
                from datetime import datetime
                dt_objects = [datetime.strptime(v[:19].replace(":", "", 2) + v[19:], "%Y%m%d %H%M%S")
                              for v in values]
                time_spread = (max(dt_objects) - min(dt_objects)).total_seconds()
                if time_spread < 60:
                    timestamp_consist = 1.0  # All consistent
                elif time_spread < 3600:
                    timestamp_consist = 0.7
                else:
                    timestamp_consist = 0.3  # Suspicious spread
            except (ValueError, IndexError):
                timestamp_consist = 0.5  # Parseable but can't compare

        # Feature 2a: Editing software detection
        editing_keywords = [
            "Image Software", "EXIF Software", "Image Artist",
            "Image HostComputer", "EXIF Software"
        ]
        editing_signatures = ["photoshop", "gimp", "fireworks", "paint",
                              "firefly", "dall-e", "midjourney", "stable diffusion",
                              "generative", "ai ", "synthogram", "deepai"]

        for tag_key in editing_keywords:
            if tag_key in tags:
                soft_val = str(tags[tag_key]).lower()
                for sig in editing_signatures:
                    if sig in soft_val:
                        editing_soft = 1.0
                        break

    except Exception:
        # EXIF read failed – image is likely AI-generated or corrupted
        pass

    return [exif_camera_pres, timestamp_consist, editing_soft]


def _estimate_jpeg_quality(file_path: str) -> float:
    """
    Estimate JPEG compression quality using quantization tables.
    Returns a value in [0, 1] where higher = less compressed (better quality).
    Returns 0.5 for non-JPEG images.
    """
    try:
        img = Image.open(file_path)
        fmt = img.format

        if fmt and "JPEG" not in fmt.upper():
            return 0.5

        # Check for quantization tables
        qtables = img.quantization
        if qtables is not None and len(qtables) > 0:
            qt = np.array(qtables[0]).flatten()
            if len(qt) == 64:
                # Quality estimation from quantization table
                q_est = min(100, max(1, int(4 * np.sum(100.0 / qt) / len(qt))))
                return q_est / 100.0

        # Try exif-based quality from EXIF
        try:
            with open(file_path, "rb") as f:
                tags = exifread.process_file(f, details=False)
            if "EXIF Compression" in tags:
                comp = str(tags["EXIF Compression"])
                if "6" in comp:  # JPEG compression
                    return 0.7  # Default moderate quality
        except Exception:
            pass

        return 0.5  # Unknown — assume moderate

    except Exception:
        return 0.5
