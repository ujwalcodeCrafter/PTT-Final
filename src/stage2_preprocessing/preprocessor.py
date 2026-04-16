"""
STAGE 2: Preprocessing

Resize, normalize, and convert color spaces for downstream analysis.
All images are converted to 256x256, pixel values normalized to [0, 1],
and both RGB and YCbCr color spaces are produced.
"""

import numpy as np
from PIL import Image
import cv2


def preprocess_image(image_array: np.ndarray) -> dict:
    """
    Preprocess an image for forensic analysis.

    Args:
        image_array: Raw image as numpy array (H, W, 3) in RGB.

    Returns:
        dict with keys:
            'resized': 256x256 RGB array, float64 [0, 1]
            'rgb': Same as resized, kept for clarity
            'ycbcr': YCbCr version (float64)
            'gray': Grayscale version (float64)
    """
    # Resize to 256x256 using bilinear interpolation
    img = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    elif img.dtype != np.float64:
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img /= 255.0

    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Convert to YCbCr
    ycbcr = _rgb_to_ycbcr(img)

    # Grayscale
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    return {
        "resized": img,
        "rgb": img.astype(np.float64),
        "ycbcr": ycbcr,
        "gray": gray,
    }


def load_and_preprocess(file_path: str) -> dict:
    """
    Load an image from disk and apply full preprocessing pipeline.

    Args:
        file_path: Path to image file.

    Returns:
        Preprocessed dict with 'resized', 'rgb', 'ycbcr', 'gray'.
    """
    img = Image.open(file_path).convert("RGB")
    img_array = np.array(img)
    return preprocess_image(img_array)


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB [0, 1] to YCbCr using ITU-R BT.601 coefficients.
    Y in [0, 1], Cb/Cr in [-0.5, 0.5].
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5643 * (b - y)
    cr = 0.7132 * (r - y)

    return np.stack([y, cb, cr], axis=-1)
