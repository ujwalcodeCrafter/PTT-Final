"""
STAGE 4: Frequency Analysis (10 features)

Analyze frequency-domain characteristics using FFT and DCT.
AI-generated images exhibit characteristic spectral anomalies:
unnatural high-frequency content, regular grid patterns,
and DCT artifacts from implicit compression.

Features (10):
  0. fft_high_freq_ratio    — Ratio of high-frequency energy vs total energy
  1. spectral_entropy       — Entropy of the magnitude spectrum
  2. dct_coeff_std          — Standard deviation of DCT coefficients
  3. texture_smoothness     — DCT-based texture smoothness metric
  4. noise_residual_energy   — Energy of high-frequency noise residual
  5. freq_band_ratio_1       — Low-freq band energy ratio
  6. freq_band_ratio_2       — Mid-low-freq band energy ratio
  7. freq_band_ratio_3       — Mid-high-freq band energy ratio
  8. freq_band_ratio_4       — High-freq band energy ratio
  9. dct_double_comp_detect  — DCT double-compression detector [NEW]
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift


def extract_stage4_features(proc_dict: dict) -> np.ndarray:
    """
    Extract 10 frequency-domain features.

    Args:
        proc_dict: Output of stage2 preprocessing.

    Returns:
        numpy array of shape (10,).
    """
    gray = proc_dict["gray"]  # (256, 256) float64

    features = []

    # Feature 0: FFT high-frequency ratio
    features.append(_fft_high_freq_ratio(gray))

    # Feature 1: Spectral entropy
    features.append(_spectral_entropy(gray))

    # Features 2-4: DCT-based features
    dct = cv2.dct(gray.astype(np.float32))
    features.append(_dct_coeff_std(dct))
    features.append(_texture_smoothness(dct))
    features.append(_noise_residual_energy(gray))

    # Features 5-8: 4 frequency band energy ratios
    features.extend(_frequency_band_ratios(gray))

    # Feature 9: DCT double-compression detection
    features.append(_dct_double_compression(dct))

    return np.array(features, dtype=np.float64)


def _fft_high_freq_ratio(gray: np.ndarray, split: float = 0.7) -> float:
    """
    Ratio of high-frequency energy to total energy in FFT magnitude spectrum.
    AI images often have excess high-frequency energy from interpolation artifacts.
    """
    fft_magnitude = np.abs(fftshift(fft2(gray)))
    fft_log = np.log1p(fft_magnitude)

    h, w = fft_log.shape
    cy, cx = h // 2, w // 2
    total_energy = np.sum(fft_log)

    # Define high-frequency region as outer 30% of the spectrum
    mask = np.zeros_like(fft_log)
    for i in range(h):
        for j in range(w):
            dy = abs(i - cy) / cy
            dx = abs(j - cx) / cx
            if dy > split or dx > split:
                mask[i, j] = 1.0
    if total_energy == 0:
        return 0.0
    return float(np.sum(fft_log * mask) / total_energy)


def _spectral_entropy(gray: np.ndarray) -> float:
    """
    Shannon entropy of the FFT magnitude spectrum (normalized).
    Lower entropy indicates more regular structure (suspicious).
    """
    fft_mag = np.abs(fftshift(fft2(gray)))
    fft_log = np.log1p(fft_mag)

    # Normalize to probability distribution
    total = np.sum(fft_log)
    if total == 0:
        return 0.0
    p = fft_log / total

    # Shannon entropy
    p_flat = p.flatten()
    p_flat = p_flat[p_flat > 0]  # Avoid log(0)
    entropy = -np.sum(p_flat * np.log2(p_flat))

    # Normalize by maximum possible entropy
    h, w = fft_mag.shape
    max_entropy = np.log2(h * w)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _dct_coeff_std(dct: np.ndarray) -> float:
    """
    Standard deviation of DCT coefficients (excluding DC).
    Natural images follow a predictable DCT coefficient distribution.
    """
    return float(np.std(np.abs(dct)))


def _texture_smoothness(dct: np.ndarray) -> float:
    """
    DCT-based texture smoothness: ratio of low-frequency DCT energy
    to total DCT energy. Smooth textures have higher ratios.
    """
    dct_abs = np.abs(dct)
    h, w = dct_abs.shape
    total = np.sum(dct_abs)
    if total == 0:
        return 0.0

    # Low-frequency region: top-left quadrant (low spatial frequencies)
    low_freq = dct_abs[:h // 2, :w // 2]
    low_energy = np.sum(low_freq)
    return float(low_energy / total)


def _noise_residual_energy(gray: np.ndarray) -> float:
    """
    Energy of the high-frequency noise residual.
    Computed by subtracting a Gaussian-smoothed version from the image.
    """
    gray_8u = (gray * 255).astype(np.uint8)
    smoothed = cv2.GaussianBlur(gray_8u, (5, 5), 0).astype(np.float64) / 255.0
    residual = gray - smoothed
    return float(np.mean(residual ** 2))


def _frequency_band_ratios(gray: np.ndarray) -> "list[float]":
    """
    4 frequency band energy ratios computed from FFT.
    Divides the spectrum into 4 concentric bands.
    """
    fft_mag = np.abs(fftshift(fft2(gray)))
    fft_log = np.log1p(fft_mag)

    h, w = fft_log.shape
    cy, cx = h // 2, w // 2

    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_max = np.sqrt(cx ** 2 + cy ** 2)
    band_bounds = [0, 0.25, 0.50, 0.75, 1.0]

    band_energies = []
    for i in range(4):
        lower = band_bounds[i] * r_max
        upper = band_bounds[i + 1] * r_max
        mask = (dist >= lower) & (dist < upper)
        band_energy = np.sum(fft_log[mask])
        band_energies.append(band_energy)

    total = sum(band_energies)
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [float(e / total) for e in band_energies]


def _dct_double_compression(dct: np.ndarray) -> float:
    """
    DCT double-compression detection [NEW].
    Analyzes the histogram of DCT coefficient values in AC bands.
    Double-compressed images show periodic peaks at quantization step intervals.

    Returns a detector value: higher = more likely double-compressed.
    """
    # Look at AC coefficients in the mid-frequency band
    h, w = dct.shape
    ac = dct[1:h//4, 1:w//4].flatten()

    # Normalize to absolute values
    ac = np.abs(ac)

    # Build histogram
    hist, _ = np.histogram(ac, bins=64, range=(0, np.percentile(ac, 99)))
    hist = hist.astype(np.float64)
    hist = hist / (np.sum(hist) + 1e-10)

    # Detect periodicity using autocorrelation of histogram
    n = len(hist)
    acorr = np.correlate(hist - np.mean(hist), hist - np.mean(hist), mode="full")
    acorr = acorr[n:]  # Positive lags only

    if np.max(acorr[1:]) < 1e-10:
        return 0.0

    # Look for peaks at regular intervals
    peaks = []
    for lag in range(2, n // 4):
        if acorr[lag] > np.mean(acorr[1:n//4]) * 1.5:
            peaks.append(acorr[lag])

    score = float(np.sum(peaks) / (np.sum(np.abs(acorr[1:n//4])) + 1e-10))
    return float(np.clip(score, 0, 1))
