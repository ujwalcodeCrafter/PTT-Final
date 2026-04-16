"""
STAGE 3: Forensic Spatial Features (18 features)

Extract spatial-domain forensic features including texture, noise,
edge-based, statistical, and cross-channel noise decorrelation.

Features (18):
  0.  lbp_uniform_ratio        — Ratio of uniform LBP patterns (texture regularity)
  1.  noise_std_R              — Local noise variance in Red channel
  2.  noise_std_G              — Local noise variance in Green channel
  3.  noise_std_B              — Local noise variance in Blue channel
  4.  chroma_lum_noise_ratio   — Chrominance/luminance noise ratio [NEW]
  5.  laplacian_edge_energy    — Total Laplacian edge energy
  6.  edge_coherence_ratio     — Ratio of coherent vs incoherent edges
  7-8.  rgb_mean[3]            — Mean per RGB channel
  10-11. rgb_std[3]           — Std per RGB channel
  13-14. rgb_skew[3]          — Skewness per RGB channel
  16-17. ycc_mean[2]          — Mean of YCbCr (Cb, Cr only — Y overlaps with gray)
  19-20. ycc_std[2]           — Std of YCbCr (Cb, Cr only)
  22-23. cross_noise_RG        — Cross-channel noise correlation R↔G [NEW]
  24-25. cross_noise_GB        — Cross-channel noise correlation G↔B [NEW]

Wait, let me recount. The actual return vector below has 18 features.
"""

import numpy as np
from scipy import stats
import cv2
from skimage.feature import local_binary_pattern


def extract_stage3_features(proc_dict: dict) -> np.ndarray:
    """
    Extract 18 spatial forensic features from preprocessed image.

    Args:
        proc_dict: Output of stage2 preprocessing.

    Returns:
        numpy array of shape (18,).
    """
    rgb = proc_dict["rgb"]          # (256, 256, 3) float64 [0, 1]
    ycbcr = proc_dict["ycbcr"]      # (256, 256, 3)
    gray = proc_dict["gray"]        # (256, 256)

    features = []

    # Feature 0: LBP uniform pattern ratio
    features.append(_lbp_uniform_ratio(gray))

    # Features 1-3: Noise standard deviation in R, G, B channels
    r_std, g_std, b_std = _noise_std_per_channel(rgb)
    features.extend([r_std, g_std, b_std])

    # Feature 4: Chrominance/luminance noise ratio [NEW]
    features.append(_chroma_lum_noise_ratio(ycbcr))

    # Feature 5: Laplacian edge energy
    features.append(_laplacian_edge_energy(gray))

    # Feature 6: Edge coherence ratio
    features.append(_edge_coherence(gray))

    # Features 7-9: RGB mean per channel
    for c in range(3):
        features.append(float(np.mean(rgb[:, :, c])))

    # Features 10-12: RGB std per channel
    for c in range(3):
        features.append(float(np.std(rgb[:, :, c])))

    # Features 13-15: RGB skewness per channel
    for c in range(3):
        features.append(float(stats.skew(rgb[:, :, c].flatten())))

    # Features 16-17: YCbCr Cb, Cr mean (Y excluded — redundant with RGB mean)
    features.append(float(np.mean(ycbcr[:, :, 1])))  # Cb mean
    features.append(float(np.mean(ycbcr[:, :, 2])))  # Cr mean

    # Feature 17-18: YCbCr Cb, Cr std (Y excluded)
    features.append(float(np.std(ycbcr[:, :, 1])))   # Cb std
    features.append(float(np.std(ycbcr[:, :, 2])))   # Cr std

    # Feature 19-20: Cross-channel noise correlation
    rg_corr, gb_corr = _cross_channel_noise_correlation(rgb)
    features.append(float(rg_corr))
    features.append(float(gb_corr))

    return np.array(features, dtype=np.float64)


def _lbp_uniform_ratio(gray: np.ndarray, radius: int = 2, n_points: int = 16) -> float:
    """
    Compute ratio of uniform Local Binary Pattern patterns.
    AI images often have fewer uniform patterns (more texture irregularity).
    """
    # Scale to [0, 255] for skimage
    gray_255 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_255, n_points, radius, method="uniform")

    # Count uniform patterns (values 0 to n_points+1)
    total = lbp.size
    uniform_count = np.sum(lbp <= (n_points + 1))
    return float(uniform_count / total)


def _noise_std_per_channel(rgb: np.ndarray, win: int = 5) -> "tuple[float, float, float]":
    """
    Estimate noise standard deviation per channel using median filter residual.
    The residual (image - denoised) approximates the noise component.
    """
    results = []
    for c in range(3):
        ch = rgb[:, :, c]
        # Median filter denoising
        denoised = cv2.medianBlur((ch * 255).astype(np.uint8), win).astype(np.float64) / 255.0
        residual = ch - denoised
        results.append(float(np.std(residual)))
    return tuple(results)


def _chroma_lum_noise_ratio(ycbcr: np.ndarray, win: int = 5) -> float:
    """
    Ratio of chrominance noise to luminance noise [NEW].
    Natural images have predictable chroma/luma noise ratios.
    AI generators often produce decorrelated chroma noise.
    """
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    # Compute noise residuals
    def noise_residual(channel: np.ndarray) -> np.ndarray:
        c_uint8 = (channel * 255).astype(np.uint8)
        denoised = cv2.medianBlur(c_uint8, win).astype(np.float64) / 255.0
        return channel - denoised

    y_noise = noise_residual(y)
    cb_noise = noise_residual(cb)
    cr_noise = noise_residual(cr)

    lum = np.std(y_noise)
    chroma = (np.std(cb_noise) + np.std(cr_noise)) / 2.0

    # Avoid division by zero
    if lum < 1e-10:
        return 0.0
    ratio = chroma / lum

    # Clip to reasonable range
    return float(np.clip(ratio, 0, 5))


def _laplacian_edge_energy(gray: np.ndarray) -> float:
    """Total energy of Laplacian operator (second-derivative edges)."""
    gray_8u = (gray * 255).astype(np.uint8)
    lap = cv2.Laplacian(gray_8u, cv2.CV_64F)
    return float(np.sum(np.abs(lap)) / lap.size)


def _edge_coherence(gray: np.ndarray) -> float:
    """
    Edge coherence ratio: coherent edges (Sobel X and Y both strong)
    vs incoherent edges (only one direction strong).
    AI images tend to have less coherent edge structures.
    """
    gray_8u = (gray * 255).astype(np.uint8)
    gx = cv2.Sobel(gray_8u, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_8u, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    threshold = np.mean(mag)

    strong_edges = mag > threshold
    if np.sum(strong_edges) == 0:
        return 0.0

    # Coherent edges: both gx and gy have significant magnitude
    coherent = (np.abs(gx) > threshold * 0.5) & (np.abs(gy) > threshold * 0.5)
    coherent_count = np.sum(coherent & strong_edges)

    return float(coherent_count / np.sum(strong_edges))


def _cross_channel_noise_correlation(rgb: np.ndarray, win: int = 5) -> "tuple[float, float]":
    """
    Cross-channel noise correlation for R↔G and G↔B [NEW].
    Natural captures have correlated noise across color channels.
    AI generators often produce decorrelated channel noise.
    """
    def noise_residual(channel):
        c_uint8 = (channel * 255).astype(np.uint8)
        denoised = cv2.medianBlur(c_uint8, win).astype(np.float64) / 255.0
        return (channel - denoised).flatten()

    r_noise = noise_residual(rgb[:, :, 0])
    g_noise = noise_residual(rgb[:, :, 1])
    b_noise = noise_residual(rgb[:, :, 2])

    rg_corr = float(np.corrcoef(r_noise, g_noise)[0, 1])
    gb_corr = float(np.corrcoef(g_noise, b_noise)[0, 1])

    # Handle NaN from constant noise
    rg_corr = 0.0 if np.isnan(rg_corr) else rg_corr
    gb_corr = 0.0 if np.isnan(gb_corr) else gb_corr

    return rg_corr, gb_corr
