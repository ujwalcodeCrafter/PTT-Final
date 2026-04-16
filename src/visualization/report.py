"""
Visualization module: Heatmaps, feature importance charts, and fraud reports.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Flask
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def generate_heatmap(proc_dict: dict, output_path: str = None) -> str:
    """
    Generate a region-level anomaly heatmap overlay on the original image.
    Returns path to saved heatmap image.

    Args:
        proc_dict: Preprocessed image dict from Stage 2.
        output_path: Optional custom output path. If None, uses default static/heatmap.png.

    Returns:
        Path to the saved heatmap PNG file.
    """
    if output_path is None:
        output_path = str(config.BASE_DIR / "static" / "heatmap.png")
    else:
        output_path = str(output_path)
    
    gray = proc_dict["gray"]
    rgb = proc_dict["rgb"]
    grid = config.PATCH_GRID_SIZE
    patch_h = 256 // grid
    patch_w = 256 // grid

    # Compute per-patch anomaly scores
    anomaly_map = np.zeros((256, 256))
    for row in range(grid):
        for col in range(grid):
            y1, y2 = row * patch_h, (row + 1) * patch_h
            x1, x2 = col * patch_w, (col + 1) * patch_w
            patch = gray[y1:y2, x1:x2]
            # Anomaly: z-score based deviation
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            deviation = abs(patch_mean - 0.5) + 0.5 * (1.0 - patch_std / 0.5)
            anomaly_map[y1:y2, x1:x2] = deviation

    # Normalize to [0, 1]
    if anomaly_map.max() > 0:
        anomaly_map = anomaly_map / anomaly_map.max()

    # Create heatmap overlay
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Display original image
    ax.imshow(rgb)

    # Overlay heatmap with transparency
    cmap = LinearSegmentedColormap.from_list(
        "fraud_heatmap",
        ["green", "yellow", "orange", "red"]
    )
    heatmap = ax.imshow(
        anomaly_map,
        cmap=cmap,
        alpha=0.5,
        vmin=0,
        vmax=1,
    )
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly Score", fontsize=8)

    ax.axis("off")
    plt.tight_layout()

    # Save using the provided or default output_path
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def generate_feature_importance_plot(importances: list, feature_names: list,
                                     top_k: int = 15) -> str:
    """
    Generate a horizontal bar chart of the top K most important features.
    Returns path to saved plot image.
    """
    # Select top K features
    indices = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[i] for i in indices]
    top_values = [importances[i] for i in indices]

    fig, ax = plt.subplots(1, 1, figsize=(8, max(6, top_k * 0.4)))

    colors = ["#2ecc71" if i < len(top_names) // 2 else "#e74c3c"
              for i in range(len(top_names))]
    bars = ax.barh(range(len(top_names)), top_values[::-1], color=colors[::-1], alpha=0.8)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=10)
    ax.set_title("Top {} Most Important Features".format(top_k), fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    output_path = config.BASE_DIR / "static" / "feature_importance.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def generate_fraud_report(result: dict, feature_vector: np.ndarray = None) -> dict:
    """
    Generate an interpretable fraud explanation report.

    Args:
        result: Output from classifier.predict()
        feature_vector: Optional raw feature vector for detailed breakdown.

    Returns:
        report dict with formatted results.
    """
    prob = result["fraud_probability"]
    risk = result["risk_level"]
    is_fraud = result["is_fraud"]

    # Color based on risk
    risk_colors = {
        "low": "#2ecc71",
        "medium": "#f39c12",
        "high": "#e74c3c",
    }

    # Top suspicious features (highest contribution for fraud)
    contributions = result.get("feature_contributions", {})
    sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_suspects = sorted_features[:5]

    report = {
        "verdict": "FRAUD DETECTED" if is_fraud else "LIKELY GENUINE",
        "risk_level": risk.upper(),
        "risk_color": risk_colors.get(risk, "#3498db"),
        "fraud_probability": round(prob * 100, 1),
        "classification": "Fraud" if is_fraud else "Genuine",
        "threshold": round(result.get("threshold", 0.52), 3),
        "top_suspicious_features": [
            {"name": name, "contribution": round(abs(val), 6)}
            for name, val in top_suspects
        ],
    }

    # Detailed feature breakdown by stage
    if feature_vector is not None:
        report["stage_breakdown"] = {
            "metadata_features": [
                {"name": fn, "value": round(float(v), 4)}
                for fn, v in zip(result["feature_names"][:4], feature_vector[:4])
            ],
            "spatial_features": [
                {"name": fn, "value": round(float(v), 4)}
                for fn, v in zip(result["feature_names"][4:22], feature_vector[4:22])
            ],
            "frequency_features": [
                {"name": fn, "value": round(float(v), 4)}
                for fn, v in zip(result["feature_names"][22:32], feature_vector[22:32])
            ],
            "region_features": [
                {"name": fn, "value": round(float(v), 4)}
                for fn, v in zip(result["feature_names"][32:40], feature_vector[32:40])
            ],
        }

    return report
