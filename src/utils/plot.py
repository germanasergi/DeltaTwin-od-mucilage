import os
import pandas as pd
import numpy as np
from requests import patch
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F



def get_rgb(patch):
    """
    Convert multi-band patch to RGB for visualization.
    bands: tuple with indices for (R, G, B).
    """
    rgb = patch[:, :, [3, 2, 1]] #now there is b1 as well
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    return rgb


def show_amei(patch, eps=1e-6, ax=None):
    green = patch[:, :, 2]
    red = patch[:,:,3]
    nir = patch[:,:,7]
    swir = patch[:,:,10]
    denom = green + 0.25 * swir
    amei  = (2*red + nir - 2*swir) / (denom + eps)

    p2, p98 = np.nanpercentile(amei, (2, 98))
    amei = np.clip((amei - p2) / (p98 - p2), 0, 1)
    cmap = plt.cm.turbo

    if ax is None:
        ax = plt.gca()
    ax.imshow(amei, cmap=cmap)
    ax.axis("off")
    return ax


def visualize_patch_prediction(patch, probs, pred, save_dir, patch_id="patch"):
    os.makedirs(save_dir, exist_ok=True)

    # Get RGB for visualization
    rgb = get_rgb(patch)  # typically b04,b03,b02 or your chosen combination
    mask = pred.astype(bool)

    # Overlay mask in red
    red_mask = np.zeros_like(rgb)
    red_mask[..., 0] = 1.0
    alpha = 0.5
    overlay = np.where(mask[..., None], (1 - alpha) * rgb + alpha * red_mask, rgb)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    axes[1].imshow(probs, cmap="viridis")
    axes[1].set_title("Predicted Probability")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red = Mucilage)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{patch_id}_prediction.png"), dpi=200)
    plt.close(fig)