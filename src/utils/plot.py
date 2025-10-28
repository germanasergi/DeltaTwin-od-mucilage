import os
import pandas as pd
import numpy as np
from requests import patch
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch.nn.functional as F
from pyproj import Transformer
import xarray as xr



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


def visualize_patches_on_tile(zarr_file, patches_coords, patch_size=256, bbox=None, target_res="r10m", save_dir=None):
    """
    Visualize the RGB tile with overlayed red rectangles representing patch positions.

    Args:
        zarr_file (str): Path to the Zarr file
        patches_coords (pd.DataFrame): Output of create_patches_dataframe (filtered for this zarr)
        patch_size (int): Size of each patch in pixels
        bbox (list): Optional [lon_min, lat_min, lon_max, lat_max]
        target_res (str): Resolution to read (usually 'r10m')
    """
    print(f"Visualizing patches on {zarr_file}...")

    # --- Load RGB bands ---
    ds = xr.open_datatree(zarr_file, engine="zarr", mask_and_scale=False, chunks={})
    b04 = ds[f"measurements/reflectance/{target_res}/b04"].values  # Red
    b03 = ds[f"measurements/reflectance/{target_res}/b03"].values  # Green
    b02 = ds[f"measurements/reflectance/{target_res}/b02"].values  # Blue
    rgb = np.stack([b04, b03, b02], axis=-1)
    
    # Normalize for display
    rgb = np.clip(rgb / np.nanpercentile(rgb, 99), 0, 1)
    H, W, _ = rgb.shape
    print(f"Tile shape: {H}x{W}")

    # --- Prepare the figure ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper")

    # --- Overlay patches ---
    for _, row in patches_coords.iterrows():
        rect = mpatches.Rectangle(
            (row["x_pix"], row["y_pix"]),  # pixel top-left corner
            patch_size,
            patch_size,
            linewidth=1,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"Patches overlay")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    # --- Optional: overlay bbox outline ---
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)

        # Convert world to pixel coords
        band = ds[f"measurements/reflectance/{target_res}/b04"]
        transform = band.rio.transform()

        def world_to_pixel(x, y, transform):
            col = (x - transform.c) / transform.a
            row = (transform.f - y) / -transform.e
            return col, row

        x0, y1 = world_to_pixel(x_min, y_min, transform)
        x1, y0 = world_to_pixel(x_max, y_max, transform)
        rect = mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(x0, y0 - 20, "BBox", color="yellow")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"patches_on_tile.png"), dpi=200)
    plt.close(fig)

def visualize_sst(patch, save_dir):
    sst = patch[:, :, -1]

    plt.figure(figsize=(5, 5))
    im = plt.imshow(sst, cmap="coolwarm", vmin=5, vmax=35)
    plt.colorbar(im, label="SST (°C)")
    plt.title(f"SST Band)")
    plt.savefig(os.path.join(save_dir, f"sst.png"), dpi=200)
    plt.close()


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