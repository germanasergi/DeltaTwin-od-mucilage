import glob
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import zipfile
#from sklearn.model_selection import train_test_split

from utils.utils import *
from utils.cdse_utils import *
from utils.torch import define_model, load_model_weights
from utils.plot import *

import gradio as gr



def run_pipeline(
    bbox,
    start_date,
    end_date,
    cdse_key,
    cdse_secret,
    earth_data_token
):
    """
    Wrapper around your pipeline for Gradio.
    Accepts input args as strings/values.
    Returns path to resulting ZIP file.
    """

    # ---------------------------
    # SETUP
    # ---------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # DeltaTwin output directory (safe) or local outputs/
    OUTPUT_DIR = os.getenv("DELTA_OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load config
    config_path = os.path.join(BASE_DIR, "cfg", "config.yaml")
    config = load_config(config_path=config_path)
    DATASET_DIR = os.path.join(BASE_DIR, config['dataset_version'])

    today = datetime.utcnow().date()

    # parse dates
    start_date = today if start_date.lower() == "today" else datetime.strptime(start_date, "%Y-%m-%d")
    end_date = (today + timedelta(days=1)) if end_date.lower() == "today" else datetime.strptime(end_date, "%Y-%m-%d")
    mid_date = start_date + (end_date - start_date) / 2

    # ---------------------------
    # QUERY SENTINEL
    # ---------------------------
    query_cfg = config['query']
    bands = config['bands']
    bbox = [float(x) for x in bbox.split(' ')] # adapted for gradio
    print(f"Querying data from {start_date} to {end_date} for bbox {bbox}...")

    l2a_results = query_sentinel_data(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_items=query_cfg['max_items'],
        max_cloud_cover=query_cfg['max_cloud_cover']
    )
    df_l2a = queries_curation(l2a_results)

    # ---------------------------
    # DOWNLOAD
    # ---------------------------
    download_sentinel_data(
        df_output=df_l2a,
        base_dir=DATASET_DIR,
        access_key=cdse_key,
        secret_key=cdse_secret,
        endpoint_url='https://eodata.dataspace.copernicus.eu'
    )

    # ---------------------------
    # PATCHIFY
    # ---------------------------
    zarr_dir = os.path.join(DATASET_DIR, "target")
    zarr_files = glob.glob(os.path.join(zarr_dir, "*.zarr"))
    if not zarr_files:
        raise RuntimeError("No ZARR files found after download!")

    patches_per_zarr, df_coords = create_patches_dataframe(
        zarr_files=zarr_files,
        bands=bands,
        bbox=bbox,
        target_res='r10m',
        stride=256,
        patch_size=config['download'].get('patch_size', 256),
        date=mid_date,
        pat=earth_data_token
    )

    df_coords = df_coords.rename(columns={"x_pix": "x", "y_pix": "y"})
    unique_zarrs = df_coords["zarr_file"].unique()

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(BASE_DIR, "weights/unet_checkpoint.pth"), map_location=device, weights_only=False)

    model_cfg = ckpt['config']
    mean = ckpt['mean']
    std = ckpt['std']
    model = load_model_weights(model_cfg, ckpt, device)

    # ---------------------------
    # SEGMENTATION + EXPORT
    # ---------------------------
    tif_paths = []

    for zarr_path in unique_zarrs:
        patches = patches_per_zarr[zarr_path]

        all_probs = []
        all_preds = []

        for patch in patches:
            patch_norm = (patch - mean) / (std + 1e-8)
            tensor = torch.from_numpy(patch_norm).permute(2,0,1).unsqueeze(0).float().to(device)

            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                if logits.shape[1] == 1:
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                else:
                    prob = torch.softmax(logits, dim=1)[:,1,:,:].squeeze().cpu().numpy()

                pred = (prob > 0.5).astype(np.uint8)

            all_probs.append(prob)
            all_preds.append(pred)

        avg_prob, binary_mask = stitch_predictions(
            zarr_file=zarr_path,
            df_coords=df_coords,
            probs_list=all_probs,
            preds_list=all_preds,
            patch_size=256
        )

        # Save TIF into OUTPUT_DIR
        tif_path = export_geotiff_and_vector(
            zarr_path=zarr_path,
            prob_map=avg_prob,
            binary_mask=binary_mask,
            confidence=None,
            amei=None,
            out_dir=OUTPUT_DIR
        )

        crop_tiff_to_bbox(tif_path, bbox, tif_path)
        tif_paths.append(tif_path)

    # ---------------------------
    # ZIP OUTPUT
    # ---------------------------
    zip_path = os.path.join(OUTPUT_DIR, "mucilage_masks.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for tif in tif_paths:
            zf.write(tif, os.path.basename(tif))

    return zip_path


demo = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(label="Bounding Box (lon_min, lat_min, lon_max, lat_max)"),
        gr.Textbox(label="Start Date (YYYY-MM-DD or 'today')"),
        gr.Textbox(label="End Date (YYYY-MM-DD or 'today')"),
        gr.Textbox(label="CDSE Access Key"),
        gr.Textbox(label="CDSE Secret Key"),
        gr.Textbox(label="Earth Data Hub Token")
    ],
    outputs=gr.File(label="Mucilage Mask ZIP"),
    title="Mucilage Segmentation Pipeline",
    description="Runs the end-to-end Sentinel-2 mucilage detector and returns a ZIP of prediction masks."
)

if __name__ == "__main__":
    demo.launch()