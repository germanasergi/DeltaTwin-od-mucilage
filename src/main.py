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
from sklearn.model_selection import train_test_split

from utils.utils import *
from utils.cdse_utils import *
from utils.torch import define_model, load_model_weights
from utils.plot import visualize_patch_prediction, visualize_patches_on_tile

def main():

    # Setup
    args = parser.parse_args()
    config_path = os.path.join('/home/ubuntu/mucilage_pipeline/DeltaTwin/src/config.yaml')
    config = load_config(config_path=config_path)
    env = setup_environment(config)

# Create Dataset
    # Parameters from dataset config
    logger.info("Starting dataset creation...")
    query_config = config['query']
    bbox = query_config['bbox']
    bands = config['bands']
    start_date = datetime.strptime(query_config['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(query_config['end_date'], '%Y-%m-%d')
    mid_date = start_date + (end_date - start_date) / 2
    max_items = query_config['max_items']
    max_cloud_cover = query_config['max_cloud_cover']

    all_l1c_results, all_l2a_results = query_sentinel_data(
        bbox, start_date, end_date, max_items, max_cloud_cover
    )

    # Process and align data
    _, df_l2a = queries_curation(all_l1c_results, all_l2a_results)
    df_l2a.to_csv(f"{env['DATASET_DIR']}/output_l2a.csv")

    logger.info("Starting download process...")
    download_sentinel_data(
        df_l2a,
        env['DATASET_DIR'],
        env['ACCESS_KEY_ID'],
        env['SECRET_ACCESS_KEY'],
        env['ENDPOINT_URL']
    )

    logger.success("All downloads completed.")


# Patchify
    logger.info("Extracting patch coordinates...")
    zarr_dir = os.path.join(env['DATASET_DIR'], "target")
    zarr_files = glob.glob(os.path.join(zarr_dir, "*.zarr"))

    if not zarr_files:
        logger.warning(f"No Zarr files found in {zarr_dir}")
        return

    patches, df_coords = create_patches_dataframe(
        zarr_files,
        bands=bands,
        bbox=bbox,
        target_res='r10m',
        stride=256,
        patch_size=config['download'].get('patch_size', 256),
        date=mid_date,
        pat=env['PAT']
    )
    np.save(os.path.join(env['DATASET_DIR'], 'patches.npy'), patches)
    logger.success("Patch extraction completed.")

    # visualize_patches_on_tile(
    #     zarr_file=zarr_files[0],
    #     patches_coords=df_coords[df_coords["zarr_file"] == zarr_files[0]],
    #     patch_size=256,
    #     bbox=config["query"]["bbox"],
    #     save_dir="results"
    # )

# # Segmentation
#     # Parameters from model config

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load model checkpoints
#     ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

#     # Load config
#     config = ckpt['config']
#     mean = ckpt['mean']
#     std = ckpt['std']

#     # Load model
#     model = load_model_weights(config, ckpt, device)
#     for i, patch in enumerate(patches):
#         patch = (patch - mean) / (std + 1e-8)
#         patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float().to(device)

#         model.eval()
#         with torch.no_grad():
#             logits = model(patch_tensor)
#             if logits.shape[1] == 1:
#                 probs = torch.sigmoid(logits).squeeze().cpu().numpy()
#             else:
#                 probs = torch.softmax(logits, dim=1)[:, 1, :, :].squeeze().cpu().numpy()

#             pred = (probs > 0.5).astype(np.uint8)

#         visualize_patch_prediction(patch, probs, pred, save_dir="results", patch_id=f"patch_{i}")

            # TO SEE :  WHAT KIND OF OUTPUT DO WE WANT?



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single patch")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth")
    main()