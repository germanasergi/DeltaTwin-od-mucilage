import yaml
from dotenv import load_dotenv
from urllib.parse import urlparse
import os
import gc
import random
import shutil
import glob
from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
import xarray as xr
import boto3

from tqdm import tqdm
from eopf.common.constants import OpeningMode
from eopf.common.file_utils import AnyPath
from eopf.store.convert import convert

from scipy.ndimage import generate_binary_structure, label, binary_fill_holes, binary_erosion
from pyproj import Transformer


######################## ENVIRONMENT SETUP ##########################################################

load_dotenv()

def load_config(config_path="cfg/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_environment(config):
    """Set up environment variables and directories for the dataset"""
    # Keep these from environment variables
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

    # Get other parameters from config
    ENDPOINT_URL = config['endpoint_url']
    ENDPOINT_STAC = config['endpoint_stac']
    BUCKET_NAME = config['bucket_name']
    DATASET_VERSION = config['dataset_version']
    BASE_DIR = config['base_dir']
    DATASET_DIR = f"{BASE_DIR}/{DATASET_VERSION}"
    BANDS = config['bands']

    # Setup connector
    connector = S3Connector(
        endpoint_url=ENDPOINT_URL,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region_name='default')

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()
    bucket = s3.Bucket(BUCKET_NAME)

    return {
        'ACCESS_KEY_ID': ACCESS_KEY_ID,
        'SECRET_ACCESS_KEY': SECRET_ACCESS_KEY,
        'ENDPOINT_URL': ENDPOINT_URL,
        'ENDPOINT_STAC': ENDPOINT_STAC,
        'BUCKET_NAME': BUCKET_NAME,
        'DATASET_VERSION': DATASET_VERSION,
        'BASE_DIR': BASE_DIR,
        'DATASET_DIR': DATASET_DIR,
        'BANDS': BANDS,
        's3': s3,
        's3_client': s3_client,
        'bucket': bucket
    }


def prepare_paths(path_dir):
    """
    Prepare paths for input and output datasets from CSV files.

    Args:
        path_dir (str): Directory containing input and target CSV files.

    Returns:
        DataFrame, DataFrame: Two DataFrames for input and output datasets.
    """
    df_input = pd.read_csv(f"{path_dir}/input.csv")
    df_output = pd.read_csv(f"{path_dir}/target.csv")

    df_input["path"] = df_input["Name"].apply(
        lambda x: os.path.join(path_dir, "input", os.path.basename(x).replace(".SAFE", ""))
    )
    df_output["path"] = df_output["Name"].apply(
        lambda x: os.path.join(path_dir, "target", os.path.basename(x).replace(".SAFE", ""))
    )

    return df_input, df_output


###################### S3 CONNECTOR ##############################################################

class S3Connector:
    """A clean connector for S3-compatible storage services"""

    def __init__(self, endpoint_url, access_key_id,
                 secret_access_key, region_name='default'):
        """Initialize the S3Connector with connection parameters"""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name

        # Create session
        self.session = boto3.session.Session()

        # Initialize S3 resource
        self.s3 = self.session.resource(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

        # Initialize S3 client
        self.s3_client = self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def get_s3_client(self):
        """Get the boto3 S3 client"""
        return self.s3_client

    def get_s3_resource(self):
        """Get the boto3 S3 resource"""
        return self.s3

    def get_bucket(self, bucket_name):
        """Get a specific bucket by name"""
        return self.s3.Bucket(bucket_name)

    def list_buckets(self):
        """List all available buckets"""
        response = self.s3_client.list_buckets()
        if 'Buckets' in response:
            return [bucket['Name'] for bucket in response['Buckets']]
        return []


###################### ACCESS DATA ##############################################################

def download_sentinel_data(df_output, base_dir, access_key, secret_key, endpoint_url):
    """Download Sentinel data from S3 to local directories"""

    output_dir = os.path.join(base_dir, "target")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Created local directory: {output_dir}")

    S3_CONFIG = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "region_name": "default"
        }
    }

    target_store_config = dict(mode=OpeningMode.CREATE_OVERWRITE)

    logger.info("Starting target file downloads...")
    for _, row in tqdm(df_output.iterrows(), total=len(df_output), desc="Target files"):
        try:
            product_url_base = row['S3Path']
            product_url = "s3://" + product_url_base.lstrip("/")
            zarr_filename = os.path.basename(product_url).replace('.SAFE', '.zarr')
            zarr_path = os.path.join(output_dir, zarr_filename)

            # Skip if file already exists
            if os.path.exists(zarr_path):
                logger.info(f"Skipping download, .zarr already exists: {zarr_path}")
                continue
            
            logger.info(f"Downloading target: {product_url} -> {zarr_path}")
            convert(AnyPath(product_url, **S3_CONFIG), zarr_path, target_store_kwargs=target_store_config)
        except Exception as e:
            logger.error(f"Error downloading {product_url if product_url else 'unknown URL'}: {e}")
        finally:
            # Cleanup SAFE temp dirs left in /tmp
            tmp_safes = glob.glob("/tmp/*.SAFE")
            for safe_dir in tmp_safes:
                try:
                    shutil.rmtree(safe_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temporary SAFE: {safe_dir}")
                except Exception as ce:
                    logger.warning(f"Could not clean temp SAFE {safe_dir}: {ce}")

def write_zarr_from_safe(safe_path, repo_path):
    """
    Convert a Sentinel-2 SAFE folder to a Zarr store inside an Icechunk repository.
    """
    # Step 1: Convert SAFE -> Zarr (classic Zarr store)
    zarr_store_path = Path(repo_path) / f"{Path(safe_path).stem}.zarr"
    convert_safe_to_zarr(safe_path, zarr_store_path)  # you already had this logic

    # Step 2: Add Zarr store to Icechunk repo
    repo = Repository(repo_path)
    repo.add_dataset(zarr_store_path)
    repo.commit("Added SAFE -> Zarr dataset")



####################### PATCHES ##################################################################

def clean_water_mask(ds, target_res="r10m"):
    """
    Fix water mask by:
    1. Keeping only the sea (remove lakes/rivers).
    2. Filling cloud holes in the sea.
    """
    scl = resample_band(ds, 'scl', target_res=target_res, ref='b04')
    scl = scl.squeeze().values
    raw_water_mask = (scl == 6)
    H, W = raw_water_mask.shape

    # Keep only the largest connected component that touches border (most prob the sea)
    st = generate_binary_structure(2, 2)   # 8-connectivity
    lab, nlab = label(raw_water_mask, structure=st) # different label for each connected water body
    
    if nlab == 0:
        return np.zeros_like(raw_water_mask, dtype=bool)

    # Find component sizes
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background, we don't consider it

    # Keep the largest component
    largest_label = sizes.argmax()
    sea_only = (lab == largest_label)

    # Fill holes inside sea (caused by clouds)
    sea_filled = binary_fill_holes(sea_only)

    return sea_filled.astype(bool)

def resample_band(ds, band, target_res="r10m", ref="b04", crs="EPSG:32632"):
    """
    Resample any band (reflectance or classification) to target resolution.
    """
    ref_band = ds[f"measurements/reflectance/{target_res}/{ref}"].rio.write_crs(crs) # Reference band at target resolution

    if band == "scl":
        band_da = ds[f"conditions/mask/l2a_classification/r20m/{band}"].rio.write_crs(crs)
        source_res = "r20m"
    else:
        # Detect which reflectance resolution contains the band
        source_res = next(
        (r for r in ["r10m", "r20m", "r60m"] if band in ds[f"measurements/reflectance/{r}"]),
        None
        )
        if source_res is None:
            raise ValueError(f"Band {band} not found in reflectance or scl folder")
        band_da = ds[f"measurements/reflectance/{source_res}/{band}"].rio.write_crs(crs)
    # If source == target, no resampling needed
    if source_res == target_res:
        return band_da
    
    return band_da.rio.reproject_match(ref_band)
    

def build_stack(ds, bands, target_res="r10m", ref_band="b04", crs="EPSG:32632"):
    """
    Build a lazy dask-backed (H, W, C) stack from bands, resampling as needed.

    Args:
        ds: xarray Dataset or DataTree
        bands: list of band names to include
        target_res: desired output resolution for all bands
        ref_band: reference band for resampling (default: 'b04' red)
        crs: CRS to assign if missing

    Returns:
        xarray.DataArray with dimensions (y, x, band)
    """
    stack = []

    for b in bands:
        if b in ds['measurements/reflectance/r10m'] or \
           b in ds['measurements/reflectance/r20m'] or \
           b in ds['measurements/reflectance/r60m']:
            arr = resample_band(ds, b, target_res=target_res, ref=ref_band, crs=crs) / 10000.0
        else:
            raise ValueError(f"Band {b} not found or not supported.")

        # Expand dims for stacking
        arr = arr.expand_dims(band=[b])
        stack.append(arr)

    # Concatenate all bands along 'band' dimension
    stacked = xr.concat(stack, dim="band").transpose("y", "x", "band")
    return stacked



def create_patches_dataframe(zarr_files, bands, bbox, target_res, stride, patch_size=256):
    """
    For each zarr file, extract top-left coordinates of sampled patches,
    and store them along with zarr path in a DataFrame.
    
    Returns:
        df_patches: DataFrame with columns ['zarr_path', 'x', 'y']
    """
    records = []

    for zf in zarr_files:
        print(f"Processing {zf} for patch coordinates...")
        ds = xr.open_datatree(zf, engine="zarr", mask_and_scale=False, chunks={})
        stack = build_stack(ds, bands,  target_res=target_res, ref_band="b04")

        # Compute water mask
        water_mask = clean_water_mask(ds, target_res=target_res)
        buffered_mask = water_mask & ~binary_erosion(water_mask, iterations=300)
        valid_mask = buffered_mask.astype(bool)
        
        # Retrieve shape bands
        band = ds['measurements/reflectance/r10m/b04']
        crs_utm = "EPSG:32633"
        transform = band.rio.transform()
        x_coords = band['x'].values
        y_coords = band['y'].values

        # Convert bbox (EPSG:4326) → tile CRS
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)

            # Convert from map coords → pixel indices
            def world_to_pixel(x, y, transform):
                col = int((x - transform.c) / transform.a)
                row = int((transform.f - y) / -transform.e)
                return col, row

            x0, y1 = world_to_pixel(x_min, y_min, transform)  # bottom-left
            x1, y0 = world_to_pixel(x_max, y_max, transform)  # top-right

            # Ensure ordering (y0 < y1, x0 < x1)
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])

            # Clip indices to valid range
            H, W = valid_mask.shape
            x0, x1 = np.clip([x0, x1], 0, W)
            y0, y1 = np.clip([y0, y1], 0, H)

            valid_mask = valid_mask[y0:y1, x0:x1]
            stack = stack.isel(x=slice(x0, x1), y=slice(y0, y1))

        # Convert to numpy for patch extraction
        stack_np = stack.transpose('y', 'x', 'band').values
        H, W, _ = stack_np.shape

        print(f"AOI shape for {zf}: {H}x{W} pixels")

        all_patches = []
        count = 0

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch_mask = valid_mask[y:y+patch_size, x:x+patch_size]

                # # Skip patches mostly outside water
                # if patch_mask.mean() < 0.99:
                #     continue

                patch = stack_np[y:y+patch_size, x:x+patch_size, :]
                if np.isnan(patch).any() or np.isinf(patch).any():
                    continue
                all_patches.append(patch)

                # zarr_name = os.path.basename(zf).replace(".zarr", "")
                # patch_name = f"{zarr_name}_y{y}_x{x}.npy"
                # patch_path = os.path.join(output_dir, patch_name)
                # records.append({
                #     "zarr_path": zf,
                #     "patch_path": patch_path,
                #     "y": y,
                #     "x": x
                # })
                count += 1
    
    X = np.stack(all_patches, axis=0)
    df_patches = pd.DataFrame(records)
    print(f"Total patches collected: {X.shape[0]}")

    return X

