"""
This module is used to extract IPUMS Mozambique dataset.
"""
import os
import pickle
from pathlib import Path
from ipumspy import IpumsApiClient, MicrodataExtract, readers#, ddi
import pandas as pd


def load_ipums_from_pkl(filepath:Path)->pd.DataFrame:
    """Load IPUMS dataset from pickle file (`filepath`)."""
    with open(filepath, 'rb') as f:
        ipums_df = pickle.load(f)

    return ipums_df


def get_ipums_data(collection:str, description:str, samples:list,
                   variables:list, api_key:str, download_dir:Path,
                   pkl_export:bool=True, pkl_path:Path=None) -> pd.DataFrame:
    """
    Create IPUMS dataset extract, download it locally, and if requested, export
    dataframe to pickle file (`pkl_export` and `pkl_path`).

    Args:
        collection (str):
            Name of an IPUMS microdata collection.
        description (str):
            Short description for extract.
        samples (list):
            List of sample IDs from an IPUMS microdata collection.
        variables (list):
            List of variable names from an IPUMS microdata collection.
        api_key (str):
            API key for registered IPUMS user.
        download_dir (pathlib.Path):
            Path to directory for data download.
        pkl_export (bool, default=True):
            Boolean for determining export to pickle.
        pkl_path (pathlib.Path, default=None):
            Path to pickle file for export.
    
    Returns:
        ipums_df (pd.DataFrame):
            Pandas DataFrame containing IPUMS dataset extract.
    """

    # Check Pickle Export Parameters
    if pkl_export and pkl_path is None:
        raise ValueError(
            "Error: pkl_export is True, pkl_path is not specified.")

    # Verify Download Directory
    os.makedirs(download_dir, exist_ok=True)

    # Get IPUMS Client
    ipums = IpumsApiClient(api_key)

    # Build Extract
    extract = MicrodataExtract(
        collection=collection,
        description=description,
        samples=samples,
        variables=variables
    )

    # Submit Extract
    ipums.submit_extract(extract)
    print(f"Extract submitted to IPUMS. Extract ID: {extract.extract_id}.")

    # Wait for Extract
    print("Waiting for extract to finish processing on IPUMS server...")
    ipums.wait_for_extract(extract)

    # Download Extract
    print(f"Downloading extract to {download_dir} ...")
    ipums.download_extract(extract, download_dir=download_dir)

    # Read Data Dictionary
    ddi_file = os.path.join(download_dir,
                            f"ipumsi_{extract.extract_id:05d}.xml")
    ddi = readers.read_ipums_ddi(ddi_file)

    # Extract Data from Dictionary
    print("Extracting data from extract to DataFrame...")
    ipums_df = readers.read_microdata(
        ddi, download_dir / ddi.file_description.filename
    )
    print(f"Shape of IPUMS Data Extract: {ipums_df.shape}")

    # Save to PKL (if specified)
    if pkl_export:
        print(f"Saving IPUMS DataFrame to {pkl_path} ...")
        with open(pkl_path, 'wb') as f:
            pickle.dump(ipums_df, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    # Report Completion
    print("IPUMS dataset extraction complete.")

    return ipums_df
