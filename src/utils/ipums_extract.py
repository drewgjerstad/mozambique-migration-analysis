"""
This module is used to extract IPUMS Mozambique dataset.
"""
import os
import pickle
from pathlib import Path
from ipumspy import IpumsApiClient, MicrodataExtract, readers#, ddi
import pandas as pd

from src.utils.preprocessing import (
    update_df_labels,
    transform_df,
    process_mig_response,
    remove_metadata_columns,
    remove_detailed_columns,
    standardize_binary_vars,
    binarize_categorical_vars,
    bin_continuous_vars,
)


def load_ipums_from_pkl(filepath:Path)->pd.DataFrame:
    """Load IPUMS dataset from pickle file (`filepath`)."""
    with open(filepath, 'rb') as f:
        mig1_df, mig5_df = pickle.load(f)

    return mig1_df, mig5_df


def get_ipums_data(collection:str, description:str, samples:list,
                   variables:list, api_key:str, download_dir:Path,
                   pkl_export:bool=True, pkl_path:Path=None) -> tuple:
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
            "Error: pkl_export is True, pkl_path is not specified."
        )

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
    print("Waiting for extract to finish processing on IPUMS server...", end="")
    ipums.wait_for_extract(extract)
    print(" [complete]")

    # Download Extract
    print(f"Downloading extract to {download_dir} ...", end="")
    ipums.download_extract(extract, download_dir=download_dir)
    print(" [complete]")

    # Read Data Dictionary
    ddi_file = os.path.join(download_dir,
                            f"ipumsi_{extract.extract_id:05d}.xml")
    ddi = readers.read_ipums_ddi(ddi_file)

    # Extract Data from Dictionary
    print("Extracting data from extract to DataFrame...", end="")
    ipums_df = readers.read_microdata(
        ddi, download_dir / ddi.file_description.filename
    )
    print(" [complete]")

    # Replace categorical columns with labels
    print("Updating DataFrame with labels...", end="")
    ipums_df = update_df_labels(ipums_df, ddi)
    print(" [complete]")

    # Transform to fix NIU, unknown values (and other issues)
    print("Transforming data to fix NIU/unknown values, other issues...", end="")
    ipums_df = transform_df(ipums_df)
    print(" [complete]")

    # Process Migration Response
    print("Processing migration response variables (MIGRATE1, MIGRATE5)...", end="")
    mig1_data, mig5_data = process_mig_response(ipums_df)
    print(" [complete]")

    # Drop Metadata Columns
    print("Removing metadata columns unnecessary for analyses...", end="")
    mig1_data, mig5_data = remove_metadata_columns(mig1_data, mig5_data)
    print(" [complete]")

    # Drop Detailed Columns
    print("Removing detailed columns unnecessary for analyses...", end="")
    mig1_data, mig5_data = remove_detailed_columns(mig1_data, mig5_data)
    print(" [complete]")

    # Standardizing Binary Variables
    print("Standardizing binary variables...", end="")
    mig1_data, mig5_data = standardize_binary_vars(mig1_data, mig5_data)
    print(" [complete]")

    # Binarizing Categorical Variables
    print("Binarizing categorical variables...", end="")
    mig1_data, mig5_data = binarize_categorical_vars(mig1_data, mig5_data)
    print(" [complete]")

    # Binning Continuous Variables
    print("Binning continuous variables...", end="")
    mig1_data, mig5_data = bin_continuous_vars(mig1_data, mig5_data)
    print(" [complete]")

    # Save to PKL (if specified)
    if pkl_export:
        print(f"Saving IPUMS DataFrame to {pkl_path} ...", end="")
        with open(pkl_path, 'wb') as f:
            pickle.dump((mig1_data, mig5_data), f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print(" [complete]")

    # Report Completion
    print("\n**** IPUMS dataset extraction and processing complete. ****")

    return mig1_data, mig5_data
