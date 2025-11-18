"""
This module is used during data extraction to handle initial preprocessing steps
required for the project.
"""
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

def update_df_labels(df:pd.DataFrame, ddi) -> pd.DataFrame:
    """Replace the categorical columns with proper labels."""

    # Define New DataFrame
    updated_df = pd.DataFrame()

    # Update Labels
    for var in df.columns:
        var_dict = ddi.get_variable_info(var).codes
        inv_var_dict = {value: key for key, value in var_dict.items()}

        if len(inv_var_dict) > 0:
            updated_df[var] = df[var].map(inv_var_dict)
        else:
            updated_df[var] = df[var]

    return updated_df


def transform_df(df:pd.DataFrame) -> pd.DataFrame:
    """Transform to fix NIU, unknown values (and other issues)."""

    # Get Variable Dictionaries
    var_dict_path = Path(__file__).parent / 'var_dictionaries.pkl'
    with open(var_dict_path, 'rb') as f:
        var_dicts = pickle.load(f)

    # Transform DF
    for v in var_dicts.keys():
        if v in df.columns:
            df[v] = df[v].map(lambda x: var_dicts[v].get(x, x))

    return df


def process_mig_response(df:pd.DataFrame) -> tuple:
    """Process migration response variables (MIGRATE1, MIGRATE5)."""

    # Remove NIU and Unknown values
    mig1_data = df[~df['MIGRATE1'].isna()].copy()
    mig5_data = df[~df['MIGRATE5'].isna()].copy()

    # Drop other response column from respective sets
    mig1_data = mig1_data.drop(['MIGRATE5'], axis=1, inplace=False)
    mig5_data = mig5_data.drop(['MIGRATE1'], axis=1, inplace=False)

    return mig1_data, mig5_data


def remove_metadata_columns(mig1_df, mig5_df) -> tuple:
    """Remove metadata columns from sets."""
    metadata_cols = ['COUNTRY', 'SAMPLE', 'SERIAL', 'HHWT', 'PERNUM', 'PERWT']
    mig1_df = mig1_df.drop(columns=metadata_cols, inplace=False)
    mig5_df = mig5_df.drop(columns=metadata_cols, inplace=False)

    return mig1_df, mig5_df


def remove_detailed_columns(mig1_df, mig5_df) -> tuple:
    """Remove detailed columns from sets."""
    detailed_cols = ['OWNERSHIPD', 'MARSTD', 'EDATTAIND', 'EMPSTATD',
                     'GEO1_MZ', 'GEO2_MZ']
    mig1_df = mig1_df.drop(columns=detailed_cols, inplace=False)
    mig5_df = mig5_df.drop(columns=detailed_cols, inplace=False)

    return mig1_df, mig5_df


def standardize_binary_vars(mig1_df, mig5_df) -> tuple:
    """Standardize binary variables."""

    # URBAN
    mig1_df['URBAN']= mig1_df['URBAN'].fillna(0).astype(int)
    mig5_df['URBAN'] = mig5_df['URBAN'].fillna(0).astype(int)

    # OWNERSHIP
    mig1_df['OWNERSHIP'] = mig1_df['OWNERSHIP'].fillna(0).astype(int)
    mig5_df['OWNERSHIP'] = mig5_df['OWNERSHIP'].fillna(0).astype(int)

    # PHONE
    mig1_df['PHONE'] = mig1_df['PHONE'].fillna(0).astype(int)
    mig5_df['PHONE'] = mig5_df['PHONE'].fillna(0).astype(int)

    # AUTOS
    mig1_df['AUTOS'] = mig1_df['AUTOS'].fillna(0).astype(int)
    mig5_df['AUTOS'] = mig5_df['AUTOS'].fillna(0).astype(int)

    # MORTMOT
    mig1_df['MORTMOT'] = mig1_df['MORTMOT'].fillna(0).astype(int)
    mig5_df['MORTMOT'] = mig5_df['MORTMOT'].fillna(0).astype(int)

    # MORTFAT
    mig1_df['MORTFAT'] = mig1_df['MORTFAT'].fillna(0).astype(int)
    mig5_df['MORTFAT'] = mig5_df['MORTFAT'].fillna(0).astype(int)

    # NATIVITY
    mig1_df['NATIVITY'] = mig1_df['NATIVITY'].fillna(0).astype(int)
    mig5_df['NATIVITY'] = mig5_df['NATIVITY'].fillna(0).astype(int)

    # CITIZEN
    mig1_df['CITIZEN'] = mig1_df['CITIZEN'].fillna(0).astype(int)
    mig5_df['CITIZEN'] = mig5_df['CITIZEN'].fillna(0).astype(int)

    # LIT
    mig1_df['LIT'] = mig1_df['LIT'].fillna(0).astype(int)
    mig5_df['LIT'] = mig5_df['LIT'].fillna(0).astype(int)

    # LABFORCE
    mig1_df['LABFORCE'] = mig1_df['LABFORCE'].fillna(0).astype(int)
    mig5_df['LABFORCE'] = mig5_df['LABFORCE'].fillna(0).astype(int)

    # MIGRATE1, MIGRATE5
    mig1_df['MIGRATE1'] = mig1_df['MIGRATE1'].fillna(0).astype(int)
    mig5_df['MIGRATE5'] = mig5_df['MIGRATE5'].fillna(0).astype(int)

    return mig1_df, mig5_df


def binarize_categorical_vars(mig1_df, mig5_df) -> tuple:
    """Binarize categorical variables."""

    # YEAR: 1 var -> 3 vars
    mig1_df['YEAR_1997'] = np.where(mig1_df['YEAR'] == '1997', 1, 0)
    mig1_df['YEAR_2007'] = np.where(mig1_df['YEAR'] == '2007', 1, 0)
    mig1_df['YEAR_2017'] = np.where(mig1_df['YEAR'] == '2017', 1, 0)

    mig5_df['YEAR_1997'] = np.where(mig5_df['YEAR'] == '1997', 1, 0)
    mig5_df['YEAR_2007'] = np.where(mig5_df['YEAR'] == '2007', 1, 0)
    mig5_df['YEAR_2017'] = np.where(mig5_df['YEAR'] == '2017', 1, 0)

    # GQ: 1 var -> 3 vars
    mig1_df['GQ_HOUSEHOLD'] = np.where(mig1_df['GQ'] == 'Households', 1, 0)
    mig1_df['GQ_INSTITUTION'] = np.where(mig1_df['GQ'] == 'Institutions', 1, 0)
    mig1_df['GQ_OTHER'] = np.where((mig1_df['GQ'] == '1-person unit created by splitting large household') |
                                    (mig1_df['GQ'] == 'Other group quarters'), 1, 0)

    mig5_df['GQ_HOUSEHOLD'] = np.where(mig5_df['GQ'] == 'Households', 1, 0)
    mig5_df['GQ_INSTITUTION'] = np.where(mig5_df['GQ'] == 'Institutions', 1, 0)
    mig5_df['GQ_OTHER'] = np.where((mig5_df['GQ'] == '1-person unit created by splitting large household') |
                                    (mig5_df['GQ'] == 'Other group quarters'), 1, 0)

    # HHTYPE: 1 var -> 3 vars
    mig1_married = mig1_df['HHTYPE'].isin([
        'Married/cohab couple with children',
        'Married/cohab couple, no children'
    ])
    mig1_single = mig1_df['HHTYPE'].isin([
        'Single-parent family',
        'One-person household'
    ])

    mig5_married = mig5_df['HHTYPE'].isin([
        'Married/cohab couple with children',
        'Married/cohab couple, no children'
    ])
    mig5_single = mig5_df['HHTYPE'].isin([
        'Single-parent family',
        'One-person household'
    ])

    mig1_df['HHTYPE_MARRIED'] = np.where(mig1_married, 1, 0)
    mig1_df['HHTYPE_SINGLE'] = np.where(mig1_single, 1, 0)
    mig1_df['HHTYPE_OTHER'] = np.where(~(mig1_married | mig1_single), 1, 0)

    mig5_df['HHTYPE_MARRIED'] = np.where(mig5_married, 1, 0)
    mig5_df['HHTYPE_SINGLE'] = np.where(mig5_single, 1, 0)
    mig5_df['HHTYPE_OTHER'] = np.where(~(mig5_married | mig5_single), 1, 0)

    # RESIDENT: 1 var -> 1 var
    mig1_df['RESIDENT'] = np.where(mig1_df['RESIDENT'] == 'Present resident', 1, 0)
    mig5_df['RESIDENT'] = np.where(mig5_df['RESIDENT'] == 'Present resident', 1, 0)

    # MARST: 1 var -> 3 vars
    mig1_married = mig1_df['MARST'] == 'Married/in union'
    mig1_single = mig1_df['MARST'] == 'Single/never married'
    mig1_other = ((mig1_df['MARST'] == 'Separated/divorced/spouse absent') |
                  (mig1_df['MARST'] == 'Widowed'))

    mig5_married = mig5_df['MARST'] == 'Married/in union'
    mig5_single = mig5_df['MARST'] == 'Single/never married'
    mig5_other = ((mig5_df['MARST'] == 'Separated/divorced/spouse absent') |
                  (mig5_df['MARST'] == 'Widowed'))

    mig1_df['MARST_MARRIED'] = np.where(mig1_married, 1, 0)
    mig1_df['MARST_SINGLE'] = np.where(mig1_single, 1, 0)
    mig1_df['MARST_OTHER'] = np.where(mig1_other, 1, 0)

    mig5_df['MARST_MARRIED'] = np.where(mig5_married, 1, 0)
    mig5_df['MARST_SINGLE'] = np.where(mig5_single, 1, 0)
    mig5_df['MARST_OTHER'] = np.where(mig5_other, 1, 0)

    # BPL1_MZ: 1 var -> 2 vars
    mig1_foreign = mig1_df['BPL1_MZ'] == 'Foreign Country'
    mig1_domestic = mig1_df['BPL1_MZ'] != 'Foreign Country'

    mig5_foreign = mig5_df['BPL1_MZ'] == 'Foreign Country'
    mig5_domestic = mig5_df['BPL1_MZ'] != 'Foreign Country'

    mig1_df['BP_FOREIGN'] = np.where(mig1_foreign, 1, 0)
    mig1_df['BP_DOMESTIC'] = np.where(mig1_domestic, 1, 0)

    mig5_df['BP_FOREIGN'] = np.where(mig5_foreign, 1, 0)
    mig5_df['BP_DOMESTIC'] = np.where(mig5_domestic, 1, 0)

    # SCHOOL: 1 var -> 1 var
    mig1_df['SCHOOL'] = np.where(mig1_df['SCHOOL'] == 'Yes', 1, 0)
    mig5_df['SCHOOL'] = np.where(mig5_df['SCHOOL'] == 'Yes', 1, 0)

    # EDATTAIN: 1 var -> 4 vars
    mig1_none = mig1_df['EDATTAIN'] == 'Less than primary completed'
    mig1_primary = mig1_df['EDATTAIN'] == 'Primary completed'
    mig1_secondary = mig1_df['EDATTAIN'] == 'Secondary completed'
    mig1_higher = mig1_df['EDATTAIN'] == 'University completed'

    mig5_none = mig5_df['EDATTAIN'] == 'Less than primary completed'
    mig5_primary = mig5_df['EDATTAIN'] == 'Primary completed'
    mig5_secondary = mig5_df['EDATTAIN'] == 'Secondary completed'
    mig5_higher = mig5_df['EDATTAIN'] == 'University completed'

    mig1_df['EDU_NONE'] = np.where(mig1_none, 1, 0)
    mig1_df['EDU_PRIMARY'] = np.where(mig1_primary, 1, 0)
    mig1_df['EDU_SECONDARY'] = np.where(mig1_secondary, 1, 0)
    mig1_df['EDU_HIGHER'] = np.where(mig1_higher, 1, 0)

    mig5_df['EDU_NONE'] = np.where(mig5_none, 1, 0)
    mig5_df['EDU_PRIMARY'] = np.where(mig5_primary, 1, 0)
    mig5_df['EDU_SECONDARY'] = np.where(mig5_secondary, 1, 0)
    mig5_df['EDU_HIGHER'] = np.where(mig5_higher, 1, 0)

    # EMPSTAT: 1 var -> 1 var
    mig1_df['EMPSTAT'] = np.where(mig1_df['EMPSTAT'] == 'Employed', 1, 0)
    mig5_df['EMPSTAT'] = np.where(mig5_df['EMPSTAT'] == 'Employed', 1, 0)

    # Drop Old Columns
    drop_cols = ['YEAR', 'GQ', 'HHTYPE', 'MARST', 'BPL1_MZ', 'EDATTAIN']
    mig1_df = mig1_df.drop(columns=drop_cols, inplace=False)
    mig5_df = mig5_df.drop(columns=drop_cols, inplace=False)

    return mig1_df, mig5_df


def bin_continuous_vars(mig1_df, mig5_df) -> tuple:
    """Bin continuous variables."""

    # PERSONS: 1 var -> 5 vars
    mig1_df['PERSONS_10'] = np.where(mig1_df['PERSONS'] <= 10, 1, 0)                                    # PERSONS ≤ 10
    mig1_df['PERSONS_20'] = np.where((mig1_df['PERSONS'] >= 11) & (mig1_df['PERSONS'] <= 20), 1, 0)     # 11 ≤ PERSONS ≤ 20
    mig1_df['PERSONS_30'] = np.where((mig1_df['PERSONS'] >= 21) & (mig1_df['PERSONS'] <= 30), 1, 0)     # 21 ≤ PERSONS ≤ 30
    mig1_df['PERSONS_40'] = np.where((mig1_df['PERSONS'] >= 31) & (mig1_df['PERSONS'] <= 40), 1, 0)     # 31 ≤ PERSONS ≤ 40
    mig1_df['PERSONS_50'] = np.where((mig1_df['PERSONS'] >= 41) & (mig1_df['PERSONS'] <= 50), 1, 0)     # 41 ≤ PERSONS ≤ 50

    mig5_df['PERSONS_10'] = np.where(mig5_df['PERSONS'] <= 10, 1, 0)                                    # PERSONS ≤ 10
    mig5_df['PERSONS_20'] = np.where((mig5_df['PERSONS'] >= 11) & (mig5_df['PERSONS'] <= 20), 1, 0)     # 11 ≤ PERSONS ≤ 20
    mig5_df['PERSONS_30'] = np.where((mig5_df['PERSONS'] >= 21) & (mig5_df['PERSONS'] <= 30), 1, 0)     # 21 ≤ PERSONS ≤ 30
    mig5_df['PERSONS_40'] = np.where((mig5_df['PERSONS'] >= 31) & (mig5_df['PERSONS'] <= 40), 1, 0)     # 31 ≤ PERSONS ≤ 40
    mig5_df['PERSONS_50'] = np.where((mig5_df['PERSONS'] >= 41) & (mig5_df['PERSONS'] <= 50), 1, 0)     # 41 ≤ PERSONS ≤ 50

    # ROOMS: 1 var -> 4 vars
    mig1_df['ROOMS_5'] = np.where(mig1_df['ROOMS'] <= 5, 1, 0)                                    # ROOMS ≤ 5
    mig1_df['ROOMS_10'] = np.where((mig1_df['ROOMS'] >= 6) & (mig1_df['ROOMS'] <= 10), 1, 0)      # 6 ≤ ROOMS ≤ 10
    mig1_df['ROOMS_15'] = np.where((mig1_df['ROOMS'] >= 11) & (mig1_df['ROOMS'] <= 15), 1, 0)     # 11 ≤ ROOMS ≤ 15
    mig1_df['ROOMS_20'] = np.where((mig1_df['ROOMS'] >= 11) & (mig1_df['ROOMS'] <= 15), 1, 0)     # 16 ≤ ROOMS ≤ 20

    mig5_df['ROOMS_5'] = np.where(mig5_df['ROOMS'] <= 5, 1, 0)                                    # ROOMS ≤ 5
    mig5_df['ROOMS_10'] = np.where((mig5_df['ROOMS'] >= 6) & (mig5_df['ROOMS'] <= 10), 1, 0)      # 6 ≤ ROOMS ≤ 10
    mig5_df['ROOMS_15'] = np.where((mig5_df['ROOMS'] >= 11) & (mig5_df['ROOMS'] <= 15), 1, 0)     # 11 ≤ ROOMS ≤ 15
    mig5_df['ROOMS_20'] = np.where((mig5_df['ROOMS'] >= 11) & (mig5_df['ROOMS'] <= 15), 1, 0)     # 16 ≤ ROOMS ≤ 20

    # FAMSIZE: 1 var -> 9 vars
    mig1_df['FAMSIZE_5'] = np.where(mig1_df['FAMSIZE'] <= 5, 1, 0)                                      # FAMSIZE ≤ 5
    mig1_df['FAMSIZE_10'] = np.where((mig1_df['FAMSIZE'] >= 6) & (mig1_df['FAMSIZE'] <= 10), 1, 0)      # 6 ≤ FAMSIZE ≤ 10
    mig1_df['FAMSIZE_15'] = np.where((mig1_df['FAMSIZE'] >= 11) & (mig1_df['FAMSIZE'] <= 15), 1, 0)     # 11 ≤ FAMSIZE ≤ 15
    mig1_df['FAMSIZE_20'] = np.where((mig1_df['FAMSIZE'] >= 16) & (mig1_df['FAMSIZE'] <= 20), 1, 0)     # 16 ≤ FAMSIZE ≤ 20
    mig1_df['FAMSIZE_25'] = np.where((mig1_df['FAMSIZE'] >= 21) & (mig1_df['FAMSIZE'] <= 25), 1, 0)     # 21 ≤ FAMSIZE ≤ 25
    mig1_df['FAMSIZE_30'] = np.where((mig1_df['FAMSIZE'] >= 26) & (mig1_df['FAMSIZE'] <= 30), 1, 0)     # 26 ≤ FAMSIZE ≤ 30
    mig1_df['FAMSIZE_35'] = np.where((mig1_df['FAMSIZE'] >= 31) & (mig1_df['FAMSIZE'] <= 35), 1, 0)     # 31 ≤ FAMSIZE ≤ 35
    mig1_df['FAMSIZE_40'] = np.where((mig1_df['FAMSIZE'] >= 36) & (mig1_df['FAMSIZE'] <= 40), 1, 0)     # 36 ≤ FAMSIZE ≤ 40
    mig1_df['FAMSIZE_45'] = np.where((mig1_df['FAMSIZE'] >= 41) & (mig1_df['FAMSIZE'] <= 45), 1, 0)     # 41 ≤ FAMSIZE ≤ 45

    mig5_df['FAMSIZE_5'] = np.where(mig5_df['FAMSIZE'] <= 5, 1, 0)                                      # FAMSIZE ≤ 5
    mig5_df['FAMSIZE_10'] = np.where((mig5_df['FAMSIZE'] >= 6) & (mig5_df['FAMSIZE'] <= 10), 1, 0)      # 6 ≤ FAMSIZE ≤ 10
    mig5_df['FAMSIZE_15'] = np.where((mig5_df['FAMSIZE'] >= 11) & (mig5_df['FAMSIZE'] <= 15), 1, 0)     # 11 ≤ FAMSIZE ≤ 15
    mig5_df['FAMSIZE_20'] = np.where((mig5_df['FAMSIZE'] >= 16) & (mig5_df['FAMSIZE'] <= 20), 1, 0)     # 16 ≤ FAMSIZE ≤ 20
    mig5_df['FAMSIZE_25'] = np.where((mig5_df['FAMSIZE'] >= 21) & (mig5_df['FAMSIZE'] <= 25), 1, 0)     # 21 ≤ FAMSIZE ≤ 25
    mig5_df['FAMSIZE_30'] = np.where((mig5_df['FAMSIZE'] >= 26) & (mig5_df['FAMSIZE'] <= 30), 1, 0)     # 26 ≤ FAMSIZE ≤ 30
    mig5_df['FAMSIZE_35'] = np.where((mig5_df['FAMSIZE'] >= 31) & (mig5_df['FAMSIZE'] <= 35), 1, 0)     # 31 ≤ FAMSIZE ≤ 35
    mig5_df['FAMSIZE_40'] = np.where((mig5_df['FAMSIZE'] >= 36) & (mig5_df['FAMSIZE'] <= 40), 1, 0)     # 36 ≤ FAMSIZE ≤ 40
    mig5_df['FAMSIZE_45'] = np.where((mig5_df['FAMSIZE'] >= 41) & (mig5_df['FAMSIZE'] <= 45), 1, 0)     # 41 ≤ FAMSIZE ≤ 45

    # NCHILD: 1 var -> 5 vars
    mig1_df['NCHILD_2'] = np.where(mig1_df['NCHILD'] <= 2, 1, 0)                                  # NCHILD ≤ 2
    mig1_df['NCHILD_4'] = np.where((mig1_df['NCHILD'] >= 3) & (mig1_df['NCHILD'] <= 4), 1, 0)     # 3 ≤ NCHILD ≤ 4
    mig1_df['NCHILD_6'] = np.where((mig1_df['NCHILD'] >= 5) & (mig1_df['NCHILD'] <= 6), 1, 0)     # 5 ≤ NCHILD ≤ 6
    mig1_df['NCHILD_8'] = np.where((mig1_df['NCHILD'] >= 7) & (mig1_df['NCHILD'] <= 8), 1, 0)     # 7 ≤ NCHILD ≤ 8
    mig1_df['NCHILD_10'] = np.where((mig1_df['NCHILD'] >= 9) & (mig1_df['NCHILD'] <= 10), 1, 0)   # 9 ≤ NCHILD ≤ 10

    mig5_df['NCHILD_2'] = np.where(mig5_df['NCHILD'] <= 2, 1, 0)                                  # NCHILD ≤ 2
    mig5_df['NCHILD_4'] = np.where((mig5_df['NCHILD'] >= 3) & (mig5_df['NCHILD'] <= 4), 1, 0)     # 3 ≤ NCHILD ≤ 4
    mig5_df['NCHILD_6'] = np.where((mig5_df['NCHILD'] >= 5) & (mig5_df['NCHILD'] <= 6), 1, 0)     # 5 ≤ NCHILD ≤ 6
    mig5_df['NCHILD_8'] = np.where((mig5_df['NCHILD'] >= 7) & (mig5_df['NCHILD'] <= 8), 1, 0)     # 7 ≤ NCHILD ≤ 8
    mig5_df['NCHILD_10'] = np.where((mig5_df['NCHILD'] >= 9) & (mig5_df['NCHILD'] <= 10), 1, 0)   # 9 ≤ NCHILD ≤ 10

    # AGE: 1 var -> 10 vars
    mig1_df['AGE_10'] = np.where(mig1_df['AGE'] <= 10, 1, 0)                                # AGE ≤ 10
    mig1_df['AGE_20'] = np.where((mig1_df['AGE'] >= 11) & (mig1_df['AGE'] <= 20), 1, 0)     # 11 ≤ AGE ≤ 20
    mig1_df['AGE_30'] = np.where((mig1_df['AGE'] >= 21) & (mig1_df['AGE'] <= 30), 1, 0)     # 21 ≤ AGE ≤ 30
    mig1_df['AGE_40'] = np.where((mig1_df['AGE'] >= 31) & (mig1_df['AGE'] <= 40), 1, 0)     # 31 ≤ AGE ≤ 40
    mig1_df['AGE_50'] = np.where((mig1_df['AGE'] >= 41) & (mig1_df['AGE'] <= 50), 1, 0)     # 41 ≤ AGE ≤ 50
    mig1_df['AGE_60'] = np.where((mig1_df['AGE'] >= 51) & (mig1_df['AGE'] <= 60), 1, 0)     # 51 ≤ AGE ≤ 60
    mig1_df['AGE_70'] = np.where((mig1_df['AGE'] >= 61) & (mig1_df['AGE'] <= 70), 1, 0)     # 61 ≤ AGE ≤ 70
    mig1_df['AGE_80'] = np.where((mig1_df['AGE'] >= 71) & (mig1_df['AGE'] <= 80), 1, 0)     # 71 ≤ AGE ≤ 80
    mig1_df['AGE_90'] = np.where((mig1_df['AGE'] >= 81) & (mig1_df['AGE'] <= 90), 1, 0)     # 81 ≤ AGE ≤ 90
    mig1_df['AGE_100'] = np.where((mig1_df['AGE'] >= 91) & (mig1_df['AGE'] <= 100), 1, 0)   # 91 ≤ AGE ≤ 100

    mig5_df['AGE_10'] = np.where(mig5_df['AGE'] <= 10, 1, 0)                                # AGE ≤ 10
    mig5_df['AGE_20'] = np.where((mig5_df['AGE'] >= 11) & (mig5_df['AGE'] <= 20), 1, 0)     # 11 ≤ AGE ≤ 20
    mig5_df['AGE_30'] = np.where((mig5_df['AGE'] >= 21) & (mig5_df['AGE'] <= 30), 1, 0)     # 21 ≤ AGE ≤ 30
    mig5_df['AGE_40'] = np.where((mig5_df['AGE'] >= 31) & (mig5_df['AGE'] <= 40), 1, 0)     # 31 ≤ AGE ≤ 40
    mig5_df['AGE_50'] = np.where((mig5_df['AGE'] >= 41) & (mig5_df['AGE'] <= 50), 1, 0)     # 41 ≤ AGE ≤ 50
    mig5_df['AGE_60'] = np.where((mig5_df['AGE'] >= 51) & (mig5_df['AGE'] <= 60), 1, 0)     # 51 ≤ AGE ≤ 60
    mig5_df['AGE_70'] = np.where((mig5_df['AGE'] >= 61) & (mig5_df['AGE'] <= 70), 1, 0)     # 61 ≤ AGE ≤ 70
    mig5_df['AGE_80'] = np.where((mig5_df['AGE'] >= 71) & (mig5_df['AGE'] <= 80), 1, 0)     # 71 ≤ AGE ≤ 80
    mig5_df['AGE_90'] = np.where((mig5_df['AGE'] >= 81) & (mig5_df['AGE'] <= 90), 1, 0)     # 81 ≤ AGE ≤ 90
    mig5_df['AGE_100'] = np.where((mig5_df['AGE'] >= 91) & (mig5_df['AGE'] <= 100), 1, 0)   # 91 ≤ AGE ≤ 100

    # Drop Old Columns
    drop_cols = ['PERSONS', 'ROOMS', 'FAMSIZE', 'NCHILD', 'AGE']
    mig1_df = mig1_df.drop(columns=drop_cols, inplace=False)
    mig5_df = mig5_df.drop(columns=drop_cols, inplace=False)

    return mig1_df, mig5_df
