"""
Based of the implementation of "An attention based deep learning model of clinical events in the intensive care unit".
Credit: DA Kaji (https://github.com/deepak-kaji/mimic-lstm)
"""

import csv
import gc
import math
import os
import pickle
import re
import warnings
from functools import reduce
from operator import add
from pathlib import Path
from time import time

import numpy as np
import pandas as pd


def split_data(X, y, train_split_percentage, val_split_percentage):
    """
    Args:
        X: features of whole dataset
        y: labels of whole dataset
        train_split_percentage: percentage of data that belongs to the train set
        val_split_percentage: percentage of data that belongs to validation set

    Returns:
        (train_x, train_y): the training set
        (val_x, val_y): the validation set
        (test_x, test_y): the test set
    """
    train_x, train_y, val_x, val_y, test_x, test_y = None, None, None, None, None, None
    train_x = X[0 : int(train_split_percentage * X.shape[0]), :, :]
    
    if len(y.shape) == 2:
        train_y = y[0 : int(train_split_percentage * y.shape[0]), :]
        train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], 1)
    else:
        train_y = y[0 : int(train_split_percentage * y.shape[0]), :, :]

    val_x = X[
        int(train_split_percentage * X.shape[0]) : int(
            (train_split_percentage + val_split_percentage) * X.shape[0]
        )
    ]
    
    val_y = y[
        int(train_split_percentage * y.shape[0]) : int(
            (train_split_percentage + val_split_percentage) * y.shape[0]
        )
    ]
    if len(y.shape) == 2:
        val_y = val_y.reshape(val_y.shape[0], val_y.shape[1], 1)
    
   
    test_x = X[int((train_split_percentage + val_split_percentage) * X.shape[0]) : :]
    test_y = y[int((train_split_percentage + val_split_percentage) * X.shape[0]) : :]
    
    if len(y.shape) == 2:
        test_y = test_y.reshape(test_y.shape[0], test_y.shape[1], 1)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


class PadSequences(object):
    def __init__(self):
        self.name = "padder"

    def pad(self, df, lb, time_steps, pad_value=-100):

        """Takes a file path for the dataframe to operate on. lb is a lower bound to discard
        ub is an upper bound to truncate on. All entries are padded to their ubber bound"""

        self.uniques = pd.unique(df["HADM_ID"])
        df = (
            df.groupby("HADM_ID")
            .filter(lambda group: len(group) > lb)
            .reset_index(drop=True)
        )
        df = (
            df.groupby("HADM_ID")
            .apply(lambda group: group[0:time_steps])
            .reset_index(drop=True)
        )
        df = (
            df.groupby("HADM_ID")
            .apply(
                lambda group: pd.concat(
                    [
                        group,
                        pd.DataFrame(
                            pad_value
                            * np.ones((time_steps - len(group), len(df.columns))),
                            columns=df.columns,
                        ),
                    ],
                    axis=0,
                )
            )
            .reset_index(drop=True)
        )

        return df

    def ZScoreNormalize(self, matrix):
        """Performs Z Score Normalization for 3rd order tensors
        matrix should be (batchsize, time_steps, features)
        Padded time steps should be masked with np.nan"""
        means = np.nanmean(matrix, axis=(0, 1))
        stds = np.nanstd(matrix, axis=(0, 1))
        return (matrix - means) / stds
        
def wbc_crit(x):
    if (x > 12 or x < 4) and x != 0:
        return 1
    else:
        return 0


def temp_crit(x):
    if (x > 100.4 or x < 96.8) and x != 0:
        return 1
    else:
        return 0

def get_target(df, target):
    target_df = pd.DataFrame()

    if target == "MI":
        target_df[target] = ((df["troponin"] > 0.4) & (df["CKD"] == 0)).apply(lambda x: int(x))
    elif target == "SEPSIS":
        df["hr_sepsis"] = df["heart rate"].apply(lambda x: 1 if x > 90 else 0)
        df["respiratory rate_sepsis"] = df["respiratory rate"].apply(
            lambda x: 1 if x > 20 else 0
        )
        df["wbc_sepsis"] = df["WBCs"].apply(wbc_crit)
        df["temperature f_sepsis"] = df["temperature (F)"].apply(temp_crit)
        df["sepsis_points"] = (
            df["hr_sepsis"]
            + df["respiratory rate_sepsis"]
            + df["wbc_sepsis"]
            + df["temperature f_sepsis"]
        )
        target_df[target] = ((df["sepsis_points"] >= 2) & (df["Infection"] == 1)).apply(
            lambda x: int(x)
        )
        del df["hr_sepsis"]
        del df["respiratory rate_sepsis"]
        del df["wbc_sepsis"]
        del df["temperature f_sepsis"]
        del df["sepsis_points"]
        del df["Infection"]
    elif target == "PE":
        df["blood_thinner"] = (
            df["heparin"] + df["enoxaparin"] + df["fondaparinux"]
        ).apply(lambda x: 1 if x >= 1 else 0)
        target_df[target] = df["blood_thinner"] & df["ct_angio"]
        del df["blood_thinner"]
    elif target == "VANCOMYCIN":
        target_df["VANCOMYCIN"] = df["vancomycin"].apply(lambda x: 1 if x > 0 else 0)
        del df["vancomycin"]
    elif isinstance(target, dict):
        # MASK FEATURES
        target_df = df.drop(columns = ['HADMID_DAY', "SUBJECT_ID", 'DOB', "YOB", "ADMITYEAR", 'ADMITTIME'])
        if target['features'] != 'all':
           target_df = target_df[target['features']]

        def nan_normalize(df):
            hadm_id = df['HADM_ID']
            df_na = df.drop(columns='HADM_ID').replace(0, np.NaN)
            df = (df_na - df_na.mean()) / df_na.std()
            df['HADM_ID'] = hadm_id
            return df.replace(np.NaN, 0)

        def window_max_delta(group):
            hadm_id = group['HADM_ID'].iloc[:-target['forward']]
            group = group.drop(columns='HADM_ID')

            max = group[::-1].shift().rolling(target['forward']).max()[::-1].iloc[:-target['forward']]
            min = group[::-1].shift().rolling(target['forward']).min()[::-1].iloc[:-target['forward']]
            group = group.iloc[:-target['forward']]

            max_diff = max.values - group.values
            min_diff = min.values - group.values
            abs_diff = np.where(np.abs(max_diff) > np.abs(min_diff), max_diff, min_diff)
            group = pd.DataFrame(abs_diff, index = group.index, columns = group.columns)
            group['HADM_ID'] = hadm_id
            return group

        target_df = nan_normalize(target_df)

        if target['forward'] > 0:
            target_df = target_df.groupby("HADM_ID").apply(window_max_delta).reset_index(drop=True)
            df = df.groupby("HADM_ID").apply(lambda group: group.iloc[:-target['forward']]).reset_index(drop=True)

        if target['trend'] > 0:
            hadm_id = target_df['HADM_ID'].copy()
            values = target_df.values
            values[values >= target['trend']] = 1
            values[values <= -target['trend']] = 1
            values[np.abs(values) != 1] = 0
            target_df = pd.DataFrame(values, index = target_df.index, columns = target_df.columns)
            target_df['HADM_ID'] = hadm_id

        if target['dropout'] > 0:
            mask = np.random.choice([True, False], size=df.shape, p=[target['dropout'], 1-target['dropout']])
            df = df.mask(mask)
        target_df = target_df.drop(columns = ['HADM_ID'])

    feature_df = df.select_dtypes(exclude=["object"])
    return feature_df, target_df

def get_feature_columns(feature_df, target):
    COLUMNS = list(feature_df.columns)

    if target == "MI":
        toss = [
            "ct_angio",
            "troponin",
            "troponin_std",
            "troponin_min",
            "troponin_max",
            "Infection",
            "CKD",
        ]
        COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == "SEPSIS":
        toss = ["ct_angio", "Infection", "CKD"]
        COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == "PE":
        toss = [
            "ct_angio",
            "heparin",
            "heparin_std",
            "heparin_min",
            "heparin_max",
            "enoxaparin",
            "enoxaparin_std",
            "enoxaparin_min",
            "enoxaparin_max",
            "fondaparinux",
            "fondaparinux_std",
            "fondaparinux_min",
            "fondaparinux_max",
            "Infection",
            "CKD",
        ]
        COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == "VANCOMYCIN":
        toss = ["ct_angio", "Infection", "CKD"]
        COLUMNS = [i for i in COLUMNS if i not in toss]
    
    toss = ["HADM_ID", "SUBJECT_ID", "YOB", "ADMITYEAR"]
    COLUMNS = [i for i in COLUMNS if i not in toss]
    return COLUMNS

def to_matrix(feature_df, target, target_df, time_steps):
    FEATURE_COLUMNS = get_feature_columns(feature_df, target)

    FEATURE_MATRIX = feature_df[FEATURE_COLUMNS].values
    FEATURE_MATRIX = FEATURE_MATRIX.reshape(int(FEATURE_MATRIX.shape[0] / time_steps), time_steps, FEATURE_MATRIX.shape[1])

    LABEL_MATRIX = target_df.values
    LABEL_MATRIX = LABEL_MATRIX.reshape(int(LABEL_MATRIX.shape[0] / time_steps), time_steps, LABEL_MATRIX.shape[1])

    return FEATURE_MATRIX, LABEL_MATRIX, FEATURE_COLUMNS
    
def normalize_matrix(MATRIX, bool_matrix, pad_value):
    # Replace zero padded rows with nan to calculate distribution statistics
    MATRIX[bool_matrix] = np.nan 
    MATRIX = PadSequences().ZScoreNormalize(MATRIX)
    # Reset padded rows to zero
    bool_matrix = np.isnan(MATRIX)
    MATRIX[bool_matrix] = pad_value
    return MATRIX

def get_bool_matrix(MATRIX):
    bool_matrix = MATRIX!=0
    mask = ~(bool_matrix.any(axis=-1))
    return mask

def shuffle_matrix(MATRIX, bool_matrix):
    # Shuffle patient axis
    np.random.seed(0)
    permutation = np.random.permutation(MATRIX.shape[0])
    MATRIX = MATRIX[permutation]
    bool_matrix = bool_matrix[permutation]
    return MATRIX, bool_matrix

def split_matrix(X_MATRIX, Y_MATRIX, bool_matrix, 
                 tt_split, val_percentage):
    
    (X_TRAIN, Y_TRAIN), (X_VAL, Y_VAL), (X_TEST, Y_TEST) = split_data(
        X_MATRIX, Y_MATRIX, tt_split, val_percentage - tt_split
    )

    val_idx_start = int(tt_split * bool_matrix.shape[0])
    val_idx_end = int(val_percentage * bool_matrix.shape[0])
    val_boolmat = bool_matrix[val_idx_start:val_idx_end]

    test_idx_start = int(val_percentage * bool_matrix.shape[0])
    test_boolmat = bool_matrix[test_idx_start : :]

    return X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, X_TEST, Y_TEST, val_boolmat, test_boolmat
    
def return_data(
    FILE,
    balancer=True,
    target="MI",
    tt_split=0.7,
    val_percentage=0.8,
    time_steps=14,
    pad=True,
    split_data=split_data,
):

    """
    Returns synthetic or real data depending on parameter
    Args:
    -----
      balance : whether or not to balance positive and negative time windows
      target : desired target, supports MI, SEPSIS, VANCOMYCIN or a known lab, medication
      return_cols : return columns used for this RNN
      tt_split : fraction of dataset to use fro training, remaining is used for test
      time_steps : 14 by default, required for padding
      split : creates test train splits
      pad : by default is True, will pad to the time_step value
    Returns:
    -------
      Training and validation splits as well as the number of columns for use in RNN
    """

    df = pd.read_csv(FILE)

    feature_df, target_df = get_target(df, target)

    if pad:
        pad_value = 0
        N_targets = len(target_df.columns)
        df = pd.concat([feature_df, target_df.add_suffix('_label')], axis = 1)
        df = PadSequences().pad(df, 1, time_steps, pad_value=pad_value)
        feature_df, target_df = df[df.columns[:-N_targets]], df[df.columns[-N_targets:]]

    FEATURE_MATRIX, LABEL_MATRIX, FEATURE_COLUMNS = to_matrix(feature_df, target, target_df, time_steps)
    bool_matrix = get_bool_matrix(FEATURE_MATRIX)

    X_MATRIX = normalize_matrix(FEATURE_MATRIX, bool_matrix, pad_value)
    Y_MATRIX = LABEL_MATRIX

    x = split_matrix(X_MATRIX, Y_MATRIX, bool_matrix, tt_split, val_percentage)
    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, X_TEST, Y_TEST, val_boolmat, test_boolmat = x

    X_TEST[test_boolmat] = pad_value
    Y_TEST[test_boolmat] = pad_value

    if balancer:
        assert isinstance(target, str)
        TRAIN = np.concatenate([X_TRAIN, Y_TRAIN], axis=2)
        pos_ind = np.unique(np.where((TRAIN[:, :, -1] == 1).any(axis=1))[0])
        np.random.shuffle(pos_ind)
        neg_ind = np.unique(np.where(~(TRAIN[:, :, -1] == 1).any(axis=1))[0])
        np.random.shuffle(neg_ind)
        length = min(pos_ind.shape[0], neg_ind.shape[0])
        total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
        np.random.shuffle(total_ind)
        ind = total_ind
        if target == "MI":
            ind = pos_ind
        else:
            ind = total_ind
        X_TRAIN = TRAIN[ind, :, 0:-1]
        Y_TRAIN = TRAIN[ind, :, -1]
        Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

    no_feature_cols = X_TRAIN.shape[2]
    no_label_cols = Y_TRAIN.shape[2]

    return (
        X_TRAIN,
        X_VAL,
        Y_TRAIN,
        Y_VAL,
        no_feature_cols,
        no_label_cols,
        X_TEST,
        Y_TEST,
        val_boolmat,
        test_boolmat,
        FEATURE_COLUMNS,
        target
    )

def build_seq_dataset(ROOT, TARGET):
    warnings.filterwarnings("ignore", message="DtypeWarning")
    TIME_STEPS = 15
    SAVED_DATA_PATH = Path(f"{ROOT}/saved_data")
    SAVED_DATA_PATH.mkdir(exist_ok=True)
    fname = SAVED_DATA_PATH / (str(TARGET) + ".pkl")

    print(f"Building sequence dataset for {TARGET}, saving to {fname}")
    fname_parsed_data = f"{ROOT}/mimic_database/mapped_elements/CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_plus_notes.csv"
    data = return_data(
        fname_parsed_data,
        balancer=isinstance(TARGET, str),
        target=TARGET,
        pad=True,
        time_steps=TIME_STEPS,
    )
    print("train shapes ", data[0].shape, data[2].shape)

    with open(fname, "wb") as f:
        pickle.dump(data, f)
    print("Done saving")

def load_seq_dataset(ROOT, TARGET="SEPSIS"):
    assert TARGET in ["SEPSIS", "VANCOMYCIN", "MI"] or isinstance(TARGET, dict)
    TIME_STEPS = 15
    SAVED_DATA_PATH = Path(f"{ROOT}/saved_data")
    fname = SAVED_DATA_PATH / (str(TARGET) + ".pkl")
    if not os.path.exists(fname):
        raise ValueError("File does not exist. Try running build_seq_datasets again")
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data
