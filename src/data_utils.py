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

    def ZScoreNormalize(self, matrix, num_targets = 1):

        """Performs Z Score Normalization for 3rd order tensors
        matrix should be (batchsize, time_steps, features)
        Padded time steps should be masked with np.nan"""

        x_matrix = matrix[:, :, 0:-num_targets]
        y_matrix = matrix[:, :, -num_targets:]
        
        def norm(matrix):
            means = np.nanmean(matrix, axis=(0, 1))
            stds = np.nanstd(matrix, axis=(0, 1))
            return (matrix - means) / stds
        
        x_matrix = norm(x_matrix)
        y_matrix = norm(y_matrix) if num_targets > 1 else y_matrix
        
        matrix = np.concatenate([x_matrix, y_matrix], axis=2)
        return matrix

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
    if target == "MI":
        df[target] = ((df["troponin"] > 0.4) & (df["CKD"] == 0)).apply(lambda x: int(x))
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
        df[target] = ((df["sepsis_points"] >= 2) & (df["Infection"] == 1)).apply(
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
        df[target] = df["blood_thinner"] & df["ct_angio"]
        del df["blood_thinner"]
    elif target == "VANCOMYCIN":
        df["VANCOMYCIN"] = df["vancomycin"].apply(lambda x: 1 if x > 0 else 0)
        del df["vancomycin"]
    elif isinstance(target, list):

        print(df)

        df_features_grouped = df.groupby("HADM_ID").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        df_labels_grouped = df.groupby("HADM_ID")[target].apply(lambda x: x.shift(-1).iloc[:-1]).reset_index(drop=True).add_suffix('_label')
        df = pd.concat([df_features_grouped, df_labels_grouped], axis=1)
        
    df = df.select_dtypes(exclude=["object"])

    return df

def get_columns(df, target):
    COLUMNS = list(df.columns)

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
    
    if isinstance(target, list):
        toss = [x + '_label' for x in target]
        COLUMNS = [i for i in COLUMNS if i not in toss]
    else:
        COLUMNS.remove(target)

    if "HADM_ID" in COLUMNS:
        COLUMNS.remove("HADM_ID")
    if "SUBJECT_ID" in COLUMNS:
        COLUMNS.remove("SUBJECT_ID")
    if "YOB" in COLUMNS:
        COLUMNS.remove("YOB")
    if "ADMITYEAR" in COLUMNS:
        COLUMNS.remove("ADMITYEAR")
    return COLUMNS

def to_matrix(df, target, time_steps):
    COLUMNS = get_columns(df, target)

    if isinstance(target, list):
        target = [x + '_label' for x in target]
        MATRIX = df[COLUMNS + target].values
    else:
        MATRIX = df[COLUMNS + [target]].values

    MATRIX = MATRIX.reshape(
        int(MATRIX.shape[0] / time_steps), time_steps, MATRIX.shape[1]
    )
    return MATRIX, COLUMNS
    
def normalize_marix(MATRIX, pad_value, num_targets):
    # Replace zero padded rows with nan to calculate distribution statistics
    bool_matrix = ~MATRIX.any(axis=2)
    MATRIX[bool_matrix] = np.nan 
    MATRIX = PadSequences().ZScoreNormalize(MATRIX, num_targets)
    # Reset padded rows to zero
    bool_matrix = np.isnan(MATRIX)
    MATRIX[bool_matrix] = pad_value
    return MATRIX, bool_matrix

def shuffle_matrix(MATRIX, bool_matrix):
    # Shuffle patient axis
    np.random.seed(0)
    permutation = np.random.permutation(MATRIX.shape[0])
    MATRIX = MATRIX[permutation]
    bool_matrix = bool_matrix[permutation]
    return MATRIX, bool_matrix

def split_matrix(X_MATRIX, Y_MATRIX, x_bool_matrix, y_bool_matrix, 
                 tt_split, val_percentage):
    
    (X_TRAIN, Y_TRAIN), (X_VAL, Y_VAL), (X_TEST, Y_TEST) = split_data(
        X_MATRIX, Y_MATRIX, tt_split, val_percentage - tt_split
    )

    x_val_boolmat = x_bool_matrix[
        int(tt_split * x_bool_matrix.shape[0]) : int(
            val_percentage * x_bool_matrix.shape[0]
        )
    ]
    y_val_boolmat = y_bool_matrix[
        int(tt_split * y_bool_matrix.shape[0]) : int(
            val_percentage * y_bool_matrix.shape[0]
        )
    ]
    if len(y_val_boolmat.shape) == 2:
        y_val_boolmat = y_val_boolmat.reshape(
            y_val_boolmat.shape[0], y_val_boolmat.shape[1], 1
        )

    x_test_boolmat = x_bool_matrix[int(val_percentage * x_bool_matrix.shape[0]) : :]
    y_test_boolmat = y_bool_matrix[int(val_percentage * y_bool_matrix.shape[0]) : :]
    
    if len(y_test_boolmat.shape) == 2:
        y_test_boolmat = y_test_boolmat.reshape(
            y_test_boolmat.shape[0], y_test_boolmat.shape[1], 1
        )
    
    return X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, X_TEST, Y_TEST, x_test_boolmat, y_test_boolmat, x_val_boolmat, y_val_boolmat
    
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
      cross_val : parameter that returns entire matrix unsplit and unbalanced for cross val purposes
      time_steps : 14 by default, required for padding
      split : creates test train splits
      pad : by default is True, will pad to the time_step value
    Returns:
    -------
      Training and validation splits as well as the number of columns for use in RNN
    """

    df = pd.read_csv(FILE)
    df = get_target(df, target)
    N_target = len(target) if isinstance(target, list) else 1

    if pad:
        pad_value = 0
        df = PadSequences().pad(df, 1, time_steps, pad_value=pad_value)
    
    MATRIX, COLUMNS = to_matrix(df, target, time_steps)
    MATRIX, bool_matrix = normalize_marix(MATRIX, pad_value, N_target)
    MATRIX, bool_matrix = shuffle_matrix(MATRIX, bool_matrix)

    X_MATRIX = MATRIX[:, :, 0:-N_target]
    Y_MATRIX = MATRIX[:, :, -N_target:]
    x_bool_matrix = bool_matrix[:, :, 0:-N_target]
    y_bool_matrix = bool_matrix[:, :, -N_target:]

    x = split_matrix(X_MATRIX, Y_MATRIX, x_bool_matrix, y_bool_matrix, tt_split, val_percentage)
    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, X_TEST, Y_TEST, x_test_boolmat, y_test_boolmat, x_val_boolmat, y_val_boolmat = x
    X_TEST[x_test_boolmat] = pad_value
    Y_TEST[y_test_boolmat] = pad_value

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
        x_test_boolmat,
        y_test_boolmat,
        x_val_boolmat,
        y_val_boolmat,
        COLUMNS,
        target
    )


def build_seq_dataset(ROOT, TARGET):
    warnings.filterwarnings("ignore", message="DtypeWarning")
    TIME_STEPS = 15
    SAVED_DATA_PATH = Path(f"{ROOT}/saved_data")
    SAVED_DATA_PATH.mkdir(exist_ok=True)
    fname = SAVED_DATA_PATH / (str(TARGET) + "_" + str(TIME_STEPS) + ".pkl")

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
    assert TARGET in ["SEPSIS", "VANCOMYCIN", "MI"] or isinstance(TARGET, list)
    TIME_STEPS = 15
    SAVED_DATA_PATH = Path(f"{ROOT}/saved_data")
    fname = SAVED_DATA_PATH / (str(TARGET) + "_" + str(TIME_STEPS) + ".pkl")
    if not os.path.exists(fname):
        raise ValueError("File does not exist. Try running build_seq_datasets again")
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data
