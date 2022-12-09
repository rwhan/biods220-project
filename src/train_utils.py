"""
Based of the implementation of "An attention based deep learning model of clinical events in the intensive care unit".

Credit: DA Kaji (https://github.com/deepak-kaji/mimic-lstm)

"""
# tmp
FILE = "/home/jamesburgess/assign2/mimic_database/mapped_elements/CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_plus_notes.csv"

import concurrent.futures
import csv
import gc
import math
import os
import pickle
import re
from functools import reduce
from operator import add
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras import regularizers  # model_from_json
from tensorflow.keras import Input, Model  # model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Embedding,
    Flatten,
    Masking,
    Permute,
    Reshape,
    TimeDistributed,
    multiply,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_utils import return_data


# Driver for training model
def train(
    model_name="kaji_mach_0",
    synth_data=False,
    target="MI",
    balancer=True,
    predict=False,
    return_model=False,
    n_percentage=1.0,
    time_steps=14,
    epochs=10,
    build_model=None,
):

    """
    Use Keras model.fit using parameter inputs
    Args:
    ----
    model_name : Parameter used for naming the checkpoint_dir
    synth_data : Default to False. Allows you to use synthetic or real data.
    Return:
    -------
    Nonetype. Fits model only.
    """

    f = open("./pickled_objects/X_TRAIN_{0}.txt".format(target), "rb")
    X_TRAIN = pickle.load(f)
    f.close()

    f = open("./pickled_objects/Y_TRAIN_{0}.txt".format(target), "rb")
    Y_TRAIN = pickle.load(f)
    f.close()

    f = open("./pickled_objects/X_VAL_{0}.txt".format(target), "rb")
    X_VAL = pickle.load(f)
    f.close()

    f = open("./pickled_objects/Y_VAL_{0}.txt".format(target), "rb")
    Y_VAL = pickle.load(f)
    f.close()

    f = open("./pickled_objects/x_boolmat_val_{0}.txt".format(target), "rb")
    X_BOOLMAT_VAL = pickle.load(f)
    f.close()

    f = open("./pickled_objects/y_boolmat_val_{0}.txt".format(target), "rb")
    Y_BOOLMAT_VAL = pickle.load(f)
    f.close()

    f = open("./pickled_objects/no_feature_cols_{0}.txt".format(target), "rb")
    no_feature_cols = pickle.load(f)
    f.close()

    # X_TRAIN = X_TRAIN[0:int(n_percentage*X_TRAIN.shape[0])]
    # Y_TRAIN = Y_TRAIN[0:int(n_percentage*Y_TRAIN.shape[0])]

    # build model
    model = build_model(
        num_features=no_feature_cols, output_summary=True, time_steps=time_steps
    )

    # init callbacks
    tb_callback = TensorBoard(
        log_dir="./logs/{0}_{1}.log".format(model_name, time),
        histogram_freq=0,
        write_grads=False,
        write_images=True,
        write_graph=True,
    )

    # Make checkpoint dir and init checkpointer
    checkpoint_dir = "./saved_models/{0}".format(model_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpointer = ModelCheckpoint(
        filepath=checkpoint_dir + "/model.{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    # fit
    model.fit(
        x=X_TRAIN,
        y=Y_TRAIN,
        batch_size=16,
        epochs=epochs,
        callbacks=[tb_callback],  # , checkpointer],
        validation_data=(X_VAL, Y_VAL),
        shuffle=True,
    )

    model.save("./saved_models/{0}.h5".format(model_name))

    if predict:
        print("TARGET: {0}".format(target))
        Y_PRED = model.predict(X_VAL)
        Y_PRED = Y_PRED[~Y_BOOLMAT_VAL]
        np.unique(Y_PRED)
        Y_VAL = Y_VAL[~Y_BOOLMAT_VAL]
        Y_PRED_TRAIN = model.predict(X_TRAIN)
        print("Confusion Matrix Validation")
        print(confusion_matrix(Y_VAL, np.around(Y_PRED)))
        print("Validation Accuracy")
        print(accuracy_score(Y_VAL, np.around(Y_PRED)))
        print("ROC AUC SCORE VAL")
        print(roc_auc_score(Y_VAL, Y_PRED))
        print("CLASSIFICATION REPORT VAL")
        print(classification_report(Y_VAL, np.around(Y_PRED)))

    if return_model:
        return model


def pickle_objects(target="MI", time_steps=14, split_data=None):

    (
        X_TRAIN,
        X_VAL,
        Y_TRAIN,
        Y_VAL,
        no_feature_cols,
        X_TEST,
        Y_TEST,
        x_boolmat_test,
        y_boolmat_test,
        x_boolmat_val,
        y_boolmat_val,
        features,
    ) = return_data(
        FILE,
        return_cols=True,
        balancer=True,
        target=target,
        pad=True,
        split=True,
        time_steps=time_steps,
        split_data=split_data,
    )

    f = open("./pickled_objects/X_TRAIN_{0}.txt".format(target), "wb")
    pickle.dump(X_TRAIN, f)
    f.close()

    f = open("./pickled_objects/X_VAL_{0}.txt".format(target), "wb")
    pickle.dump(X_VAL, f)
    f.close()

    f = open("./pickled_objects/Y_TRAIN_{0}.txt".format(target), "wb")
    pickle.dump(Y_TRAIN, f)
    f.close()

    f = open("./pickled_objects/Y_VAL_{0}.txt".format(target), "wb")
    pickle.dump(Y_VAL, f)
    f.close()

    f = open("./pickled_objects/X_TEST_{0}.txt".format(target), "wb")
    pickle.dump(X_TEST, f)
    f.close()

    f = open("./pickled_objects/Y_TEST_{0}.txt".format(target), "wb")
    pickle.dump(Y_TEST, f)
    f.close()

    f = open("./pickled_objects/x_boolmat_test_{0}.txt".format(target), "wb")
    pickle.dump(x_boolmat_test, f)
    f.close()

    f = open("./pickled_objects/y_boolmat_test_{0}.txt".format(target), "wb")
    pickle.dump(y_boolmat_test, f)
    f.close()

    f = open("./pickled_objects/x_boolmat_val_{0}.txt".format(target), "wb")
    pickle.dump(x_boolmat_val, f)
    f.close()

    f = open("./pickled_objects/y_boolmat_val_{0}.txt".format(target), "wb")
    pickle.dump(y_boolmat_val, f)
    f.close()

    f = open("./pickled_objects/no_feature_cols_{0}.txt".format(target), "wb")
    pickle.dump(no_feature_cols, f)
    f.close()

    f = open("./pickled_objects/features_{0}.txt".format(target), "wb")
    pickle.dump(features, f)
    f.close()


def run_trainer(build_model_fn, split_data_fn):

    if not os.path.exists("./pickled_objects"):
        os.makedirs("./pickled_objects")

    """
    pickle_objects(target='MI', time_steps=14, split_data=split_data_fn)
    pickle_objects(target='SEPSIS', time_steps=14, split_data=split_data_fn)
    pickle_objects(target='VANCOMYCIN', time_steps=14, split_data=split_data_fn)
    """

    print("##### Training Myocardian Infarction Model #####")
    train(
        model_name="MI",
        epochs=13,
        synth_data=False,
        predict=True,
        target="MI",
        time_steps=14,
        build_model=build_model_fn,
    )

    print("##### Training Vancomycin Model #####")
    train(
        model_name="VANCOMYCIN",
        epochs=14,
        synth_data=False,
        predict=True,
        target="VANCOMYCIN",
        time_steps=14,
        build_model=build_model_fn,
    )

    print("##### Training Sepsis Model #####")
    train(
        model_name="SEPSIS",
        epochs=17,
        synth_data=False,
        predict=True,
        target="SEPSIS",
        time_steps=14,
        build_model=build_model_fn,
    )
