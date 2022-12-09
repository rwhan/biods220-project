"""
Based of the implementation of "An attention based deep learning model of clinical events in the intensive care unit".

Credit: DA Kaji (https://github.com/deepak-kaji/mimic-lstm)

"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import kurtosis
from seaborn import heatmap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model


def get_reduced_features_and_indices(features):
    ## FEATURES BY CATEGORY ##

    cbc_diff_features = [
        "RBCs",
        "WBCs",
        "platelets",
        "hemoglobin",
        "hemocrit",
        "atypical lymphocytes",
        "bands",
        "basophils",
        "eosinophils",
        "neutrophils",
        "lymphocytes",
        "monocytes",
        "polymorphonuclear leukocytes",
    ]

    vital_features = [
        "temperature (F)",
        "heart rate",
        "respiratory rate",
        "systolic",
        "diastolic",
        "pulse oximetry",
    ]

    lab_features = [
        "troponin",
        "HDL",
        "LDL",
        "BUN",
        "INR",
        "PTT",
        "PT",
        "triglycerides",
        "creatinine",
        "glucose",
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "blood culture",
        "urine culture",
        "surface culture",
        "sputum" + " culture",
        "wound culture",
        "Inspired O2 Fraction",
        "central venous pressure",
        "PEEP Set",
        "tidal volume",
        "anion gap",
    ]

    demographic_features = [
        "age",
        "m",
        "black",
        "daily weight",
        "tobacco",
        "diabetes",
        "history of CV events",
    ]

    med_features = [
        "epoetin",
        "warfarin",
        "heparin",
        "enoxaparin",
        "fondaparinux",
        "asprin",
        "ketorolac",
        "acetominophen",
        "insulin",
        "glucagon",
        "potassium_med",
        "calcium gluconate",
        "fentanyl",
        "magensium sulfate",
        "D5W",
        "dextrose",
        "ranitidine",
        "ondansetron",
        "pantoprazole",
        "metoclopramide",
        "lisinopril",
        "captopril",
        "statin",
        "hydralazine",
        "diltiazem",
        "carvedilol",
        "metoprolol",
        "labetalol",
        "atenolol",
        "amiodarone",
        "digoxin",
        "clopidogrel",
        "nitroprusside",
        "nitroglycerin",
        "vasopressin",
        "hydrochlorothiazide",
        "furosemide",
        "atropine",
        "neostigmine",
        "levothyroxine",
        "oxycodone",
        "hydromorphone",
        "fentanyl citrate",
        "tacrolimus",
        "prednisone",
        "phenylephrine",
        "norepinephrine",
        "haloperidol",
        "phenytoin",
        "trazodone",
        "levetiracetam",
        "diazepam",
        "clonazepam",
        "propofol",
        "zolpidem",
        "midazolam",
        "albuterol",
        "ipratropium",
        "diphenhydramine",
        "0.9% Sodium Chloride",
        "phytonadione",
        "metronidazole",
        "cefazolin",
        "cefepime",
        "vancomycin",
        "levofloxacin",
        "ciprofloxacin",
        "fluconazole",
        "meropenem",
        "ceftriaxone",
        "piperacillin",
        "ampicillin-sulbactam",
        "nafcillin",
        "oxacillin",
        "amoxicillin",
        "penicillin",
        "SMX-TMP",
    ]

    cbc_diff_features = [
        [i.lower(), i.lower() + "_min", i.lower() + "_max", i.lower() + "_std"]
        for i in cbc_diff_features
    ]
    vital_features = [
        [i.lower(), i.lower() + "_min", i.lower() + "_max", i.lower() + "_std"]
        for i in vital_features
    ]
    lab_features = [
        [i.lower(), i.lower() + "_min", i.lower() + "_max", i.lower() + "_std"]
        for i in lab_features
    ]
    demographic_features = [i.lower() for i in demographic_features]
    med_features = [i.lower() for i in med_features]

    cbc_diff_feature_array = np.array(cbc_diff_features).flatten()
    vital_features_array = np.array(vital_features).flatten()
    lab_features_array = np.array(lab_features).flatten()
    demographic_feature_array = np.array(demographic_features).flatten()
    med_features_array = np.array(med_features).flatten()
    features_built = np.hstack(
        [
            cbc_diff_feature_array,
            vital_features_array,
            lab_features_array,
            demographic_feature_array,
            med_features_array,
        ]
    )
    features_built_reduced = [i for i in features_built if i in features]

    ## Identifies the index in the features list in the desired order ##
    arranged_indices = [features.index(i) for i in features_built_reduced]
    return features_built_reduced, arranged_indices


def return_loaded_model(model_name=""):
    loaded_model = load_model("./saved_models/{0}.h5".format(model_name))
    return loaded_model


def load_data_and_model(target):

    X_TRAIN_MI = pickle.load(open("./pickled_objects/X_TRAIN_MI.txt", "rb"))
    X_TRAIN_SEPSIS = pickle.load(open("./pickled_objects/X_TRAIN_SEPSIS.txt", "rb"))
    X_TRAIN_VANCOMYCIN = pickle.load(
        open("./pickled_objects/X_TRAIN_VANCOMYCIN.txt", "rb")
    )

    Y_TRAIN_MI = pickle.load(open("./pickled_objects/Y_TRAIN_MI.txt", "rb"))
    Y_TRAIN_SEPSIS = pickle.load(open("./pickled_objects/Y_TRAIN_SEPSIS.txt", "rb"))
    Y_TRAIN_VANCOMYCIN = pickle.load(
        open("./pickled_objects/Y_TRAIN_VANCOMYCIN.txt", "rb")
    )

    Y_VAL_MI = pickle.load(open("./pickled_objects/Y_VAL_MI.txt", "rb"))
    Y_VAL_SEPSIS = pickle.load(open("./pickled_objects/Y_VAL_SEPSIS.txt", "rb"))
    Y_VAL_VANCOMYCIN = pickle.load(open("./pickled_objects/Y_VAL_VANCOMYCIN.txt", "rb"))

    X_VAL_MI = pickle.load(open("./pickled_objects/X_VAL_MI.txt", "rb"))
    X_VAL_SEPSIS = pickle.load(open("./pickled_objects/X_VAL_SEPSIS.txt", "rb"))
    X_VAL_VANCOMYCIN = pickle.load(open("./pickled_objects/X_VAL_VANCOMYCIN.txt", "rb"))

    Y_TEST_MI = pickle.load(open("./pickled_objects/Y_TEST_MI.txt", "rb"))
    Y_TEST_SEPSIS = pickle.load(open("./pickled_objects/Y_TEST_SEPSIS.txt", "rb"))
    Y_TEST_VANCOMYCIN = pickle.load(
        open("./pickled_objects/Y_TEST_VANCOMYCIN.txt", "rb")
    )

    X_TEST_MI = pickle.load(open("./pickled_objects/X_TEST_MI.txt", "rb"))
    X_TEST_SEPSIS = pickle.load(open("./pickled_objects/X_TEST_SEPSIS.txt", "rb"))
    X_TEST_VANCOMYCIN = pickle.load(
        open("./pickled_objects/X_TEST_VANCOMYCIN.txt", "rb")
    )

    y_boolmat_test_MI = pickle.load(
        open("./pickled_objects/y_boolmat_test_MI.txt", "rb")
    )
    y_boolmat_test_SEPSIS = pickle.load(
        open("./pickled_objects/y_boolmat_test_SEPSIS.txt", "rb")
    )
    y_boolmat_test_VANCOMYCIN = pickle.load(
        open("./pickled_objects/y_boolmat_test_VANCOMYCIN.txt", "rb")
    )

    x_boolmat_test_MI = pickle.load(
        open("./pickled_objects/x_boolmat_test_MI.txt", "rb")
    )
    x_boolmat_test_SEPSIS = pickle.load(
        open("./pickled_objects/x_boolmat_test_SEPSIS.txt", "rb")
    )
    x_boolmat_test_VANCOMYCIN = pickle.load(
        open("./pickled_objects/x_boolmat_test_VANCOMYCIN.txt", "rb")
    )

    no_features_cols_MI = pickle.load(
        open("./pickled_objects/no_feature_cols_MI.txt", "rb")
    )
    no_features_cols_SEPSIS = pickle.load(
        open("./pickled_objects/no_feature_cols_SEPSIS.txt", "rb")
    )
    no_features_cols_VANCOMYCIN = pickle.load(
        open("./pickled_objects/no_feature_cols_VANCOMYCIN.txt", "rb")
    )

    features_MI = pickle.load(open("./pickled_objects/features_MI.txt", "rb"))
    features_SEPSIS = pickle.load(open("./pickled_objects/features_SEPSIS.txt", "rb"))
    features_VANCOMYCIN = pickle.load(
        open("./pickled_objects/features_VANCOMYCIN.txt", "rb")
    )

    if target == "MI":
        my_cmap = ListedColormap(sns.color_palette("Reds", 150))
        color_list = sns.color_palette("Reds", 14)
        color_list_reduced = sns.color_palette("Reds", 7)
        X_TRAIN = X_TRAIN_MI
        X_VAL = X_VAL_MI
        Y_TRAIN = Y_TRAIN_MI
        Y_VAL = Y_VAL_MI
        Y_TEST = Y_TEST_MI
        X_TEST = X_TEST_MI
        y_boolmat_test = y_boolmat_test_MI
        x_boolmat_test = x_boolmat_test_MI
        features = features_MI

    elif target == "SEPSIS":
        my_cmap = sns.cubehelix_palette(
            14, start=2, rot=0, dark=0.25, light=0.95, as_cmap=True
        )
        color_list = sns.cubehelix_palette(14, start=2, rot=0, dark=0.15, light=0.8)
        color_list_reduced = sns.cubehelix_palette(
            7, start=2, rot=0, dark=0.15, light=0.8
        )
        X_TRAIN = X_TRAIN_SEPSIS
        X_VAL = X_VAL_SEPSIS
        Y_TRAIN = Y_TRAIN_SEPSIS
        Y_VAL = Y_VAL_SEPSIS
        Y_TEST = Y_TEST_SEPSIS
        X_TEST = X_TEST_SEPSIS
        y_boolmat_test = y_boolmat_test_SEPSIS
        x_boolmat_test = x_boolmat_test_SEPSIS
        features = features_SEPSIS

    elif target == "VANCOMYCIN":
        my_cmap = sns.cubehelix_palette(14, as_cmap=True)
        color_list = sns.cubehelix_palette(14)
        color_list_reduced = sns.cubehelix_palette(7)
        X_TRAIN = X_TRAIN_VANCOMYCIN
        X_VAL = X_VAL_VANCOMYCIN
        Y_TRAIN = Y_TRAIN_VANCOMYCIN
        Y_VAL = Y_VAL_VANCOMYCIN
        Y_TEST = Y_TEST_VANCOMYCIN
        X_TEST = X_TEST_VANCOMYCIN
        y_boolmat_test = y_boolmat_test_VANCOMYCIN
        x_boolmat_test = x_boolmat_test_VANCOMYCIN
        features = features_VANCOMYCIN

    # Y_TRAIN[Y_TRAIN == -1] = np.nan
    # Y_VAL[Y_VAL == -1] = np.nan
    # Y_TEST[Y_TEST == -1] = np.nan
    Y_TOTAL = np.concatenate([Y_TRAIN, Y_VAL, Y_TEST], axis=0)

    Y_MI = np.concatenate([Y_TRAIN_MI, Y_VAL_MI], axis=0)
    Y_SEPSIS = np.concatenate([Y_TRAIN_SEPSIS, Y_VAL_SEPSIS], axis=0)
    Y_VANCOMYCIN = np.concatenate([Y_TRAIN_VANCOMYCIN, Y_VAL_VANCOMYCIN], axis=0)

    TIME_STEPS = X_VAL.shape[1]  # number of time_steps

    m = return_loaded_model(model_name=target)

    """ Due to the way features are selectd from the EMR and the fact potassium can be a 
    delivered medication or a lab value, special care was taken to ensure proper representation on heatmaps """

    if "digoxin(?!.*fab)" in features:
        indexy = features.index("digoxin(?!.*fab)")
        features[indexy] = "digoxin"

    if "potassium_y" in features:
        indexy = features.index("potassium_y")
        features[indexy] = "potassium_med"

    if "potassium_x" in features:
        indexy = features.index("potassium_x")
        features[indexy] = "potassium"

    if "cipfloxacin" in features:
        indexy = features.index("cipfloxacin")
        features[indexy] = "ciprofloxacin"

    features = [feature.lower() for feature in features]
    return X_TEST, x_boolmat_test, Y_TEST, y_boolmat_test, features, m
