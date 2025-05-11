import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_classification
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import sys

sys.path.append("../utils")
sys.path.append("../models")
from imputation_utils import simple_mask, MNAR_mask
from imputation_models import GCImputer, LRGCImputer, MIM, Oracle_MIM
from tabular_utils import (
    time_fit,
    time_fit_transform,
    make_val_split,
    get_dataset_details,
    load_dataset,
    get_gc_type,
)

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["OMP_WARNINGS"] = "FALSE"


def get_model_name(model):
    if model in [LogisticRegression, LinearRegression]:
        return "linear"
    if model in [XGBClassifier, XGBRegressor]:
        return "xgb"
    if model in [MLPClassifier, MLPRegressor]:
        return "mlp"


def openml_scores(
    *,
    imputer_name,
    seed=10,
    alpha=0.05,
):

    print(f"RUNNING LOS FOR seed={seed}, imputer={imputer_name}")

    # Read in mimic mortality data
    train_data = pd.read_csv("../mimic_data/los_tab_train.csv")
    X_train = train_data[[c for c in train_data.columns if c != "target"]].copy(
        deep=False
    )
    X_train = X_train[[c for c in X_train.columns if "full" in c and "mean" in c]]
    y_train = train_data["target"].copy(deep=False)

    # Binary classification only on if length of stay was long (>= 7 days)
    y_train = np.where(y_train > 6, 1, 0)

    test_data = pd.read_csv("../mimic_data/los_tab_test.csv")
    X_test = test_data[[c for c in test_data.columns if c != "target"]].copy(deep=False)
    X_test = X_test[[c for c in X_test.columns if "full" in c and "mean" in c]]
    y_test = test_data["target"].copy(deep=False)

    y_test = np.where(y_test > 6, 1, 0)

    train_mask = np.where(np.isnan(X_train), 1, 0)
    test_mask = np.where(np.isnan(X_test), 1, 0)

    # Gets tasks and models for multiclass task
    # task, models, model_params_sets = get_dataset_details("volkert")
    task, models, model_params_sets = get_dataset_details("higgs")

    if imputer_name == "mean":
        imputer = make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="constant", fill_value=0),
        )
    elif imputer_name == "gc":
        imputer = make_pipeline(
            StandardScaler(),
            GCImputer(random_state=seed, min_ord_ratio=np.inf, max_iter=50),
        )
    elif imputer_name == "lrgc":
        imputer = make_pipeline(
            StandardScaler(),
            LRGCImputer(random_state=seed, rank=10, min_ord_ratio=np.inf, max_iter=50),
        )
    elif imputer_name == "mf":
        imputer = make_pipeline(
            StandardScaler(),
            IterativeImputer(estimator=RandomForestRegressor(random_state=seed)),
        )
    else:
        raise ValueError("imputer_name must be one of mean, gc, or mf")

    X_train_imputed, impute_time = time_fit_transform(
        imputer, X_train, y_train, time_limit=60 * 60 * 3
    )

    # Is imputer didn't finish, report none results
    if X_train_imputed is None:
        results = []
        for model, model_params in zip(models, model_params_sets):
            model_name = get_model_name(model)
            results.append(
                [seed, alpha, imputer_name, model_name, "No_MIM", None, None, None]
            )
            results.append(
                [seed, alpha, imputer_name, model_name, "MIM", None, None, None]
            )
            results.append(
                [seed, alpha, imputer_name, model_name, "SMIM", None, None, None]
            )
        print(f"FINISHING LOS FOR seed={seed}, imputer={imputer_name}")
        return results

    X_test_imputed = imputer.transform(X_test)

    if task == "regression":
        metric = mean_squared_error
        metric_kwargs = {"squared": False}
    elif task == "binary":
        metric = roc_auc_score
        metric_kwargs = {}
    elif task == "multiclass":
        metric = accuracy_score
        metric_kwargs = {}
    else:
        raise ValueError(
            f"task must be one of regression, binary, multiclass, got {task}"
        )

    # Evaluate each model for no MIM, MIM, and dynanic MIM
    results = []
    for model, model_params in zip(models, model_params_sets):
        model_name = get_model_name(model)
        if model != LinearRegression:
            model_params["random_state"] = seed

        # # Performance for no MIM
        # model_ = model(**model_params)
        # no_mim_time = time_fit(model_, X_train_imputed, y_train)
        # no_mim_score = metric(y_test, model_.predict(X_test_imputed), **metric_kwargs)
        # results.append([seed, alpha, imputer_name, model_name, "No_MIM", no_mim_score, impute_time, no_mim_time])

        # # Performance for normal MIM
        # X_train_input = np.hstack((
        #     X_train_imputed,
        #     train_mask,
        # ))
        # X_test_input = np.hstack((
        #     X_test_imputed,
        #     test_mask,
        # ))
        # model_ = model(**model_params)
        # mim_time = time_fit(model_, X_train_input, y_train)
        # mim_score = metric(y_test, model_.predict(X_test_input), **metric_kwargs)
        # results.append([seed, alpha, imputer_name, model_name, "MIM", mim_score, impute_time, mim_time])

        # # Performance for SMIM
        # smim = MIM(features="dynamic", alpha=alpha)

        # train_mask_feats = smim.fit_transform(X_train, y_train)
        # test_mask_feats = smim.transform(X_test)

        # X_train_input = np.hstack((
        #     X_train_imputed,
        #     train_mask_feats,
        # ))
        # X_test_input = np.hstack((
        #     X_test_imputed,
        #     test_mask_feats,
        # ))
        # model_ = model(**model_params)
        # smim_time = time_fit(model_, X_train_input, y_train)
        # smim_score = metric(y_test, model_.predict(X_test_input), **metric_kwargs)
        # results.append([seed, alpha, imputer_name, model_name, "SMIM", smim_score, impute_time, smim_time])

        # Performance for no MIM
        model_ = model(**model_params)
        no_mim_time = time_fit(model_, X_train_imputed, y_train)
        no_mim_score = metric(
            y_test, model_.predict_proba(X_test_imputed)[:, 1], **metric_kwargs
        )
        results.append(
            [
                seed,
                alpha,
                imputer_name,
                model_name,
                "No_MIM",
                no_mim_score,
                impute_time,
                no_mim_time,
            ]
        )

        # Performance for normal MIM
        X_train_input = np.hstack(
            (
                X_train_imputed,
                train_mask,
            )
        )
        X_test_input = np.hstack(
            (
                X_test_imputed,
                test_mask,
            )
        )
        model_ = model(**model_params)
        mim_time = time_fit(model_, X_train_input, y_train)
        mim_score = metric(
            y_test, model_.predict_proba(X_test_input)[:, 1], **metric_kwargs
        )
        results.append(
            [
                seed,
                alpha,
                imputer_name,
                model_name,
                "MIM",
                mim_score,
                impute_time,
                mim_time,
            ]
        )

        # Performance for SMIM
        smim = MIM(features="dynamic", alpha=alpha)

        train_mask_feats = smim.fit_transform(X_train, y_train)
        test_mask_feats = smim.transform(X_test)

        X_train_input = np.hstack(
            (
                X_train_imputed,
                train_mask_feats,
            )
        )
        X_test_input = np.hstack(
            (
                X_test_imputed,
                test_mask_feats,
            )
        )
        model_ = model(**model_params)
        smim_time = time_fit(model_, X_train_input, y_train)
        smim_score = metric(
            y_test, model_.predict_proba(X_test_input)[:, 1], **metric_kwargs
        )
        results.append(
            [
                seed,
                alpha,
                imputer_name,
                model_name,
                "SMIM",
                smim_score,
                impute_time,
                smim_time,
            ]
        )

    print(f"FINISHING LOS FOR seed={seed}, imputer={imputer_name}")
    return results


dataset_name = "los"
# imputers = ["mean", "gc", "mf"]
imputers = ["mean", "gc"]
n_trials = 20
seeds = range(10, 10 + n_trials)
# seeds = [10]


@ignore_warnings(category=ConvergenceWarning)
def run():
    results = Parallel(n_jobs=60, backend="multiprocessing")(
        delayed(openml_scores)(
            imputer_name=imputer,
            seed=seed,
            alpha=0.1,
        )
        for seed, imputer in itertools.product(seeds, imputers)
    )
    return results


results = run()
results = sum(results, [])

df = pd.DataFrame(
    results,
    columns=[
        "Seed",
        "Alpha",
        "Imputer",
        "Model",
        "MIM",
        "Score",
        "Impute_Time",
        "Model_Time",
    ],
)

df.to_csv(
    f"/data/jmv249/Informative_Missingness/mimic_outputs/los_binary.csv", index=False
)
