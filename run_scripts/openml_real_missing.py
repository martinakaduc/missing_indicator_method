import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
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
from tabular_utils import time_fit, time_fit_transform, make_val_split, get_dataset_details, load_dataset, get_gc_type

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
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
    dataset_name,
    seed=10,
    alpha=0.05,
):

    print(f"RESULTS FOR seed={seed}")

    X, y = load_dataset(dataset_name=dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    train_mask = np.where(np.isnan(X_train), 1, 0)
    test_mask = np.where(np.isnan(X_test), 1, 0)

    task, models, model_params_sets = get_dataset_details(dataset_name)


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
    elif imputer_name == "categorical":
        print("THIS ISNT IMPLEMENTED YET")
        return
        # num_feats = [c for c in X.columns if X[c].dtype.name != "category"]
        # cat_feats = [c for c in X.columns if X[c].dtype.name == "category"]

        # num_imputer = make_pipeline(
        #     StandardScaler(),
        #     SimpleImputer(strategy="constant", fill_value=0),
        # )
        # cat_imputer = make_pipeline(

        # )

        # imputer = make_pipeline(
        #     make_column_transformer(
        #         ()
        #     )
        # )
    else:
        raise ValueError("imputer_name must be one of mean, gc, or mf")


    X_train_imputed, impute_time = time_fit_transform(imputer, X_train, y_train, time_limit=60*60*3)

    # Is imputer didn't finish, report none results
    if X_train_imputed is None:
        results = []
        for model, model_params in zip(models, model_params_sets):
            model_name = get_model_name(model)
            results.append([seed, alpha, imputer_name, model_name, "No_MIM", None, None, None])
            results.append([seed, alpha, imputer_name, model_name, "MIM", None, None, None])
            results.append([seed, alpha, imputer_name, model_name, "Dynamic_MIM", None, None, None])
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
        raise ValueError(f"task must be one of regression, binary, multiclass, got {task}")

    # Evaluate each model for no MIM, MIM, and dynanic MIM
    results = []
    for model, model_params in zip(models, model_params_sets):
        model_name = get_model_name(model)
        if model != LinearRegression:
            model_params["random_state"] = seed

        if model == LinearRegression and dataset_name in ["crime", "meta"]:
            model = Ridge
            model_params = {
                "random_state": seed,
                "alpha": 0.0001
            }


        # Performance for no MIM
        model_ = model(**model_params)
        no_mim_time = time_fit(model_, X_train_imputed, y_train)
        no_mim_score = metric(y_test, model_.predict(X_test_imputed), **metric_kwargs)
        results.append([seed, alpha, imputer_name, model_name, "No_MIM", no_mim_score, impute_time, no_mim_time])

        # Performance for normal MIM
        X_train_input = np.hstack((
            X_train_imputed,
            train_mask,
        ))
        X_test_input = np.hstack((
            X_test_imputed,
            test_mask,
        ))
        model_ = model(**model_params)
        mim_time = time_fit(model_, X_train_input, y_train)
        mim_score = metric(y_test, model_.predict(X_test_input), **metric_kwargs)
        results.append([seed, alpha, imputer_name, model_name, "MIM", mim_score, impute_time, mim_time])

        # Performance for dynamic MIM
        dynamic_mim = MIM(features="dynamic", alpha=alpha)

        train_mask_feats = dynamic_mim.fit_transform(X_train, y_train)
        test_mask_feats = dynamic_mim.transform(X_test)

        X_train_input = np.hstack((
            X_train_imputed,
            train_mask_feats,
        ))
        X_test_input = np.hstack((
            X_test_imputed,
            test_mask_feats,
        ))
        model_ = model(**model_params)
        dynamic_mim_time = time_fit(model_, X_train_input, y_train)
        dynamic_mim_score = metric(y_test, model_.predict(X_test_input), **metric_kwargs)
        results.append([seed, alpha, imputer_name, model_name, "Dynamic_MIM", dynamic_mim_score, impute_time, dynamic_mim_time])

    return results



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--n_jobs", type=int)
args = parser.parse_args()

dataset_name = args.dataset
imputers = ["mean", get_gc_type(dataset_name), "mf"]
# imputers = ["mean"]
n_trials = 20
seeds = range(10, 10+n_trials)

@ignore_warnings(category=ConvergenceWarning)
def run():
    results = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(openml_scores)(
            imputer_name=imputer, 
            dataset_name=dataset_name,
            seed=seed,
            alpha=0.1,
        ) 
        for seed, imputer in itertools.product(seeds, imputers)
    )
    return results

results = run()
results = sum(results, [])

df = pd.DataFrame(results, columns=["Seed", "Alpha", "Imputer", "Model", "MIM", "Score", "Impute_Time", "Model_Time"])

df.to_csv(f"openml_outputs/{dataset_name}_real_missing.csv", index=False)