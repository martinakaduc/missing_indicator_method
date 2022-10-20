import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_classification, make_low_rank_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
import sys
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["OMP_WARNINGS"] = "FALSE"

sys.path.append("../utils")
sys.path.append("../models")
from imputation_utils import simple_mask, MNAR_mask
from imputation_models import GCImputer, LRGCImputer, MIM, Oracle_MIM
from tabular_utils import time_fit, get_dataset_details, time_fit_transform, gen_low_rank_data, make_regression


def get_model_name(model):
    if model in [LogisticRegression, LinearRegression]:
        return "linear"
    if model in [XGBClassifier, XGBRegressor]:
        return "xgb"
    if model in [MLPClassifier, MLPRegressor]:
        return "mlp"


def gen_results(n, p, imputer_name, power=1, seed=10):

    print(f"RESULTS FOR power={power}, seed={seed}, imputer={imputer_name}")

    power = np.ones(p) * power

    cov = np.ones(shape=(p, p)) * 0.3
    np.fill_diagonal(cov, 1)
    X, y = make_regression(n=n, p=p, seed=seed, cov=cov)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    # Get models for regression case using some regression dataset
    _, models, model_params_sets = get_dataset_details("housing")

    # MNAR Mask 
    _, train_mask = MNAR_mask(X_train, side="right", power=power, seed=seed, return_na=True, standardize=True)
    X_train_masked = X_train * train_mask

    _, test_mask = MNAR_mask(X_test, side="right", power=power, seed=seed, return_na=True, standardize=True)
    X_test_masked = X_test * test_mask


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

    X_train_imputed, impute_time = time_fit_transform(imputer, X_train_masked, y_train)

    # Is imputer didn't finish, report none results
    if X_train_imputed is None:
        results = []
        for model, model_params in zip(models, model_params_sets):
            model_name = get_model_name(model)
            results.append([seed, power[0], imputer_name, model_name, "No_MIM", None, None, None])
            results.append([seed, power[0], imputer_name, model_name, "MIM", None, None, None])
        return results

    X_test_imputed = imputer.transform(X_test_masked)

    metric = mean_squared_error
    metric_kwargs = {"squared": False}

    # Evaluate each model for no MIM, MIM, and dynanic MIM
    results = []
    for model, model_params in zip(models, model_params_sets):
        model_name = get_model_name(model)
        if model != LinearRegression:
            model_params["random_state"] = seed

        # Performance for no MIM
        model_ = model(**model_params)
        no_mim_time = time_fit(model_, X_train_imputed, y_train)
        no_mim_score = metric(y_test, model_.predict(X_test_imputed), **metric_kwargs)
        results.append([seed, power[0], imputer_name, model_name, "No_MIM", no_mim_score, impute_time, no_mim_time])

        # Performance for normal MIM
        train_mask_feats = np.where(np.isnan(train_mask), 1, 0)
        test_mask_feats = np.where(np.isnan(test_mask), 1, 0)
        X_train_input = np.hstack((
            X_train_imputed,
            train_mask_feats,
        ))
        X_test_input = np.hstack((
            X_test_imputed,
            test_mask_feats,
        ))
        model_ = model(**model_params)
        mim_time = time_fit(model_, X_train_input, y_train)
        mim_score = metric(y_test, model_.predict(X_test_input), **metric_kwargs)
        results.append([seed, power[0], imputer_name, model_name, "MIM", mim_score, impute_time, mim_time])

    return results



n = 10000
p = 10

n_trials = 20
seeds = range(10, 10 + n_trials)
powers = np.linspace(0, 5, 25)
imputers = ["mean", "gc", "mf"]


@ignore_warnings(category=ConvergenceWarning)
def test():
    results = Parallel(n_jobs=50, backend="multiprocessing")(
        delayed(gen_results)(n, p, imputer, power=power, seed=seed)
        for seed, power, imputer in itertools.product(seeds, powers, imputers)
    )
    return results

results = test()

results = sum(results, [])

results = pd.DataFrame(results, columns=["Seed", "Power", "Imputer", "Model", "MIM", "Score", "Impute_Time", "Model_Time"])

results.to_csv("sim_outputs/sim_low_dim_reg.csv", index=False)