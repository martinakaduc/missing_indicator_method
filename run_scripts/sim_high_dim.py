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
from joblib import Parallel, delayed
import itertools
from collections import Counter
import argparse

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
import sys
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
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
from tabular_utils import (
    time_fit, get_dataset_details, time_fit_transform, 
    gen_low_rank_data, make_regression, 
    make_block_diagonal_regression,
)


def get_model_name(model):
    if model in [LogisticRegression, LinearRegression]:
        return "linear"
    if model in [XGBClassifier, XGBRegressor]:
        return "xgb"
    if model in [MLPClassifier, MLPRegressor]:
        return "mlp"


def gen_results(n, p, imputer_name, n_blocks=10, inf_prob=0.5, seed=10, alpha=0.05):

    print(f"RESULTS FOR inf_prob={inf_prob}, seed={seed}")

    # Make regression from block diagonal covariance
    block_size = p // n_blocks
    X, y, latent_X = make_block_diagonal_regression(
        n = n,
        p = p,
        block_size = block_size,
        block_corr = 0.5,
        seed = seed,
    )
    (
        X_train, X_test, 
        latent_X_train, latent_X_test,
        y_train, y_test 
    ) = train_test_split(X, latent_X, y, test_size=0.25, random_state=seed)

    # Get models for regression case using some regression dataset
    _, models, model_params_sets = get_dataset_details("housing")


    # Generate mask using latent X (mean by block)
    rng = np.random.default_rng(seed=seed)
    power = rng.choice([0, 2], p=[1-inf_prob, inf_prob], size=n_blocks)

    # MNAR Mask 
    train_probs, _ = MNAR_mask(latent_X_train, side="right", power=power, seed=seed, return_na=True, standardize=True)
    train_probs = np.repeat(train_probs, repeats=block_size, axis=1)
    train_mask = rng.binomial(1, train_probs, size=train_probs.shape)
    train_mask = np.where(train_mask == 0, np.nan, train_mask)
    X_train_masked = X_train * train_mask

    test_probs, _ = MNAR_mask(latent_X_test, side="right", power=power, seed=seed, return_na=True, standardize=True)
    test_probs = np.repeat(test_probs, repeats=block_size, axis=1)
    test_mask = rng.binomial(1, test_probs, size=test_probs.shape)
    test_mask = np.where(test_mask == 0, np.nan, test_mask)
    X_test_masked = X_test * test_mask

    mim = MIM(features="dynamic", alpha=0.1)
    mim.fit(X_train_masked, y_train)
    indicators = mim.mask
    used_indicator = [int(i in mim.features_) for i in range(p) ]
    matched = list(zip(np.repeat(power, repeats=block_size), used_indicator))

    if imputer_name == "mean":
        imputer = make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="constant", fill_value=0),
        )
    elif imputer_name == "gc":
        imputer = make_pipeline(
            StandardScaler(),
            GCImputer(random_state=seed, min_ord_ratio=np.inf, max_iter=50, n_jobs=1),
        )
    elif imputer_name == "lrgc":
        imputer = make_pipeline(
            StandardScaler(),
            LRGCImputer(random_state=seed, rank=10, min_ord_ratio=np.inf, max_iter=50, n_jobs=1),
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
            results.append([seed, alpha, inf_prob, imputer_name, model_name, "No_MIM", None, None])
            results.append([seed, alpha, inf_prob, imputer_name, model_name, "MIM", None, None])
            results.append([seed, alpha, inf_prob, imputer_name, model_name, "Dynamic_MIM", None, None])
            results.append([seed, alpha, inf_prob, imputer_name, model_name, "Oracle_MIM", None, None])
        return results

    X_test_imputed = imputer.transform(X_test_masked)

    metric = mean_squared_error
    metric_kwargs = {"squared": False}

    # Evaluate each model for no MIM, MIM, and dynanic MIM
    results = []
    for model, model_params in zip(models, model_params_sets):
        model_name = get_model_name(model)
        # print(model_name)
        if model != LinearRegression:
            model_params["random_state"] = seed

        # Performance for no MIM
        model_ = model(**model_params)
        no_mim_time = time_fit(model_, X_train_imputed, y_train)

        no_mim_score = metric(y_test, model_.predict(X_test_imputed), **metric_kwargs)
        results.append([seed, alpha, inf_prob, imputer_name, model_name, "No_MIM", no_mim_score, impute_time + no_mim_time])

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
        results.append([seed, alpha, inf_prob, imputer_name, model_name, "MIM", mim_score, impute_time + mim_time])

        # Performance for dynamic MIM
        dynamic_mim = MIM(features="dynamic", alpha=alpha)

        train_mask_feats = dynamic_mim.fit_transform(X_train_masked, y_train)
        test_mask_feats = dynamic_mim.transform(X_test_masked)

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
        results.append([seed, alpha, inf_prob, imputer_name, model_name, "Dynamic_MIM", dynamic_mim_score, impute_time + dynamic_mim_time])


        # Performance for Oracle MIM
        oracle_mim = Oracle_MIM(
            indicators_to_keep=np.flatnonzero(np.repeat(power, block_size))
        )

        train_mask_feats = oracle_mim.fit_transform(X_train_masked, y_train)
        test_mask_feats = oracle_mim.transform(X_test_masked)

        X_train_input = np.hstack((
            X_train_imputed,
            train_mask_feats,
        ))
        X_test_input = np.hstack((
            X_test_imputed,
            test_mask_feats,
        ))
        model_ = model(**model_params)
        oracle_mim_time = time_fit(model_, X_train_input, y_train)
        oracle_mim_score = metric(y_test, model_.predict(X_test_input), **metric_kwargs)
        results.append([seed, alpha, inf_prob, imputer_name, model_name, "Oracle_MIM", oracle_mim_score, impute_time + oracle_mim_time])

    return results


parser = argparse.ArgumentParser()
parser.add_argument("--imputer", type=str)
parser.add_argument("--n_jobs", type=int, default=1)
args = parser.parse_args()

n = 1000
p = 1000

n_trials = 20
seeds = range(10, 10 + n_trials)
inf_probs = np.arange(0, 11) / 10
imputer = args.imputer
alpha = 0.1
n_blocks = 20

@ignore_warnings(category=ConvergenceWarning)
def test():
    results = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(gen_results)(n, p, imputer, n_blocks=n_blocks, inf_prob=inf_prob, seed=seed, alpha=alpha)
        for seed, inf_prob in itertools.product(seeds, inf_probs)
    )
    return results

results = test()
results = sum(results, [])

results = pd.DataFrame(results, columns=["Seed", "Alpha", "Inf_Prob", "Imputer", "Model", "MIM", "Score", "Time"])

results.to_csv("sim_outputs/sim_dynamic_mim_block_{imputer}.csv", index=False)