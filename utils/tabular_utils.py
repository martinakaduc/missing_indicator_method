import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.datasets import fetch_openml, make_low_rank_matrix
from time import time
import multiprocessing
import signal
from scipy.linalg import block_diag

from xgboost import XGBRegressor, XGBClassifier



def make_classifier(n, p, k, seed=10):
    rng = np.random.default_rng(seed)
    if p > k:
        A = rng.normal(0, 1, size=(k, k))
        cov = A.T @ A
        X = rng.multivariate_normal(mean=rng.normal(size=k), cov=cov, size=n)

        n_extra_feats = p - k
        W = rng.normal(size=(k, n_extra_feats))
        X = np.column_stack([X, X @ W])
    else:
        A = rng.normal(0, 1, size=(p // 2, p))
        cov = A.T @ A
        X = rng.multivariate_normal(mean=rng.normal(size=p), cov=cov, size=n)

    beta = rng.normal(size=p)
    logits = X @ beta + rng.normal(loc=0, scale=5, size=n)
    logits = scale(logits.reshape(-1, 1)).reshape(-1)
    y = rng.binomial(n=1, p = 1 / (1 + np.exp(-2*logits)), size=n)

    return X, y

def make_regression(n, p, k=None, rank=None, seed=10, cov=None):
    rng = np.random.default_rng(seed)

    if cov is not None:
        X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    else:
        if k is not None:
            if rank is None:
                rank = k // 2

            A = rng.normal(0, 1, size=(k, rank))
            cov = A @ A.T + np.diag(rng.uniform(low=0.01, high=0.1, size=k))
            X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

            n_extra_feats = p - k
            W = rng.normal(size=(k, n_extra_feats))
            new_feats = X @ W

            new_feats_noise_level = np.var(new_feats, axis=0) / 10

            new_feats += rng.multivariate_normal(mean=np.zeros(p - k), cov=np.diag(new_feats_noise_level))

            X = np.column_stack([X, new_feats])
        else:
            if rank is None:
                rank = p // 2

            A = rng.normal(0, 1, size=(rank, p))
            cov = A.T @ A + np.diag(rng.uniform(low=0.01, high=0.1, size=p))
            X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    beta = rng.normal(size=p)
    y = X @ beta

    noise_var = np.var(y) / 10
    noise = rng.normal(loc=0, scale=np.sqrt(noise_var), size=n)
    y += noise

    # Scale y
    y = (y - y.mean()) / y.std()

    return X, y

def make_block_diagonal_regression(n, p, block_size, block_corr=0.3, seed=10, snr=10):
    rng = np.random.default_rng(seed)

    if p % block_size != 0:
        raise ValueError(f"p must be divisible by block size, got p={p}, block_size={block_size}")

    n_blocks = p // block_size

    block = np.ones(shape=(block_size, block_size)) * block_corr
    np.fill_diagonal(block, 1)

    cov = block_diag(*[
        block.copy() for _ in range(n_blocks)
    ])

    # X = rng.multivariate_normal(mean=rng.normal(size=p), cov=cov, size=n)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    # Generate y from means
    means = np.stack(np.split(X, block_size, axis=1), axis=2).mean(axis=2)
    beta = rng.normal(size=n_blocks)
    y = means @ beta

    # Scale y
    y = (y - y.mean()) / y.std()
    noise_var = np.var(y) / snr
    noise = rng.normal(loc=0, scale=np.sqrt(noise_var), size=n)
    y += noise

    return X, y, means





# Make low rank X matrix
# First generates eigenvalues, then multiplies by orthogonal matrices
def gen_low_rank_data(n, p, rank, seed=10, snr=10):
    # Make covariance matrix
    rng = np.random.default_rng(seed)
    X = make_low_rank_matrix(
        n_samples=n,
        n_features=p,
        effective_rank=rank,
    )
    beta = rng.normal(size=p)
    y = X @ beta

    # Fix signal to noise ratio
    noise_var = np.var(y) / snr
    noise = rng.normal(loc=0, scale=np.sqrt(noise_var), size=n)
    y += noise

    # scale y
    y = (y - y.mean()) / y.std()

    return X, y


def time_fit(model, X_train, y_train):
    start = time()
    model.fit(X_train, y_train)
    end = time()
    return end - start

def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("Ran out of time")

def time_fit_transform(transform, X, y, time_limit=None):
    if time_limit is None:
        return time_fit_transform_(transform, X, y)
    else:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time_limit)    

        try:
            results = time_fit_transform_(transform, X, y)
        except:
            results = (None, time_limit)
        finally:
            signal.alarm(0)

        return results

def time_fit_transform_(transform, X, y):
    start = time()
    X_transformed = transform.fit_transform(X, y)
    end = time()
    return X_transformed, end - start

def make_val_split(X_train, seed=10, val_size=0.2):
    X_train_, _ = train_test_split(X_train, test_size=val_size, random_state=seed)

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if i in X_train_.index else 0 for i in X_train.index]
    pds = PredefinedSplit(test_fold = split_index)
    return pds

def load_dataset(dataset_name):
    dataset_ids = {
        "phoneme": 1489,
        "miniboone": 41150,
        "wine": 40498,
        "higgs": 23512,
        "christine": 41142,
        "volkert": 41166,
        "dilbert": 41163,
        "housing": 537,
        "elevator": 216,
        "yolanda": 42705,
        "tecator": 505,
        "space": 507,
        "arcene": 1458,
        "eucalyptus": 188,
        "crime": 315,
        "college": 488,
        "meta": 566,
        "dating": 40536,
        "bands": 6332,
        "voting": 56,
        "arrhythmia": 1017,
        "philippine": 41145,
        "mice": 40966,
    }

    id = dataset_ids[dataset_name]
    task, _, _ = get_dataset_details(dataset_name)

    if task == "regression":
        data = fetch_openml(data_id=id)
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = (y - y.mean()) / y.std()

        if dataset_name in ["crime", "meta"]:
            # X = X[[c for c in X.columns if X[c].dtype.name != "category"]]
            X = X[[c for c in X.columns if X[c].dtype.name != "category" and X[c].isna().sum() > 0]]

    
    else:
        data = fetch_openml(data_id=id)
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)

        if dataset_name in ["volkert", "christine", "arcene"]:
            X = X[[c for c in X.columns if X[c].nunique() > 2]] # Remove unnecessary features
        elif dataset_name == "higgs":
            X = X.iloc[:-1, :] # last columns has NAs
            y = y[:-1]
        elif dataset_name in ["arrhythmia"]:
            X = X[[c for c in X.columns if X[c].dtype.name != "category" and X[c].isna().sum() == 0]]
        elif dataset_name in ["eucalyptus", "college", "mice", "dating", "bands"]:
            X = X[[c for c in X.columns if X[c].dtype.name != "category" and X[c].isna().sum() > 0]]

    return X, y

def get_gc_type(dataset_name):
    if dataset_name in ["christine", "volkert", "dilbert", "yolanda", "arcene", "tecator"]:
        return "lrgc"
    else:
        return "gc"

def get_dataset_details(dataset_name):
    if dataset_name in ["phoneme", "christine", "arcene", "higgs", "miniboone", "dating", "bands", "voting", "arrhythmia", "philippine"]:
        task = "binary"
        models = [LogisticRegression, XGBClassifier, MLPClassifier]
        model_params = [get_default_params(model) for model in models]
        return task, models, model_params
    if dataset_name in ["volkert", "wine", "dilbert", "eucalyptus", "college", "mice"]:
        task = "multiclass"
        models = [LogisticRegression, XGBClassifier, MLPClassifier]
        model_params = [get_default_params(model) for model in models]
        return task, models, model_params
    if dataset_name in ["space", "tecator", "housing", "yolanda", "elevator", "crime", "meta"]:
        task = "regression"
        models = [LinearRegression, XGBRegressor, MLPRegressor]
        model_params = [get_default_params(model) for model in models]
        return task, models, model_params

def get_default_params(model):
    if model == LogisticRegression:
        return {"penalty": "none", "n_jobs": 1}
    if model == LinearRegression:
        return {"n_jobs": 1}
    if model == XGBRegressor:
        return {"n_jobs": 1}
        # return {}
    if model == XGBClassifier:
        return {
            "use_label_encoder": False, 
            "eval_metric": "logloss",
            "n_jobs": 1
        }
    if model in [MLPClassifier, MLPRegressor]:
        return {
            "hidden_layer_sizes": (32, 32), 
            "alpha": 0, 
            "batch_size": 128, 
            "max_iter": 30, 
            "solver": "adam",
        }