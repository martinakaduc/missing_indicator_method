import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from category_encoders.ordinal import OrdinalEncoder
from sklearn.compose import make_column_transformer
from pyampute.ampute import MultivariateAmputation
from time import time

def simple_mask(X, p=0.5, rng=None, seed=None, return_na=False):
    if not rng and not seed:
        rng = np.random.default_rng()
    elif not rng:
        rng = np.random.default_rng(seed)

    # Simple MCAR mask
    mask = rng.binomial(n=1, p=p, size=X.shape)

    if return_na:
        mask = np.where(mask == 0, np.nan, mask)

    return mask

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MNAR_mask(x, side="tail", rng=None, seed=None, power=1, standardize=False, return_na=False):
    # if not isinstance(x, np.ndarray):
    #     x = x.to_numpy("float")

    if not rng and not seed:
        rng = np.random.default_rng()
    elif not rng:
        rng = np.random.default_rng(seed)

    if standardize:
        x = StandardScaler().fit_transform(x)

    if side == "tail":
        probs = sigmoid((np.abs(x) - 0.75)*power)
    elif side == "mid":
        probs = sigmoid((-np.abs(x) + 0.75)*power)
    elif side == "left":
        probs = sigmoid(-x*power)
    elif side == "right":
        probs = sigmoid(x*power)
    else:
        raise ValueError(f"Side must be one of tail, mid, left, or right, got {side}")

    mask = rng.binomial(1, probs, size=x.shape)
    
    if return_na:
        mask = np.where(mask == 0, np.nan, mask)
    
    return probs, mask

def MAR_mask(X, patterns):

    if isinstance(X, pd.DataFrame):
        columns = X.columns
        X = X.to_numpy()

    masker = MultivariateAmputation(
        patterns=[{"incomplete_vars": p, "mechanism": "MAR", "score_to_probability_func": "sigmoid-left"} for p in patterns], 
        prop=0.5
    )

    X_masked = masker.fit_transform(X)
    mask = np.where(np.isnan(X_masked), np.nan, 1)

    return mask


# def make_pattern(n_feats, seed=int(time())):
#     np.random.seed(seed)
#     while True:
#         miss_feats = np.random.binomial(n=1, p=0.25, size=(n_feats))
#         selected = [x for m, x in zip(miss_feats, range(n_feats)) if m]
        
#         if len(selected) == 1:
#             return selected
        
# #         if len(selected) > 0 and len(selected) < n_feats:
# #             return selected

# def mask_data(X, mech="MCAR", seed=int(time()), reorder=False):
    
#     col_order = X.columns.tolist()
    
#     # First, ordinally encode categorical columns
#     cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
#     num_cols = [c for c in X.columns if X[c].dtype.name != "category"]
    
#     if cat_cols:
    
#         ord_encoder = make_column_transformer(
#             (OrdinalEncoder(), cat_cols),
#             remainder="passthrough"
#         )

#         X = ord_encoder.fit_transform(X)
        
#     else:
#         X = X.to_numpy()
    
#     # Randomly choose patterns set
#     n_patterns = 3
#     n_feats = len(col_order)
#     patterns = {tuple(make_pattern(n_feats, seed+i)) for i in range(n_patterns)}
    
#     # Use pyampute to generate mask
#     masker = MultivariateAmputation(
#         patterns=[{"incomplete_vars": list(p), "mechanism": mech, "score_to_probability_func": "sigmoid-right"} for p in patterns], 
#         prop=0.99
#     )
#     print(masker)

#     X_masked = masker.fit_transform(X)
    
#     if cat_cols:


#         # Convert back to original mapping
#         oe = ord_encoder.transformers_[0][1]
#         X_masked_cat = np.where(np.isnan(X_masked[:, :len(cat_cols)]), -2, X_masked[:, :len(cat_cols)])
#         X_masked_cat = pd.DataFrame(X_masked_cat, columns=cat_cols)
#         X_masked_cat = oe.inverse_transform(X_masked_cat)

#         # Concat and potentially Redo original dataset column order
#         X_masked = pd.concat( (X_masked_cat, pd.DataFrame(X_masked[:, len(cat_cols):], columns=num_cols)), axis=1 )
        
#     else:
#         X_masked = pd.DataFrame(X_masked, columns=num_cols)
    
#     if not reorder:
#         X_masked = X_masked[col_order]
    
#     return X_masked, patterns
