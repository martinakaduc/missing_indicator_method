import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import itertools
from tqdm import tqdm

import sys
sys.path.append("mimic3-benchmarks")
from mimic3benchmark.readers import (
    InHospitalMortalityReader, 
    DecompensationReader, 
    LengthOfStayReader, 
    PhenotypingReader,
)
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from mimic3models import metrics

n_bins = 10

def one_hot(index):
    x = np.zeros((n_bins,), dtype=np.int32)
    x[index] = 1
    return x


def read_and_extract_features(reader, count, period, features):
    read_chunk_size = 1000
    Xs = []
    ys = []
    names = []
    ts = []
    for i in tqdm(range(0, count, read_chunk_size)):
        j = min(count, i + read_chunk_size)
        ret = common_utils.read_chunk(reader, j - i)
        X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
        Xs.append(X)
        ys += ret['y']
        names += ret['name']
        ts += ret['t']
    Xs = np.concatenate(Xs, axis=0)
    bins = np.array([one_hot(metrics.get_bin_custom(x, n_bins)) for x in ys])
    return (Xs, bins, ys, names, ts)

# To run this, you must have already prepared the data into
# train, val, and test splits using the mimic-benchmark scripts
# according to their README.
data_dir = 'mimic3-benchmarks/data/length-of-stay/'
period = "all"
features = "all"

train_reader = LengthOfStayReader(dataset_dir=os.path.join(data_dir, 'train'),
                                  listfile=os.path.join(data_dir, 'train_listfile.csv'))

val_reader = LengthOfStayReader(dataset_dir=os.path.join(data_dir, 'train'),
                                listfile=os.path.join(data_dir, 'val_listfile.csv'))

test_reader = LengthOfStayReader(dataset_dir=os.path.join(data_dir, 'test'),
                                 listfile=os.path.join(data_dir, 'test_listfile.csv'))

print('Reading data and extracting features ...')
n_train = min(100000, train_reader.get_number_of_examples())
n_val = min(100000, val_reader.get_number_of_examples())

(train_X, train_y, train_actual, train_names, train_ts) = read_and_extract_features(
    train_reader, n_train, period, features)

(val_X, val_y, val_actual, val_names, val_ts) = read_and_extract_features(
    val_reader, n_val, period, features)

(test_X, test_y, test_actual, test_names, test_ts) = read_and_extract_features(
    test_reader, test_reader.get_number_of_examples(), period, features)

ret = common_utils.read_chunk(train_reader, 1)
headers = ret["header"][1:]
subseqs = ["full", "0-10%", "0-25%", "0-50%", "50-100%", "75-100%", "90-100%"]
fns = ["min", "max", "mean", "std", "skew", "len"]

columns = [f"{header} | {subseq} | {fn}" for header, subseq, fn in itertools.product(headers, subseqs, fns)]

train_X_df = pd.DataFrame(train_X, columns=columns)
train_X_df["target"] = np.array([np.where(r==1)[0][0] for r in train_y])
val_X_df = pd.DataFrame(val_X, columns=columns)
val_X_df["target"] = np.array([np.where(r==1)[0][0] for r in val_y])
test_X_df = pd.DataFrame(test_X, columns=columns)
test_X_df["target"] = np.array([np.where(r==1)[0][0] for r in test_y])

train_X_df.to_csv("../mimic_data/los_tab_train.csv", index=False)
val_X_df.to_csv("../mimic_data/los_tab_val.csv", index=False)
test_X_df.to_csv("../mimic_data/los_tab_test.csv", index=False)