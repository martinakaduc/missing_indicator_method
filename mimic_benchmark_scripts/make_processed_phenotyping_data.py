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
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from mimic3models import metrics

def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'], ret['t'])

# To run this, you must have already prepared the data into
# train, val, and test splits using the mimic-benchmark scripts
# according to their README.
data_dir = 'mimic3-benchmarks/data/phenotyping/'
period = "all"
features = "all"

train_reader = PhenotypingReader(dataset_dir=os.path.join(data_dir, 'train'),
                                     listfile=os.path.join(data_dir, 'train_listfile.csv'))

val_reader = PhenotypingReader(dataset_dir=os.path.join(data_dir, 'train'),
                               listfile=os.path.join(data_dir, 'val_listfile.csv'))

test_reader = PhenotypingReader(dataset_dir=os.path.join(data_dir, 'test'),
                                listfile=os.path.join(data_dir, 'test_listfile.csv'))

print('Reading data and extracting features ...')

(train_X, train_y, train_names, train_ts) = read_and_extract_features(train_reader, period, features)
train_y = np.array(train_y)

(val_X, val_y, val_names, val_ts) = read_and_extract_features(val_reader, period, features)
val_y = np.array(val_y)

(test_X, test_y, test_names, test_ts) = read_and_extract_features(test_reader, period, features)
test_y = np.array(test_y)

print("train set shape:  {}".format(train_X.shape))
print("test set shape: {}".format(test_X.shape))

ret = common_utils.read_chunk(train_reader, 1)
headers = ret["header"][1:]
subseqs = ["full", "0-10%", "0-25%", "0-50%", "50-100%", "75-100%", "90-100%"]
fns = ["min", "max", "mean", "std", "skew", "len"]

columns = [f"{header} | {subseq} | {fn}" for header, subseq, fn in itertools.product(headers, subseqs, fns)]

train_X_df = pd.DataFrame(train_X, columns=columns)
val_X_df = pd.DataFrame(val_X, columns=columns)
test_X_df = pd.DataFrame(test_X, columns=columns)

train_X_df.to_csv("../mimic_data/phenotyping_tab_train.csv", index=False)
val_X_df.to_csv("../mimic_data/phenotyping_tab_val.csv", index=False)
test_X_df.to_csv("../mimic_data/phenotyping_tab_test.csv", index=False)

train_y_df = pd.DataFrame(train_y)
val_y_df = pd.DataFrame(val_y)
test_y_df = pd.DataFrame(test_y)

train_y_df.to_csv("../mimic_data/phenotyping_train_labels.csv", index=False)
val_y_df.to_csv("../mimic_data/phenotyping_val_labels.csv", index=False)
test_y_df.to_csv("../mimic_data/phenotyping_test_labels.csv", index=False)