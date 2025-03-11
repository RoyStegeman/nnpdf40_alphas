import pandas as pd
import numpy as np
from validphys.loader import FallbackLoader as Loader
from validphys.api import API
from collections import defaultdict
import matplotlib.pyplot as plt
import os

def measure(replica_data, tr_size, val_size):
    return replica_data.training*tr_size + replica_data.validation*val_size

def count_csv_rows(file_path):
    if os.path.exists(file_path):
        with open(file_path) as file:
            return sum(1 for _ in file) - 1  # Subtract 1 for the header
    else:
        return np.nan  # Return nan if the file doesn't exist

def get_tr_masks(fit):
    # Initialize a list to store the number of entries for each replica
    tr_sizes = []

    # Path to the directory containing replica folders
    replica_dir = os.path.join(fit.path, 'nnfit')

    # Iterate over all replica_* directories
    for replica_folder in os.listdir(replica_dir):
        if replica_folder.startswith('replica_'):
            tr_file_path = os.path.join(replica_dir, replica_folder, 'datacuts_theory_closuretest_fitting_training_pseudodata.csv')
            tr_sizes.append(count_csv_rows(tr_file_path))
    return tr_sizes

def get_val_masks(fit):
    # Initialize a list to store the number of entries for each replica
    val_sizes = []

    # Path to the directory containing replica folders
    replica_dir = os.path.join(fit.path, 'nnfit')

    # Iterate over all replica_* directories
    for replica_folder in os.listdir(replica_dir):
        if replica_folder.startswith('replica_'):
            val_file_path = os.path.join(replica_dir, replica_folder, 'datacuts_theory_closuretest_fitting_validation_pseudodata.csv')
            val_sizes.append(count_csv_rows(val_file_path))
    return val_sizes

results = []
for i in range(1, 26):
    fit_names = [f"250306-{i:03d}-rs-closuretest-alphas-0{n}" for n in range(114,125+1)]

    l = Loader()
    fits = [l.check_fit(f) for f in fit_names]

    # Create the dictionaries for training and validation sizes
    tr_sizes = {f: get_tr_masks(f) for f in fits}
    val_sizes = {f: get_val_masks(f) for f in fits}

    as_fits = defaultdict(list)
    for f in fits:
        th = f.as_input()["theory"]["theoryid"]
        alpha = API.theory_info_table(theory_db_id = th).loc["alphas"].iloc[0]
        as_fits[alpha].append(f)
    as_fits = dict(as_fits)

    indexes = {f: API.fitted_replica_indexes(pdf=f.name) for f in fits}
    replica_data = {f: API.replica_data(fit=f.name) for f in fits}

    min_values = {}
    for alpha, flist in as_fits.items():
        series = []
        for f in flist:
            s = [measure(d, tr, vl) for d, tr, vl in zip(replica_data[f], tr_sizes[f], val_sizes[f])]
            series.append(pd.Series(s, index=indexes[f]))
        min_values[alpha] = pd.DataFrame(series).min()
    data = pd.DataFrame(min_values).dropna()

    # for index, row in data.iterrows():
    #     plt.plot(data.columns, row, label=f'Row {index+1}')
    # plt.savefig(f'test{i}.png')
    # plt.close()

    # quadratic polynomial
    mins = {}
    filter_this_row = [] # rows that are filtered
    invcov = np.linalg.inv(np.cov(data.T))
    for ind, row in data.iterrows():
        p2, p1, p0 = np.polyfit(data.columns, row, 2)
        if not np.isnan(p1): # NaN if not all replicas passed postfit
            mins[ind] = -p1 / 2 / p2
    mins = pd.Series(mins)

    lower = mins.describe(percentiles=[0.16,0.84]).iloc[4]
    upper = mins.describe(percentiles=[0.16,0.84]).iloc[6]
    middle = (lower + upper)/2
    uncertainty = (upper - lower)/2
    results.append((f"250306-{i:03d}-rs-closuretest-alphas-0114", middle, uncertainty))
    print(f"{i:03d}  {middle:.4f} ± {uncertainty:.4f}")

results_df = pd.DataFrame(results, columns=['Fit Name', 'Mean', 'Uncertainty'])
results_df.to_csv('./results/crm_nopos_noint.csv', index=False, header=False, float_format="%.7f")

