#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from validphys.loader import FallbackLoader as Loader
from validphys.api import API
from collections import defaultdict
import os
import sys
import logging
from nnpdf_data import legacy_to_new_map


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
            tr_file_path = os.path.join(replica_dir, replica_folder, 'datacuts_theory_fitting_training_pseudodata.csv')
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
            val_file_path = os.path.join(replica_dir, replica_folder, 'datacuts_theory_fitting_validation_pseudodata.csv')
            val_sizes.append(count_csv_rows(val_file_path))
    return val_sizes

if __name__ == "__main__":
    # Read the fitname from command-line arguments
    if len(sys.argv) > 1:
        fitname = sys.argv[1]
    else:
        raise ValueError("Fitname not provided as a command-line argument.")

    fit_names = [f"{fitname}-0{n}-batch1" for n in range(114,125+1)]

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



    # EXP method --------------------------------------------------------------

    naive_dict = dict(
        fit=fit_names[0],
        dataset_inputs={"from_": "fit"},
        pdf={"from_": "fit"},
        use_cuts="fromfit",
        theory={"from_": "fit"},
        theoryid={"from_": "theory"},
    )

    # t0 covariance matrix (the correct one, see bottom of page 15 of https://arxiv.org/pdf/1802.03398)
    C = API.groups_covmat(
        fit=fit_names[0],
        use_t0 = True,
        use_cuts="fromfit",
        datacuts={"from_": "fit"},
        t0pdfset={"from_": "datacuts"},
        dataset_inputs={"from_": "fit"},
        theoryid=API.fit(fit=fit_names[0]).as_input()["theory"]["t0theoryid"],
    )

    # the datapoint is already uniquely defined by the dataset and datapoint, we dont need the process
    C = C.droplevel(0, axis=0).droplevel(0, axis=1)

    try:
        stored_covmat = pd.read_csv(
            fits[0].path / "tables/datacuts_theory_theorycovmatconfig_theory_covmat_custom.csv",
            index_col=[0, 1, 2],
            header=[0, 1, 2],
            sep="\t|,",
            engine="python",
        ).fillna(0)
        tmp = stored_covmat.droplevel(0, axis=0).droplevel(0, axis=1) # drop process level
        new_names = {d[0]: legacy_to_new_map(d[0])[0] for d in tmp.index}
        tmp.rename(columns=new_names, index=new_names, level=0, inplace=True) # rename datasets using the legacy to new map
        tmp_index = pd.MultiIndex.from_tuples(
            [(bb, np.int64(cc)) for bb, cc in tmp.index],
            names=["dataset", "id"],
        ) # make sure the index is an int, just as it is in C
        tmp = pd.DataFrame(
            tmp.values, index=tmp_index, columns=tmp_index
        )
        stored_covmat = tmp.reindex(C.index).T.reindex(C.index)
        if stored_covmat.isnull().values.any():
            print("some values are NaN, meaning that not all indices in C.index exist in tmp")
        invcov = np.linalg.inv(C+stored_covmat)
    except:
        invcov = np.linalg.inv(C)

    chi2_values = []
    alphas_values = []
    for fitname in fit_names:
        naive_dict["fit"] = fitname
        central_preds_and_data = API.group_result_central_table_no_table(**naive_dict)

        theory_db_id = API.fit(fit=fitname).as_input()["theory"]["theoryid"]
        alphas_values.append(API.theory_info_table(theory_db_id = theory_db_id).loc["alphas"].iloc[0])

        # compute chi2
        diff = central_preds_and_data.theory_central - central_preds_and_data.data_central
        chi2_values.append(diff @ invcov @ diff / diff.size)

    a, b, c = np.polyfit(alphas_values, chi2_values, 2)

    central = -b / 2 / a
    ndata = C.shape[0]
    unc = np.sqrt(1/a/ndata)

    xgrid = np.linspace(min(alphas_values),max(alphas_values))
    print(f"{central:.5f} ± {unc:.5f}")

    logging.error(rf"""{fitname}
                  CRM: {middle:.5f} ± {uncertainty:.5f}
                  EXP: {central:.5f} ± {unc:.5f}""")
