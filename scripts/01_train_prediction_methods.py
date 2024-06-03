import argparse
from os.path import join as oj
import os
import pandas as pd
import numpy as np
import copy
import pickle as pkl
import importlib

from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
from functions.pipeline import run_binary_classification_pipeline


if __name__ == '__main__':

    # load inputs
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_fpath', type=str, default='../data/data_cleaned.csv')
    parser.add_argument('--data_index', action='store_true', default=False)
    parser.add_argument('--filter_samples', type=str, default=None)
    parser.add_argument('--config', type=str, default="models")
    parser.add_argument('--include_clin', action='store_true', default=False)
    parser.add_argument('--keep_models', type=str, default=None)
    parser.add_argument('--n_folds', type=int, default=4)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str, default="results")
    parser.add_argument('--scale_X', action='store_true', default=False)

    args = parser.parse_args()

    # define helper variables
    CLIN_COLS = ["age", "aa", "fhx", "dre_abnl", "bx_prior_neg", "psa", "prostate_volume"]
    DROP_CLIN_COLS = ["prostate_volume"]
    Y_COL = "grouping"
    TEST_SIZE = 0.2
    CONFIG = importlib.import_module(f'model_config.{args.config}')
    cv_param_grid_all = CONFIG.CV_PARAM_GRID
    models_all = CONFIG.MODELS
    fi_models_all = CONFIG.FI_MODELS
    if args.keep_models is not None:
        keep_models = args.keep_models.split(",")
    else:
        keep_models = None

    # load in data
    if args.data_index:
        data_orig = pd.read_csv(args.data_fpath, index_col=0)
    else:
        data_orig = pd.read_csv(args.data_fpath)
    y = data_orig[Y_COL] == "high"

    if args.scale_X:
        data_orig = (data_orig - data_orig.mean()) / data_orig.std()
        data_suffix = "_scaled"
    else:
        data_suffix = ""

    if args.include_clin:
        X = data_orig.drop(columns=DROP_CLIN_COLS + [Y_COL])
        res_subdir = "train_with_clinical"
    else:
        X = data_orig.drop(columns=CLIN_COLS + [Y_COL])
        res_subdir = "train_without_clinical"

    out_dir = oj(args.results_path, res_subdir, str(args.split_seed))
    os.makedirs(out_dir, exist_ok=True)
    n_folds = args.n_folds

    # data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.split_seed
    )
    if args.filter_samples:
        keep_samples_df = pd.read_csv(args.filter_samples)
        keep_samples_idx = X_test.index.isin(keep_samples_df["x"])
        X_test = X_test[keep_samples_idx]
        y_test = y_test[keep_samples_idx]
    X_train.to_csv(oj(out_dir, f"X_train{data_suffix}.csv"), index=False)
    y_train.to_csv(oj(out_dir, "y_train.csv"), index=False)
    X_test.to_csv(oj(out_dir, f"X_test{data_suffix}.csv"), index=False)
    y_test.to_csv(oj(out_dir, "y_test.csv"), index=False)

    n_train = X_train.shape[0]
    fold_ids = np.random.choice(
        [i for i in range(n_folds)] * int(np.ceil(n_train / n_folds)),
        n_train, replace=False
    )
    pd.DataFrame(fold_ids).to_csv(oj(out_dir, "cv_fold_ids.csv"), index=False)

    # fit prediction models
    valid_errs_all = {}
    boot_valid_errs_all = {}
    valid_preds_all = {}
    valid_prob_preds_all = {}
    tuned_pipelines_all = {}
    vimps_all = {}
    agg_vimps_all = {}
    for fold_id in range(n_folds):
        X_train_fold = X_train[fold_ids != fold_id]
        y_train_fold = y_train[fold_ids != fold_id]
        X_valid_fold = X_train[fold_ids == fold_id]
        y_valid_fold = y_train[fold_ids == fold_id]
        valid_errs, valid_preds, valid_prob_preds, tuned_pipelines, vimps, boot_valid_errs = \
            run_binary_classification_pipeline(
                X_train_fold, y_train_fold, X_valid_fold, y_valid_fold,
                models_all=models_all, cv_param_grid_all=cv_param_grid_all,
                fi_models_all=fi_models_all, keep_models=keep_models,
                importance=True
            )
        valid_errs_all[fold_id] = copy.deepcopy(valid_errs)
        boot_valid_errs_all[fold_id] = copy.deepcopy(boot_valid_errs)
        valid_preds_all[fold_id] = copy.deepcopy(valid_preds)
        valid_prob_preds_all[fold_id] = copy.deepcopy(valid_prob_preds)
        tuned_pipelines_all[fold_id] = copy.deepcopy(tuned_pipelines)
        vimps_all[fold_id] = copy.deepcopy(vimps)

    # aggregate feature importances
    for pipe_name in vimps_all[0].keys():
        agg_vimp_df = pd.concat([vimps_all[i][pipe_name] for i in range(n_folds)]). \
            groupby(level=0).mean().sort_values("var")
        agg_vimp_df["var"] = agg_vimp_df["var"].astype(int)
        agg_vimp_df["varname"] = X.columns
        if args.include_clin:
            agg_vimp_df = agg_vimp_df[~agg_vimp_df["varname"].isin(CLIN_COLS)]
        agg_vimps_all[pipe_name] = agg_vimp_df

    # save results
    pkl.dump(valid_errs_all, open(oj(out_dir, f"valid_errs{data_suffix}.pkl"), "wb"))
    pkl.dump(boot_valid_errs_all, open(oj(out_dir, f"boot_valid_errs{data_suffix}.pkl"), "wb"))
    pkl.dump(valid_preds_all, open(oj(out_dir, f"valid_preds{data_suffix}.pkl"), "wb"))
    pkl.dump(valid_prob_preds_all, open(oj(out_dir, f"valid_prob_preds{data_suffix}.pkl"), "wb"))
    pkl.dump(tuned_pipelines_all, open(oj(out_dir, f"tuned_pipelines{data_suffix}.pkl"), "wb"))
    pkl.dump(vimps_all, open(oj(out_dir, f"vimps{data_suffix}.pkl"), "wb"))
    pkl.dump(agg_vimps_all, open(oj(out_dir, f"agg_vimps{data_suffix}.pkl"), "wb"))

    print('Completed training!')

# %%
