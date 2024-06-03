import argparse
from os.path import join as oj
import os
import pandas as pd
import numpy as np
import copy
import pickle as pkl
import importlib

import sys
sys.path.append("..")
from functions.pipeline import run_binary_classification_pipeline


if __name__ == '__main__':

    # load inputs
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="models")
    parser.add_argument('--include_clin', action='store_true', default=False)
    parser.add_argument('--keep_models', type=str, default=None)
    parser.add_argument('--topk', type=int, default=17)
    parser.add_argument('--topk_mode', type=str, default="naive")
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str, default="results")
    parser.add_argument('--scale_X', action='store_true', default=False)
    parser.add_argument('--ignore_cache', action='store_true', default=False)

    args = parser.parse_args()
    assert args.topk_mode in ["naive", "ensemble", "ensemble_small", "ensemble_linear", "ensemble_nonlinear", "pcs", "pcs_small"]
    assert args.topk is not None

    # define helper variables
    CLIN_COLS = ["age", "aa", "fhx", "dre_abnl", "bx_prior_neg", "psa", "prostate_volume"]
    DROP_CLIN_COLS = ["prostate_volume"]

    CONFIG = importlib.import_module(f'model_config.{args.config}')
    cv_param_grid_all = CONFIG.CV_PARAM_GRID
    models_all = CONFIG.MODELS
    fi_models_all = CONFIG.FI_MODELS

    if args.include_clin:
        res_subdir = "train_with_clinical"
    else:
        res_subdir = "train_without_clinical"

    if args.scale_X:
        data_suffix = "_scaled"
    else:
        data_suffix = ""

    # get ranked list of important features
    out_dir = oj(args.results_path, res_subdir, str(args.split_seed))
    naive_vimps = pkl.load(open(oj(out_dir, f"naive_vimps{data_suffix}.pkl"), "rb"))
    
    # load in data
    data_dir = oj(args.results_path, "train_with_clinical", str(args.split_seed))
    X_train = pd.read_csv(oj(data_dir, f"X_train{data_suffix}.csv"))
    y_train = pd.read_csv(oj(data_dir, "y_train.csv"))
    X_test = pd.read_csv(oj(data_dir, f"X_test{data_suffix}.csv"))
    y_test = pd.read_csv(oj(data_dir, "y_test.csv"))

    # evaluate models with and without clinical data
    for key in ["eval_with_clinical", "eval_without_clinical"]:
        out_dir = oj(args.results_path, key, res_subdir, str(args.topk), str(args.split_seed))
        os.makedirs(out_dir, exist_ok=True)
        test_errs_all = {}
        test_preds_all = {}
        test_prob_preds_all = {}
        full_tuned_pipelines_all = {}
        
        # get models to fit
        if args.keep_models is None:
            keep_models = naive_vimps.keys()
        else:
            keep_models = args.keep_models.split(",")
            
        # retrieve cache
        if not args.ignore_cache and os.path.exists(oj(out_dir, f"tuned_pipelines_{args.topk_mode}{data_suffix}.pkl")):
            test_errs_all = pkl.load(open(oj(out_dir, f"valid_errs_{args.topk_mode}{data_suffix}.pkl"), "rb"))
            keep_models = [model_name for model_name in keep_models if model_name not in test_errs_all.keys()]
            if len(keep_models) == 0:
                print('Evaluation has been cached previously!')
                continue
            test_preds_all = pkl.load(open(oj(out_dir, f"valid_preds_{args.topk_mode}{data_suffix}.pkl"), "rb"))
            test_prob_preds_all = pkl.load(open(oj(out_dir, f"valid_prob_preds_{args.topk_mode}{data_suffix}.pkl"), "rb"))
            full_tuned_pipelines_all = pkl.load(open(oj(out_dir, f"tuned_pipelines_{args.topk_mode}{data_suffix}.pkl"), "rb"))
      
        for pipe_name in keep_models:
            if args.topk_mode in ["ensemble", "ensemble_small", "ensemble_linear", "ensemble_nonlinear"]:
                ranked_vimps = pkl.load(open(oj(args.results_path, res_subdir, f"{args.topk_mode}_vimps{data_suffix}.pkl"), "rb"))
                ranked_vimps = ranked_vimps.loc[ranked_vimps["rep"] == args.split_seed]
                top_genes = list(ranked_vimps["varname"][:args.topk])
            elif args.topk_mode in ["pcs", "pcs_small"]:
                ranked_vimps = pkl.load(open(oj(os.path.dirname(args.results_path), res_subdir, f"{args.topk_mode}_vimps{data_suffix}.pkl"), "rb"))
                ranked_vimps = ranked_vimps.loc[ranked_vimps["rep"] == args.split_seed]
                top_genes = list(ranked_vimps["varname"][:args.topk])
            else:
                top_genes = list(naive_vimps[pipe_name]["varname"][:args.topk])
            if key == "eval_with_clinical":
                keep_features = top_genes + [x for x in CLIN_COLS if x not in DROP_CLIN_COLS]
            else:
                keep_features = top_genes
            test_errs, test_preds, test_prob_preds, full_tuned_pipelines, _, _ = \
                run_binary_classification_pipeline(
                    X_train.loc[:, keep_features], y_train, X_test.loc[:, keep_features], y_test,
                    models_all=models_all, cv_param_grid_all=cv_param_grid_all,
                    fi_models_all=fi_models_all, keep_models=[pipe_name]
                )
            test_errs_all[pipe_name] = copy.deepcopy(test_errs[pipe_name])
            test_preds_all[pipe_name] = copy.deepcopy(test_preds[pipe_name])
            test_prob_preds_all[pipe_name] = copy.deepcopy(test_prob_preds[pipe_name])
            full_tuned_pipelines_all[pipe_name] = copy.deepcopy(full_tuned_pipelines[pipe_name])
            
            # save results
            pkl.dump(test_errs_all, open(oj(out_dir, f"valid_errs_{args.topk_mode}{data_suffix}.pkl"), "wb"))
            pkl.dump(test_preds_all, open(oj(out_dir, f"valid_preds_{args.topk_mode}{data_suffix}.pkl"), "wb"))
            pkl.dump(test_prob_preds_all, open(oj(out_dir, f"valid_prob_preds_{args.topk_mode}{data_suffix}.pkl"), "wb"))
            pkl.dump(full_tuned_pipelines_all, open(oj(out_dir, f"tuned_pipelines_{args.topk_mode}{data_suffix}.pkl"), "wb"))

    print('Completed evaluation!')

# %%
