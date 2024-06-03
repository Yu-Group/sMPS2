import argparse
from os.path import join as oj
import os
import pandas as pd
import numpy as np
import pickle as pkl
import copy


if __name__ == '__main__':

    # load inputs
    parser = argparse.ArgumentParser()

    parser.add_argument('--include_clin', action='store_true', default=False)
    parser.add_argument('--keep_models', type=str, default=None)
    parser.add_argument('--split_seeds', type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument('--results_root_path', type=str, default="results")
    parser.add_argument('--results_dirs', type=str, default="original,no_sample_filtering,ct40,normct21")
    parser.add_argument('--scale_X', action='store_true', default=False)

    args = parser.parse_args()
    split_seeds = [int(x) for x in args.split_seeds.split(",")]
    results_dirs = args.results_dirs.split(",")

    if args.include_clin:
        res_subdir = "train_with_clinical"
    else:
        res_subdir = "train_without_clinical"

    if args.scale_X:
        data_suffix = "_scaled"
    else:
        data_suffix = ""

    ## aggregate results
    vimps_all = {}
    for results_dir in results_dirs:
        cv_errs_dict = {}
        cv_vimps_dict = {}
        ensemble_vimps = {}
        out_dir = oj(args.results_root_path, results_dir, res_subdir)
        for split_seed in split_seeds:
            # get CV errors
            cv_errs = pkl.load(open(oj(out_dir, str(split_seed), f"valid_errs{data_suffix}.pkl"), "rb"))
            cv_errs_df = pd.DataFrame.from_dict({(i, j): cv_errs[i][j] for i in cv_errs.keys() for j in cv_errs[i].keys()}, orient='index').reset_index()
            cv_errs_df = cv_errs_df.rename(columns={"level_0": "fold_id","level_1": "method"})
            cv_errs_dict[split_seed] = cv_errs_df

            # get CV variable importances
            cv_vimps = pkl.load(open(oj(out_dir, str(split_seed), f"vimps{data_suffix}.pkl"), "rb"))
            cv_vimps_folds = {}
            for fold_id, fold_vimps_dict in cv_vimps.items():
                cv_vimps_folds[fold_id] = pd.concat(fold_vimps_dict).reset_index().\
                    drop(columns=["level_1"]).rename(columns={"level_0": "method"})
            cv_vimps_dict[split_seed] = pd.concat(cv_vimps_folds).reset_index().\
                drop(columns=["level_1"]).rename(columns={"level_0": "fold_id"})

            # get variables importances, aggregated across CV folds
            vimps = pkl.load(open(oj(out_dir, str(split_seed), f"agg_vimps{data_suffix}.pkl"), "rb"))
            naive_vimps = {}
            # get top k features using naive method (i.e., for each method and replicate separately)
            for pipe_name, vimp_df in vimps.items():
                if args.keep_models is not None and pipe_name not in args.keep_models:
                    continue
                if "sparsity" in vimp_df.columns:
                    sort_cols = ["sparsity", "importance"]
                else:
                    sort_cols = ["importance"]
                vimp_df = vimp_df.sort_values(sort_cols, ascending=False)
                vimp_df["rank"] = np.arange(vimp_df.shape[0]) + 1
                naive_vimps[pipe_name] = vimp_df
            pkl.dump(naive_vimps, open(oj(out_dir, str(split_seed), f"naive_vimps{data_suffix}.pkl"), "wb"))
            vimps_df = pd.concat(naive_vimps).reset_index().\
                drop(columns=["level_1"]).rename(columns={"level_0": "method"})
            ensemble_vimps[split_seed] = vimps_df

        # save fold results
        cv_errs_df = pd.concat(cv_errs_dict).reset_index().\
            drop(columns=["level_1"]).rename(columns={"level_0": "rep"})
        cv_errs_df.to_csv(oj(out_dir, f"cv_errs{data_suffix}.csv"), index=False)

        cv_vimps_df = pd.concat(cv_vimps_dict).reset_index().\
            drop(columns=["level_1"]).rename(columns={"level_0": "rep"})
        cv_vimps_df.to_csv(oj(out_dir, f"cv_vimps{data_suffix}.csv"), index=False)

        # get top k features across all methods (ensemble) and across 2 linear + 2 nonlinear methods (ensemble_small)
        ensemble_vimps = pd.concat(ensemble_vimps).reset_index().\
            drop(columns=["level_1"]).rename(columns={"level_0": "rep"})
        ensemble_vimps_ranked = ensemble_vimps.groupby(["rep", "var", "varname"]).\
            agg({"rank": "mean"}).sort_values("rank").reset_index()
        ensemble_small_vimps = copy.deepcopy(ensemble_vimps)[ensemble_vimps.method.isin(["randf", "rfplus", "lasso", "ridge"])]
        ensemble_small_vimps_ranked = ensemble_small_vimps.groupby(["rep", "var", "varname"]).\
            agg({"rank": "mean"}).sort_values("rank").reset_index()
        ensemble_linear_vimps = copy.deepcopy(ensemble_vimps)[ensemble_vimps.method.isin(["lasso", "ridge"])]
        ensemble_linear_vimps_ranked = ensemble_linear_vimps.groupby(["rep", "var", "varname"]).\
            agg({"rank": "mean"}).sort_values("rank").reset_index()
        ensemble_nonlinear_vimps = copy.deepcopy(ensemble_vimps)[ensemble_vimps.method.isin(["randf", "rfplus"])]
        ensemble_nonlinear_vimps_ranked = ensemble_nonlinear_vimps.groupby(["rep", "var", "varname"]).\
            agg({"rank": "mean"}).sort_values("rank").reset_index()    
        ensemble_vimps.to_csv(oj(out_dir, f"agg_vimps{data_suffix}.csv"), index=False)
        pkl.dump(ensemble_vimps_ranked, open(oj(out_dir, f"ensemble_vimps{data_suffix}.pkl"), "wb"))
        pkl.dump(ensemble_small_vimps_ranked, open(oj(out_dir, f"ensemble_small_vimps{data_suffix}.pkl"), "wb"))
        pkl.dump(ensemble_linear_vimps_ranked, open(oj(out_dir, f"ensemble_linear_vimps{data_suffix}.pkl"), "wb"))
        pkl.dump(ensemble_nonlinear_vimps_ranked, open(oj(out_dir, f"ensemble_nonlinear_vimps{data_suffix}.pkl"), "wb"))
        vimps_all[results_dir] = ensemble_vimps

    # get top k features across methods and data preprocessing
    out_dir = oj(args.results_root_path, res_subdir)
    os.makedirs(out_dir, exist_ok=True)
    vimps_all = pd.concat(vimps_all).reset_index().\
        drop(columns=["level_1"]).rename(columns={"level_0": "pipeline"})
    vimps_ranked = vimps_all.groupby(["rep", "var", "varname"]).\
        agg({"rank": "mean"}).sort_values("rank").reset_index()
    vimps_small_all = copy.deepcopy(vimps_all)[vimps_all.method.isin(["randf", "rfplus", "lasso", "ridge"])]
    vimps_small_ranked = vimps_small_all.groupby(["rep", "var", "varname"]).\
        agg({"rank": "mean"}).sort_values("rank").reset_index()
    pkl.dump(vimps_all, open(oj(out_dir, f"vimps{data_suffix}.pkl"), "wb"))
    pkl.dump(vimps_ranked, open(oj(out_dir, f"pcs_vimps{data_suffix}.pkl"), "wb"))
    pkl.dump(vimps_small_ranked, open(oj(out_dir, f"pcs_small_vimps{data_suffix}.pkl"), "wb"))

    print('Completed aggregating training results!')

# %%
