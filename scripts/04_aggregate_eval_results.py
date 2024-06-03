import argparse
from os.path import join as oj
import os
import pandas as pd
import numpy as np
import pickle as pkl


if __name__ == '__main__':

    # load inputs
    parser = argparse.ArgumentParser()

    parser.add_argument('--include_clin', action='store_true', default=False)
    parser.add_argument('--topk_mode', type=str, default="naive")
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

    id_cols = ["pipeline", "eval_type", "k", "rep"]

    ## aggregate results
    errs_ls = []
    preds_ls = []
    prob_preds_ls = []
    for results_dir in results_dirs:
        for eval_type in ["eval_with_clinical", "eval_without_clinical"]:
            ks = os.listdir(oj(args.results_root_path, results_dir, eval_type, res_subdir))
            for k in ks:
                for split_seed in split_seeds:
                    out_dir = oj(args.results_root_path, results_dir, eval_type, res_subdir, k, str(split_seed))
                    if not os.path.exists(oj(out_dir, f"valid_errs_{args.topk_mode}{data_suffix}.pkl")):
                        print(f"Warning: No results at {out_dir}.")
                        continue

                    # get test errors
                    errs = pkl.load(open(oj(out_dir, f"valid_errs_{args.topk_mode}{data_suffix}.pkl"), "rb"))
                    errs_df = pd.DataFrame.from_dict(errs, orient="index").reset_index().rename(columns={"index": "method"})
                    errs_df["rep"] = split_seed
                    errs_df["k"] = k
                    errs_df["eval_type"] = eval_type
                    errs_df["pipeline"] = results_dir
                    ordered_cols = id_cols + [col for col in errs_df.columns if col not in id_cols]
                    errs_df = errs_df[ordered_cols]
                    errs_ls.append(errs_df)

                    # get test predictions
                    preds = pkl.load(open(oj(out_dir, f"valid_preds_{args.topk_mode}{data_suffix}.pkl"), "rb"))
                    preds_df = pd.DataFrame(preds)
                    preds_df["sample_id"] = np.arange(preds_df.shape[0]) + 1
                    preds_df["rep"] = split_seed
                    preds_df["k"] = k
                    preds_df["eval_type"] = eval_type
                    preds_df["pipeline"] = results_dir
                    ordered_cols = id_cols + [col for col in preds_df.columns if col not in id_cols]
                    preds_df = preds_df[ordered_cols]
                    preds_ls.append(preds_df)

                    # get test predicted probabilities
                    prob_preds = pkl.load(open(oj(out_dir, f"valid_prob_preds_{args.topk_mode}{data_suffix}.pkl"), "rb"))
                    prob_preds_df = pd.DataFrame(prob_preds)
                    prob_preds_df["sample_id"] = np.arange(prob_preds_df.shape[0]) + 1
                    prob_preds_df["rep"] = split_seed
                    prob_preds_df["k"] = k
                    prob_preds_df["eval_type"] = eval_type
                    prob_preds_df["pipeline"] = results_dir
                    ordered_cols = id_cols + [col for col in prob_preds_df.columns if col not in id_cols]
                    prob_preds_df = prob_preds_df[ordered_cols]
                    prob_preds_ls.append(prob_preds_df)

    errs_df = pd.concat(errs_ls)
    errs_df.to_csv(oj(args.results_root_path, res_subdir, f"test_errors_{args.topk_mode}{data_suffix}.csv"), index=False)
    preds_df = pd.concat(preds_ls)
    preds_df.to_csv(oj(args.results_root_path, res_subdir, f"predictions_{args.topk_mode}{data_suffix}.csv"), index=False)
    prob_preds_df = pd.concat(prob_preds_ls)
    prob_preds_df.to_csv(oj(args.results_root_path, res_subdir, f"prob_predictions_{args.topk_mode}{data_suffix}.csv"), index=False)

    print('Completed aggregating evaluation results!')

# %%
