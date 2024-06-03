import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def run_binary_classification_pipeline(
    X_train, y_train, X_valid, y_valid,
    models_all, cv_param_grid_all, fi_models_all,
    keep_models=None, importance=False, B=100
):
    metrics = {
        "accuracy": (accuracy_score, "predict"),
        "auroc": (roc_auc_score, "predict_proba"),
        "auprc": (average_precision_score, "predict_proba")
    }
    if keep_models is not None:
        models = {}
        for keep_model in keep_models:
            models[keep_model] = copy.deepcopy(models_all[keep_model])
    else:
        models = models_all
    pipes = {}
    for model_name, model in models.items():
        pipe = Pipeline(steps=[(model_name, model)])
        pipes[model_name] = pipe

    valid_errs = {}
    boot_valid_errs = {}
    valid_preds = {}
    valid_prob_preds = {}
    tuned_pipelines = {}
    vimps = {}
    for pipe_name, pipe in pipes.items():
        print(pipe_name)
        # get relevant CV parameters given the steps of the pipeline
        cv_param_grid = {
            key: cv_param_grid_all[key] for key in cv_param_grid_all.keys() \
                if key.startswith(tuple(pipe.named_steps.keys()))
        }
        # if no need for tuning since only one possible hyperparameter choice
        for value in cv_param_grid.values():
            if len(value) == 1:
                nfolds = 2
            else:
                nfolds = 5
                break
        # run CV for pipeline
        pipe_search = GridSearchCV(pipe, cv_param_grid, cv=nfolds)
        if pipe_name in ["rfplus"]:
            pipe_search.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
        else:
            pipe_search.fit(X_train, y_train.to_numpy().ravel())
        tuned_pipelines[pipe_name] = copy.deepcopy(pipe_search)

        # make predictions
        if X_valid is not None and y_valid is not None:
            preds_out = pipe_search.predict(X_valid)
            if len(np.unique(preds_out)) > 2:
                valid_prob_preds[pipe_name] = preds_out
                valid_preds[pipe_name] = (preds_out > 0.5).astype(int)
            else:
                valid_preds[pipe_name] = preds_out
                if hasattr(pipe_search, "predict_proba"):
                    preds_proba_out = pipe_search.predict_proba(X_valid)
                    if preds_proba_out.ndim == 2:
                        valid_prob_preds[pipe_name] = preds_proba_out[:, 1]
                    else:
                        valid_prob_preds[pipe_name] = preds_proba_out

            # evaluate predictions
            err_out = {}
            for metric_name, metric in metrics.items():
                metric_fun = metric[0]
                metric_type = metric[1]
                if metric_type == "predict":
                    err_out[metric_name] = metric_fun(y_valid, valid_preds[pipe_name])
                elif metric_type == "predict_proba":
                    if pipe_name in valid_prob_preds.keys():
                        err_out[metric_name] = metric_fun(y_valid, valid_prob_preds[pipe_name])
                    else:
                        err_out[metric_name] = np.nan
            valid_errs[pipe_name] = copy.deepcopy(err_out)

            # evaluate predictions via bootstrap to get uncertainty estimate
            boot_valid_errs[pipe_name] = {}
            for metric_name, metric in metrics.items():
                metric_fun = metric[0]
                metric_type = metric[1]
                if metric_type == "predict":
                    y_est = valid_preds[pipe_name]
                elif metric_type == "predict_proba":
                    if pipe_name in valid_prob_preds.keys():
                        y_est = valid_prob_preds[pipe_name]
                    else:
                        continue
                boot_valid_errs_ls = []
                for b in range(B):
                    boot_idx = np.random.choice(len(y_valid), len(y_valid))
                    boot_valid_errs_ls.append(metric_fun(y_valid.iloc[boot_idx], y_est[boot_idx]))
                boot_valid_errs[pipe_name][metric_name] = copy.deepcopy(boot_valid_errs_ls)

        # evaluate feature importances
        if importance:
            best_estimator = pipe_search.best_estimator_[0]
            fi_model = fi_models_all[pipe_name]
            if fi_model is not None:
                vimp_df = fi_model(
                    best_estimator,
                    X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid
                )
                vimp_df = vimp_df.rename(columns={vimp_df.columns[0]: "var"})
                vimps[pipe_name] = copy.deepcopy(vimp_df)

    return valid_errs, valid_preds, valid_prob_preds, tuned_pipelines, vimps, boot_valid_errs
