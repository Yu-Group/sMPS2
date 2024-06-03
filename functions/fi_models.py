import pandas as pd
import numpy as np
from imodels.importance.rf_plus import _fast_r2_score


def get_coefficient(estimator, abs=True, return_sparsity=False, **kwargs):
    vimp_vals = estimator.coef_.T
    if abs:
        vimp_vals = np.abs(vimp_vals)
    vimp_df = pd.DataFrame(vimp_vals, columns=["importance"]).reset_index()
    if return_sparsity:
        vimp_df["sparsity"] = (vimp_df["importance"] != 0).astype(int)
    return vimp_df


def get_feature_importances(estimator, **kwargs):
    vimp_vals = estimator.feature_importances_
    vimp_df = pd.DataFrame(vimp_vals, columns=["importance"]).reset_index()
    return vimp_df


def get_mdiplus_importances(estimator, **kwargs):
    vimp_df = estimator.get_mdi_plus_scores(
        kwargs["X_train"].to_numpy(), kwargs["y_train"].to_numpy(), scoring_fns={"importance": _fast_r2_score}
    )
    return vimp_df