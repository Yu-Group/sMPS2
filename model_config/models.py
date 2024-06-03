import numpy as np
from functools import partial

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from imodels import FIGSClassifier, RuleFitClassifier
from imodels.importance import RandomForestPlusClassifier
from imodels.importance.ppms import LogisticClassifierPPM, RidgeClassifierPPM

import sys
sys.path.append("..")
from functions.prediction_models import *
from functions.fi_models import *


CV_PARAM_GRID = {
    "randf__min_samples_leaf": [1, 3, 5, 10],
    "logistic__C": [1],
    "lasso__C": np.logspace(-3, 3, 13),
    "ridge__C": np.logspace(-3, 3, 13),
    "elnet__C": np.logspace(-3, 3, 13),
    "elnet__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
    "gb__learning_rate": [0.05, 0.1, 0.15],
    "gb__min_samples_leaf": [1, 5, 10],
    "gb__max_depth": [3, 5],
    "figs__max_rules": [5, 10, 12, 15, 20, 30, 50],
    "rfplus__prediction_model": [LogisticClassifierPPM()],
    "rulefit__max_rules": [5, 10, 30, 50]
}

MODELS = {
    "randf": RandomForestClassifier(n_estimators=500, min_samples_leaf=1),
    "gb": GradientBoostingClassifier(n_estimators=500),
    "logistic": LogisticRegression(penalty="none"),
    "lasso": LogisticRegression(penalty="l1", solver="liblinear"),
    "ridge": LogisticRegression(penalty="l2"),
    "elnet": LogisticRegression(penalty="elasticnet", solver="saga"),
    "figs": FIGSClassifier(),
    "rfplus": RandomForestPlusClassifier(),
    "rulefit": RuleFitClassifier(),
}

FI_MODELS = {
    "randf": get_feature_importances,
    "gb": None, #get_feature_importances
    "logistic": get_coefficient,
    "lasso": partial(get_coefficient, return_sparsity=True),
    "ridge": get_coefficient,
    "elnet": partial(get_coefficient, return_sparsity=True),
    "figs": None,
    "rfplus": get_mdiplus_importances,
    "rulefit": None
}
