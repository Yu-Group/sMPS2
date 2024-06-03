from os.path import join as oj
import os
import pandas as pd
import numpy as np
import copy
import random

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from imodels.importance.ppms import LogisticClassifierPPM, GlmClassifierPPM
from imodels.importance.block_transformers import CompositeTransformer, IdentityTransformer


class RandomForestPlusDistilledClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_model=None, prediction_model=None,
                 include_raw=True, drop_features=True, add_transformers=None,
                 tree_transformer="auto", center=True, normalize=False):
        super().__init__()
        if isinstance(self, RegressorMixin):
            self._task = "regression"
        elif isinstance(self, ClassifierMixin):
            self._task = "classification"
        else:
            raise ValueError("Unknown task.")
        if rf_model is None:
            if self._task == "regression":
                rf_model = RandomForestRegressor()
            elif self._task == "classification":
                rf_model = RandomForestClassifier()
        if prediction_model is None:
            if self._task == "regression":
                prediction_model = RidgeRegressorPPM()
            elif self._task == "classification":
                prediction_model = LogisticClassifierPPM()
        if tree_transformer == "auto":
            self.tree_transformer = TreeTransformer
        else:
            self.tree_transformer = tree_transformer
        self.rf_model = rf_model
        self.prediction_model = prediction_model
        self.include_raw = include_raw
        self.drop_features = drop_features
        self.add_transformers = add_transformers
        self.center = center
        self.normalize = normalize
#         self._is_ppm = isinstance(prediction_model, PartialPredictionModelBase)

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Fit (or train) Random Forest Plus (RF+) prediction model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.
        sample_weight: array-like of shape (n_samples,) or None
            Sample weights to use in random forest fit.
            If None, samples are equally weighted.
        **kwargs:
            Additional arguments to pass to self.prediction_model.fit()

        """
        self.transformer_ = []
        self.estimator_ = []
        self.feature_names_ = None
        self._n_samples_train = X.shape[0]

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise ValueError("Input X must be a pandas DataFrame or numpy array.")
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        elif not isinstance(y, np.ndarray):
            raise ValueError("Input y must be a pandas DataFrame or numpy array.")

        # fit random forest
        n_samples = X.shape[0]
        self.rf_model.fit(X, y, sample_weight=sample_weight)
        # onehot encode multiclass response for GlmClassiferPPM
        if isinstance(self.prediction_model, GlmClassifierPPM):
            self._multi_class = False
            if len(np.unique(y)) > 2:
                self._multi_class = True
                self._y_encoder = OneHotEncoder()
                y = self._y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        # fit transformer
        if self.include_raw:
            base_transformer_list = [self.tree_transformer(self.rf_model), IdentityTransformer()]
        else:
            base_transformer_list = [self.tree_transformer(self.rf_model)]
        if self.add_transformers is not None:
            base_transformer_list += self.add_transformers
        transformer = CompositeTransformer(base_transformer_list, drop_features=self.drop_features)
        blocked_data = transformer.fit_transform(X_array, center=self.center, normalize=self.normalize)
        self.prediction_model.fit(blocked_data.get_all_data(), y_train, **kwargs)
        self.estimator_ = copy.deepcopy(self.prediction_model)
        self.transformer_ = copy.deepcopy(transformer)

    def predict(self, X):
        """
        Make predictions on new data using the fitted model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix, for which to make predictions.

        Returns
        -------
        y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            The predicted values

        """
        X = check_array(X)
        check_is_fitted(self, "estimator_")
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                X_array = X.loc[:, self.feature_names_].values
            else:
                X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise ValueError("Input X must be a pandas DataFrame or numpy array.")

        blocked_data = self.transformer_.transform(X_array, center=self.center, normalize=self.normalize)
        predictions = self.estimator_.predict(blocked_data.get_all_data())
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities on new data using the fitted
        (classification) model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix, for which to make predictions.

        Returns
        -------
        y: ndarray of shape (n_samples, n_classes)
            The predicted class probabilities.

        """
        X = check_array(X)
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("'{}' object has no attribute 'predict_proba'".format(
                self.estimator_.__class__.__name__)
            )
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                X_array = X.loc[:, self.feature_names_].values
            else:
                X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise ValueError("Input X must be a pandas DataFrame or numpy array.")

        blocked_data = self.transformer_.transform(X_array, center=self.center, normalize=self.normalize)
        predictions = self.estimator_.predict_proba(blocked_data.get_all_data())
        return predictions
