"""
Base Selector — Hardened Abstract Foundation

All selectors inherit from this class. It provides:
- Input validation (same as base_transformer)
- Fitted-state tracking
- Reporter integration helpers (DRY logging)
- sklearn get_support() and get_feature_names_out() compatibility
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.logger import get_logger


class _BaseSelector(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract Base Class for all Feature Engine Pro selectors.

    Provides defensive validation, reporter integration,
    and full sklearn API compatibility (get_support, get_feature_names_out).
    """
    def __init__(self, target_column=None, problem_type='classification'):
        self.target_column = target_column
        self.problem_type = problem_type
        self.selected_features_ = None
        self.dropped_features_ = None
        self.explanation = {}
        self.original_feature_names = None
        self.reporter = None
        self.is_fitted_ = False
        self._logger = get_logger()

    def _validate_input(self, X, context='fit'):
        """Validate input DataFrame with clear error messages."""
        if X is None:
            raise ValueError(
                f"[{self.__class__.__name__}] X cannot be None during {context}()."
            )
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except Exception as e:
                raise TypeError(
                    f"[{self.__class__.__name__}] Cannot convert input to DataFrame: {e}"
                )
        if X.empty:
            raise ValueError(
                f"[{self.__class__.__name__}] Received an empty DataFrame during {context}(). "
                f"Shape: {X.shape}."
            )
        dupes = X.columns[X.columns.duplicated()].tolist()
        if dupes:
            raise ValueError(
                f"[{self.__class__.__name__}] DataFrame has duplicate column names: {dupes}."
            )
        return X

    def _check_is_fitted(self):
        """Raise an error if the selector has not been fitted."""
        if not self.is_fitted_:
            raise RuntimeError(
                f"[{self.__class__.__name__}] This selector has not been fitted yet. "
                f"Call .fit() before .transform()."
            )

    def _log_to_reporter(self, feature, status, reason, step_name):
        """
        Log an event to the reporter, if attached.

        This DRYs up the reporter logging pattern across all selectors.
        """
        if self.reporter is not None:
            self.reporter.log_event(feature, status, reason, step_name)

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the selector. Subclasses must implement this."""
        pass

    def transform(self, X):
        """Reduces the input X to the selected features."""
        self._check_is_fitted()
        X = self._validate_input(X, context='transform')

        if self.selected_features_ is None:
            raise ValueError(
                f"[{self.__class__.__name__}] No features were selected during fit."
            )

        cols_to_keep = [col for col in self.selected_features_ if col in X.columns]

        if not cols_to_keep:
            self._logger.warning(
                f"[{self.__class__.__name__}] None of the selected features are present "
                f"in the input DataFrame. Returning empty DataFrame."
            )

        return X[cols_to_keep]

    def get_support(self, indices=False):
        """
        Get a boolean mask or integer indices of selected features.

        Matches sklearn.feature_selection API.

        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices. If False, return boolean mask.

        Returns
        -------
        np.ndarray
        """
        self._check_is_fitted()
        if self.original_feature_names is None:
            raise RuntimeError("Original feature names not stored.")

        mask = np.array([f in self.selected_features_ for f in self.original_feature_names])

        if indices:
            return np.where(mask)[0]
        return mask

    def get_feature_names_out(self, input_features=None):
        """Get the names of features that survived selection."""
        self._check_is_fitted()
        if self.selected_features_ is not None:
            return np.array(self.selected_features_)
        raise RuntimeError(
            f"[{self.__class__.__name__}] Feature names not available."
        )

    # Backward compatibility properties
    @property
    def selected_features(self):
        return self.selected_features_

    @selected_features.setter
    def selected_features(self, value):
        self.selected_features_ = value
