import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class _BaseSelector(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract Base Class for all feature selection methods.
    Defines the interface that all selectors must implement.
    """
    def __init__(self, target_column=None, problem_type='classification'):
        self.target_column = target_column
        self.problem_type = problem_type # 'classification' or 'regression'
        self.selected_features_ = None
        self.explanation = {}
        self.original_feature_names = None

    @abstractmethod
    def fit(self, X, y=None):
        """Fits the selector to the data."""
        pass

    def transform(self, X):
        """Reduces the input X to the selected features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.selected_features_ is None:
            raise ValueError("This selector has not been fitted yet.")
        # Only keep columns that are both selected and actually present in X
        cols_to_keep = [col for col in self.selected_features_ if col in X.columns]
        return X[cols_to_keep]

    # Property for backward compatibility with previous code if needed
    @property
    def selected_features(self):
        return self.selected_features_

    @selected_features.setter
    def selected_features(self, value):
        self.selected_features_ = value
