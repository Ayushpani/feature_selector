import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import logging

from feature_engine_pro.selectors.base_selector import _BaseSelector

class VarianceThresholdSelector(_BaseSelector):
    """
    A filter method that removes features with variance below a certain threshold.
    Features with zero variance (i.e., constant features) are always removed if threshold is 0.0.
    """
    def __init__(self, threshold=0.0, target_column=None, problem_type='classification'):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self._selector = VarianceThreshold(threshold=self.threshold)
        self.reporter = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.original_feature_names = X.columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        if not numerical_cols:
            self.selected_features = X.columns.tolist()
            if self.reporter:
                for col in X.columns:
                    self.reporter.log_event(col, 'kept', 'Not numerical, Variance Threshold skipped', 'VarianceThreshold')
            return self

        try:
            self._selector.fit(X[numerical_cols])
        except Exception as e:
            logging.error(f"Error fitting VarianceThresholdSelector: {e}. Retaining all features.")
            self.selected_features = X.columns.tolist()
            return self

        selected_numerical_mask = self._selector.get_support()
        selected_numerical_features = [col for idx, col in enumerate(numerical_cols) if selected_numerical_mask[idx]]
        non_numerical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        self.selected_features = selected_numerical_features + non_numerical_features

        if self.reporter:
            # Log removed features
            for col in numerical_cols:
                if col not in selected_numerical_features:
                    variance_val = X[col].var()
                    self.reporter.log_event(col, 'dropped', f'Variance {variance_val:.4f} is below threshold {self.threshold}', 'VarianceThreshold')
                else:
                    variance_val = X[col].var()
                    self.reporter.log_event(col, 'kept', f'Variance {variance_val:.4f} is above threshold {self.threshold}', 'VarianceThreshold')

            # Log non-numerical as kept
            for col in non_numerical_features:
                self.reporter.log_event(col, 'kept', 'Not numerical, ignored by Variance Threshold', 'VarianceThreshold')

        return self
