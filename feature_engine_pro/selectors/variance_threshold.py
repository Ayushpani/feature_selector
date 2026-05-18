import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import logging

from feature_engine_pro.selectors.base_selector import _BaseSelector

class VarianceThresholdSelector(_BaseSelector):
    """
    A filter method that removes features with normalized variance below a certain threshold.
    Features are internally MinMax scaled to [0, 1] so variance is evaluated objectively 
    across all scales (max possible variance is 0.25).
    """
    def __init__(self, threshold=0.01, target_column=None, problem_type='classification'):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self._selector = VarianceThreshold(threshold=self.threshold)
        self.reporter = None
        self._scaler = MinMaxScaler()

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
            # Scale data internally to [0, 1] to calculate normalized variance objectively
            X_scaled = pd.DataFrame(self._scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)
            self._selector.fit(X_scaled)
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
                    variance_val = X_scaled[col].var()
                    self.reporter.log_event(col, 'dropped', f'Normalized Variance (σ²) = {variance_val:.4f} < {self.threshold}. Evaluated on [0,1] scale. Mathematically approaches a constant, carrying ~0 Shannon Entropy. Eliminated immediately to remove zero-signal dimensionality space.', 'VarianceThreshold')
                else:
                    variance_val = X_scaled[col].var()
                    self.reporter.log_event(col, 'kept', f'Normalized Variance (σ²) = {variance_val:.4f} >= {self.threshold}. Evaluated on [0,1] scale. Feature contains sufficient informational entropy to proceed to collinearity checks.', 'VarianceThreshold')

            # Log non-numerical as kept
            for col in non_numerical_features:
                self.reporter.log_event(col, 'kept', 'Not numerical, ignored by Variance Threshold', 'VarianceThreshold')

        return self
