"""
VarianceThresholdSelector — Enhanced Variance-Based Feature Filter

Improvements over v1:
- Quasi-constant detection (>99% same value)
- Scaled variance option (normalize before comparison)
- Binary feature safeguard (won't accidentally kill boolean features)
- Proper reporter integration via base class helpers
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from feature_engine_pro.selectors.base_selector import _BaseSelector
from feature_engine_pro.logger import get_logger


class VarianceThresholdSelector(_BaseSelector):
    """
    Removes features with low variance or near-constant distributions.

    Parameters
    ----------
    threshold : float, default=0.01
        Minimum variance required to keep a feature.
    target_column : str, optional
    problem_type : str, default='classification'
    quasi_constant_threshold : float, default=0.99
        If a single value dominates > this fraction of a feature, it's quasi-constant
        and will be dropped regardless of numerical variance.
    scale_before_check : bool, default=False
        If True, normalize features to [0, 1] before computing variance.
        Useful when features have very different scales.
    protect_binary : bool, default=True
        If True, binary/boolean features are never dropped by variance filtering.
    """
    def __init__(self, threshold=0.01, target_column=None, problem_type='classification',
                 quasi_constant_threshold=0.99, scale_before_check=False,
                 protect_binary=True):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self.quasi_constant_threshold = quasi_constant_threshold
        self.scale_before_check = scale_before_check
        self.protect_binary = protect_binary

    def fit(self, X, y=None):
        """Fit: identify features to keep based on variance analysis."""
        X = self._validate_input(X, context='fit')
        self.original_feature_names = X.columns.tolist()

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        non_numerical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        if not numerical_cols:
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            for col in X.columns:
                self._log_to_reporter(col, 'kept', 'Not numerical, skipped.', 'VarianceThreshold')
            self.is_fitted_ = True
            return self

        to_drop = set()
        variance_info = {}

        for col in numerical_cols:
            series = X[col].dropna()
            variance = float(series.var()) if len(series) > 0 else 0.0
            n_unique = series.nunique()
            is_binary = n_unique <= 2

            # --- Check 1: Quasi-constant detection ---
            if len(series) > 0:
                top_freq = series.value_counts(normalize=True).iloc[0]
                if top_freq >= self.quasi_constant_threshold:
                    to_drop.add(col)
                    reason = (
                        f"Quasi-constant: {top_freq:.1%} of values are "
                        f"'{series.value_counts().index[0]}'. "
                        f"Threshold: {self.quasi_constant_threshold:.0%}."
                    )
                    self._log_to_reporter(col, 'dropped', reason, 'VarianceThreshold')
                    variance_info[col] = {'variance': variance, 'reason': reason, 'status': 'dropped'}
                    continue

            # --- Check 2: Binary protection ---
            if is_binary and self.protect_binary:
                reason = f"Binary feature protected. Variance: {variance:.6f}."
                self._log_to_reporter(col, 'kept', reason, 'VarianceThreshold')
                variance_info[col] = {'variance': variance, 'reason': reason, 'status': 'kept'}
                continue

            # --- Check 3: Variance threshold ---
            check_variance = variance
            if self.scale_before_check and len(series) > 0:
                range_val = series.max() - series.min()
                if range_val > 0:
                    scaled = (series - series.min()) / range_val
                    check_variance = float(scaled.var())

            if check_variance < self.threshold:
                to_drop.add(col)
                reason = (
                    f"Variance {variance:.6f} "
                    f"{'(scaled: ' + f'{check_variance:.6f})' if self.scale_before_check else ''}"
                    f" below threshold {self.threshold}."
                )
                self._log_to_reporter(col, 'dropped', reason, 'VarianceThreshold')
                variance_info[col] = {'variance': variance, 'reason': reason, 'status': 'dropped'}
            else:
                reason = f"Variance {variance:.6f} above threshold {self.threshold}."
                self._log_to_reporter(col, 'kept', reason, 'VarianceThreshold')
                variance_info[col] = {'variance': variance, 'reason': reason, 'status': 'kept'}

        # Non-numerical columns always pass through
        for col in non_numerical_cols:
            self._log_to_reporter(col, 'kept', 'Non-numerical, skipped.', 'VarianceThreshold')

        self.selected_features_ = [c for c in self.original_feature_names if c not in to_drop]
        self.dropped_features_ = list(to_drop)

        self._logger.info(
            f"[VarianceThreshold] Kept {len(self.selected_features_)}, "
            f"dropped {len(self.dropped_features_)} features."
        )

        self.is_fitted_ = True
        return self
