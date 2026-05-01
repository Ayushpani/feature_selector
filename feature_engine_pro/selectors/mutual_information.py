"""
MutualInformationSelector — Robust Non-Linear Dependency Detection

Improvements over v1:
- Auto-detection of discrete vs continuous features
- Repeated runs (n_repeats) with averaging for stability
- Adaptive threshold option (keep top-K% by MI score)
- Proper NaN handling before MI computation
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from feature_engine_pro.selectors.base_selector import _BaseSelector
from feature_engine_pro.logger import get_logger


class MutualInformationSelector(_BaseSelector):
    """
    Selects features based on Mutual Information scores.

    Parameters
    ----------
    threshold : float, default=0.01
        Minimum MI score to keep a feature.
    target_column : str, optional
    problem_type : str, default='classification'
    n_repeats : int, default=5
        Number of repeated MI calculations to average (for stability).
    adaptive_percentile : float or None, default=None
        If set (e.g., 0.25), keep features above the given percentile of MI scores.
        Overrides `threshold`.
    auto_detect_discrete : bool, default=True
        Automatically detect which features are discrete (integer-valued with
        low cardinality) for more accurate MI estimation.
    """
    def __init__(self, threshold=0.01, target_column=None, problem_type='classification',
                 n_repeats=5, adaptive_percentile=None, auto_detect_discrete=True):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self.n_repeats = n_repeats
        self.adaptive_percentile = adaptive_percentile
        self.auto_detect_discrete = auto_detect_discrete
        self.mi_scores_ = None

    def fit(self, X, y=None):
        """Compute MI scores and select features."""
        X = self._validate_input(X, context='fit')
        self.original_feature_names = X.columns.tolist()

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        non_numerical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        if y is None or len(numerical_cols) == 0:
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            for col in self.original_feature_names:
                self._log_to_reporter(col, 'kept', 'MI skipped: no target or no numerical columns.', 'MutualInformation')
            self.is_fitted_ = True
            return self

        X_num = X[numerical_cols].fillna(0)

        # Auto-detect discrete features
        if self.auto_detect_discrete:
            discrete_mask = np.array([
                (X_num[col].dtype in [np.int64, np.int32, np.int16, np.int8] and
                 X_num[col].nunique() < 20)
                for col in numerical_cols
            ])
        else:
            discrete_mask = np.array([False] * len(numerical_cols))

        # Select MI function
        if self.problem_type == 'classification':
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        # Repeated MI computation for stability
        all_scores = []
        for i in range(self.n_repeats):
            scores = mi_func(
                X_num, y,
                discrete_features=discrete_mask,
                random_state=42 + i
            )
            all_scores.append(scores)

        # Average across runs
        avg_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)

        self.mi_scores_ = pd.DataFrame({
            'feature': numerical_cols,
            'mi_score': avg_scores,
            'mi_std': std_scores,
        }).set_index('feature')

        # Determine effective threshold
        if self.adaptive_percentile is not None:
            effective_threshold = np.percentile(avg_scores, self.adaptive_percentile * 100)
            self._logger.debug(
                f"[MutualInfo] Adaptive threshold at {self.adaptive_percentile:.0%} percentile: "
                f"{effective_threshold:.4f}"
            )
        else:
            effective_threshold = self.threshold

        mi_series = pd.Series(avg_scores, index=numerical_cols)
        selected_numerical = mi_series[mi_series >= effective_threshold].index.tolist()

        self.selected_features_ = selected_numerical + non_numerical_cols
        self.dropped_features_ = [c for c in numerical_cols if c not in selected_numerical]

        # Log to reporter
        for col in numerical_cols:
            score = mi_series[col]
            std = std_scores[numerical_cols.index(col)]
            if col in selected_numerical:
                self._log_to_reporter(
                    col, 'kept',
                    f'MI = {score:.4f} ± {std:.4f} ≥ threshold {effective_threshold:.4f}. '
                    f'({self.n_repeats} runs averaged).',
                    'MutualInformation'
                )
            else:
                self._log_to_reporter(
                    col, 'dropped',
                    f'MI = {score:.4f} ± {std:.4f} < threshold {effective_threshold:.4f}. '
                    f'Weak non-linear dependency.',
                    'MutualInformation'
                )

        for col in non_numerical_cols:
            self._log_to_reporter(col, 'kept', 'Non-numerical, skipped by MI.', 'MutualInformation')

        self._logger.info(
            f"[MutualInfo] Kept {len(selected_numerical)} of {len(numerical_cols)} numerical features "
            f"(threshold={effective_threshold:.4f}, method={self.problem_type}, "
            f"repeats={self.n_repeats})."
        )

        self.is_fitted_ = True
        return self
