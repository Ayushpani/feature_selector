"""
SHAPSelector — Model-Agnostic Feature Selection via SHAP

The gold standard for interpretable feature importance.
Uses SHAP (SHapley Additive exPlanations) values to rank and select features.

This is an OPTIONAL component — only available if `shap` is installed.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from feature_engine_pro.selectors.base_selector import _BaseSelector
from feature_engine_pro.logger import get_logger


def _shap_available():
    """Check if SHAP is installed."""
    try:
        import shap
        return True
    except ImportError:
        return False


class SHAPSelector(_BaseSelector):
    """
    Feature selection based on SHAP values.

    Parameters
    ----------
    threshold : float or None, default=None
        Minimum mean absolute SHAP value to keep a feature.
        If None, uses top_k_features instead.
    top_k_features : int or None, default=None
        Keep the top K features by SHAP importance.
        If None and threshold is None, keeps top 50%.
    target_column : str, optional
    problem_type : str, default='classification'
    n_estimators : int, default=100
        Number of trees for the background model.
    max_samples : int, default=500
        Maximum samples to compute SHAP values on (for performance).
    """
    def __init__(self, threshold=None, top_k_features=None,
                 target_column=None, problem_type='classification',
                 n_estimators=100, max_samples=500):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self.top_k_features = top_k_features
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.shap_values_ = None
        self.mean_abs_shap_ = None

    def fit(self, X, y=None):
        """Compute SHAP values and select features."""
        if not _shap_available():
            self._logger.warning(
                "[SHAPSelector] SHAP is not installed. "
                "Install with: pip install feature-engine-pro[explain]. "
                "Keeping all features."
            )
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            self.original_feature_names = X.columns.tolist()
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            self.is_fitted_ = True
            return self

        import shap

        X = self._validate_input(X, context='fit')
        self.original_feature_names = X.columns.tolist()

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        non_numerical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        if y is None or len(numerical_cols) == 0:
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            self.is_fitted_ = True
            return self

        X_num = X[numerical_cols].fillna(0)

        # Subsample for performance
        if len(X_num) > self.max_samples:
            sample_idx = np.random.RandomState(42).choice(
                len(X_num), self.max_samples, replace=False
            )
            X_sample = X_num.iloc[sample_idx]
            y_sample = (
                y.iloc[sample_idx] if hasattr(y, 'iloc')
                else pd.Series(y).iloc[sample_idx]
            )
        else:
            X_sample = X_num
            y_sample = y

        # Train background model
        if self.problem_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=42, n_jobs=-1
            )

        model.fit(X_sample, y_sample)

        # Compute SHAP values
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # For classification, shap_values may be a list (one per class)
            if isinstance(shap_values, list):
                # Use mean absolute across all classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values = np.abs(shap_values)

            self.shap_values_ = shap_values
            self.mean_abs_shap_ = pd.Series(
                np.mean(shap_values, axis=0),
                index=numerical_cols
            ).sort_values(ascending=False)

        except Exception as e:
            self._logger.error(
                f"[SHAPSelector] SHAP computation failed: {e}. "
                f"Falling back to feature_importances_."
            )
            importances = model.feature_importances_
            self.mean_abs_shap_ = pd.Series(importances, index=numerical_cols).sort_values(ascending=False)

        # Determine which features to keep
        if self.threshold is not None:
            selected_numerical = self.mean_abs_shap_[
                self.mean_abs_shap_ >= self.threshold
            ].index.tolist()
        elif self.top_k_features is not None:
            k = min(self.top_k_features, len(numerical_cols))
            selected_numerical = self.mean_abs_shap_.head(k).index.tolist()
        else:
            # Default: keep top 50%
            k = max(1, len(numerical_cols) // 2)
            selected_numerical = self.mean_abs_shap_.head(k).index.tolist()

        self.selected_features_ = selected_numerical + non_numerical_cols
        self.dropped_features_ = [c for c in numerical_cols if c not in selected_numerical]

        # Log to reporter
        for col in numerical_cols:
            score = self.mean_abs_shap_.get(col, 0)
            if col in selected_numerical:
                self._log_to_reporter(
                    col, 'kept',
                    f'SHAP importance: {score:.4f} (selected).',
                    'SHAP'
                )
            else:
                self._log_to_reporter(
                    col, 'dropped',
                    f'SHAP importance: {score:.4f} (below selection threshold).',
                    'SHAP'
                )

        for col in non_numerical_cols:
            self._log_to_reporter(col, 'kept', 'Non-numerical, skipped by SHAP.', 'SHAP')

        self._logger.info(
            f"[SHAP] Selected {len(selected_numerical)} of {len(numerical_cols)} features "
            f"by SHAP importance."
        )

        self.is_fitted_ = True
        return self
