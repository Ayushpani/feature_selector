"""
RFESelector — Cross-Validated Recursive Feature Elimination

Improvements over v1:
- RFECV (cross-validated) to automatically find optimal number of features
- Custom estimator support (RF, GBM, XGBoost, LightGBM)
- Permutation importance fallback for estimators without feature_importances_
- Adaptive step size for large feature sets
- Robust error handling
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold

from feature_engine_pro.selectors.base_selector import _BaseSelector
from feature_engine_pro.logger import get_logger


class RFESelector(_BaseSelector):
    """
    Recursive Feature Elimination with optional cross-validation.

    Parameters
    ----------
    n_features_to_select : int or None, default=None
        Target number of features. If None and use_cv=True, RFECV auto-selects.
        If None and use_cv=False, selects half.
    step : int or float, default='auto'
        Features to remove per iteration. 'auto' uses 1 for ≤30 features,
        10% for >30 features.
    target_column : str, optional
    problem_type : str, default='classification'
    estimator : estimator or str, default='random_forest'
        Estimator for RFE. Can be 'random_forest', 'gradient_boosting',
        or a fitted/unfitted sklearn estimator.
    use_cv : bool, default=True
        If True, use RFECV to auto-determine optimal feature count.
    cv_folds : int, default=5
        Number of cross-validation folds (when use_cv=True).
    n_estimators : int, default=100
        Number of trees for ensemble estimators.
    """
    def __init__(self, n_features_to_select=None, step='auto',
                 target_column=None, problem_type='classification',
                 estimator='random_forest', use_cv=True, cv_folds=5,
                 n_estimators=100):
        super().__init__(target_column, problem_type)
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.estimator = estimator
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.feature_importances_ = None
        self.rfe_ranking_ = None
        self.optimal_n_features_ = None

    def _get_estimator(self):
        """Resolve estimator parameter to an actual sklearn estimator."""
        if isinstance(self.estimator, str):
            if self.estimator == 'random_forest':
                if self.problem_type == 'classification':
                    return RandomForestClassifier(
                        n_estimators=self.n_estimators, random_state=42, n_jobs=-1
                    )
                else:
                    return RandomForestRegressor(
                        n_estimators=self.n_estimators, random_state=42, n_jobs=-1
                    )
            elif self.estimator == 'gradient_boosting':
                if self.problem_type == 'classification':
                    return GradientBoostingClassifier(
                        n_estimators=self.n_estimators, random_state=42
                    )
                else:
                    return GradientBoostingRegressor(
                        n_estimators=self.n_estimators, random_state=42
                    )
            else:
                raise ValueError(f"Unknown estimator string: '{self.estimator}'")
        else:
            # User passed an estimator instance
            return self.estimator

    def _get_step(self, n_features):
        """Determine step size based on feature count."""
        if self.step == 'auto':
            if n_features <= 30:
                return 1
            else:
                return max(1, int(n_features * 0.1))
        return self.step

    def fit(self, X, y=None):
        """Fit RFE/RFECV and select features."""
        X = self._validate_input(X, context='fit')
        self.original_feature_names = X.columns.tolist()

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        non_numerical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        if y is None or len(numerical_cols) == 0:
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            for col in self.original_feature_names:
                self._log_to_reporter(col, 'kept', 'RFE skipped: no target or no numerical columns.', 'RFE')
            self.is_fitted_ = True
            return self

        X_num = X[numerical_cols].fillna(0)
        est = self._get_estimator()
        step = self._get_step(len(numerical_cols))

        try:
            if self.use_cv and self.n_features_to_select is None:
                # --- RFECV: auto-find optimal feature count ---
                if self.problem_type == 'classification':
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

                rfe = RFECV(
                    estimator=est,
                    step=step,
                    cv=cv,
                    scoring='accuracy' if self.problem_type == 'classification' else 'r2',
                    n_jobs=-1,
                    min_features_to_select=max(1, len(numerical_cols) // 4)
                )
                rfe.fit(X_num, y)
                self.optimal_n_features_ = rfe.n_features_

                self._logger.info(
                    f"[RFE] RFECV determined optimal feature count: {self.optimal_n_features_} "
                    f"(from {len(numerical_cols)} candidates, {self.cv_folds}-fold CV)."
                )
            else:
                # --- Standard RFE with specified n_features ---
                n_select = self.n_features_to_select
                if n_select is None:
                    n_select = max(1, len(numerical_cols) // 2)
                n_select = min(n_select, len(numerical_cols))
                self.optimal_n_features_ = n_select

                rfe = RFE(
                    estimator=est,
                    n_features_to_select=n_select,
                    step=step
                )
                rfe.fit(X_num, y)

            # Extract results
            mask = rfe.support_
            self.rfe_ranking_ = dict(zip(numerical_cols, rfe.ranking_))
            selected_numerical = [col for col, sel in zip(numerical_cols, mask) if sel]

            # Try to get feature importances from the underlying estimator
            try:
                if hasattr(rfe.estimator_, 'feature_importances_'):
                    importances = rfe.estimator_.feature_importances_
                    self.feature_importances_ = dict(zip(selected_numerical, importances))
            except Exception:
                pass

        except Exception as e:
            self._logger.error(
                f"[RFE] RFE fitting failed: {e}. Falling back to permutation importance."
            )
            # Fallback: use permutation importance
            try:
                from sklearn.inspection import permutation_importance
                est.fit(X_num, y)
                result = permutation_importance(est, X_num, y, n_repeats=10, random_state=42, n_jobs=-1)
                importances = result.importances_mean

                n_select = self.n_features_to_select or max(1, len(numerical_cols) // 2)
                n_select = min(n_select, len(numerical_cols))

                sorted_idx = np.argsort(importances)[::-1]
                selected_indices = sorted_idx[:n_select]
                selected_numerical = [numerical_cols[i] for i in selected_indices]
                self.rfe_ranking_ = {
                    col: rank + 1 for rank, col in
                    enumerate(np.array(numerical_cols)[sorted_idx])
                }
                self.optimal_n_features_ = n_select
            except Exception as e2:
                self._logger.error(f"[RFE] Permutation importance also failed: {e2}. Keeping all.")
                selected_numerical = numerical_cols
                self.rfe_ranking_ = {col: 1 for col in numerical_cols}
                self.optimal_n_features_ = len(numerical_cols)

        self.selected_features_ = selected_numerical + non_numerical_cols
        self.dropped_features_ = [c for c in numerical_cols if c not in selected_numerical]

        # Log to reporter
        for col in numerical_cols:
            rank = self.rfe_ranking_.get(col, 999)
            if col in selected_numerical:
                self._log_to_reporter(
                    col, 'kept',
                    f'RFE Rank {rank} (selected, top-tier feature importance).',
                    'RFE'
                )
            else:
                self._log_to_reporter(
                    col, 'dropped',
                    f'RFE Rank {rank}. Eliminated: low tree-based feature importance.',
                    'RFE'
                )

        for col in non_numerical_cols:
            self._log_to_reporter(col, 'kept', 'Non-numerical, skipped by RFE.', 'RFE')

        self._logger.info(
            f"[RFE] Selected {len(selected_numerical)} of {len(numerical_cols)} features "
            f"(optimal={self.optimal_n_features_})."
        )

        self.is_fitted_ = True
        return self
