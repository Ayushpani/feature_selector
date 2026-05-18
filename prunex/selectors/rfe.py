import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from prunex.selectors.base_selector import _BaseSelector

class RFESelector(_BaseSelector):
    """
    Recursive Feature Elimination (RFE) Selector.
    Uses a Tree-based estimator to recursively prune the least important features.
    """
    def __init__(self, n_features_to_select=None, step=1, target_column=None, problem_type='classification'):
        super().__init__(target_column, problem_type)
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.reporter = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.original_feature_names = X.columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

        # We need a target variable y to train the model for RFE
        if y is None or len(numerical_cols) <= 1:
            self.selected_features_ = X.columns.tolist()
            if self.reporter:
                 for col in self.original_feature_names:
                     self.reporter.log_event(col, 'kept', 'RFE skipped: No target y provided or not enough numerical cols.', 'RFE')
            return self

        # Initialize the appropriate estimator based on the problem type
        if self.problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        # Set n_features_to_select to half if not provided
        n_select = self.n_features_to_select
        if n_select is None:
             n_select = max(1, len(numerical_cols) // 2)

        # Guard against n_features_to_select > n_features
        original_n_select = n_select
        n_select = min(n_select, len(numerical_cols))
        if original_n_select > len(numerical_cols):
             import warnings
             warnings.warn(f"rfe_n_features ({original_n_select}) is greater than the number of remaining numerical features ({len(numerical_cols)}). Automatically adjusting to {n_select}.", UserWarning)

        rfe = RFE(estimator=estimator, n_features_to_select=n_select, step=self.step)

        # Fit RFE only on numerical features
        rfe.fit(X[numerical_cols], y)

        selected_numerical_mask = rfe.support_
        selected_numerical_features = [col for idx, col in enumerate(numerical_cols) if selected_numerical_mask[idx]]

        non_numerical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        self.selected_features_ = selected_numerical_features + non_numerical_features

        # Target Leakage Detection
        leakage_warning_col = None
        if hasattr(rfe, 'estimator_') and hasattr(rfe.estimator_, 'feature_importances_'):
            importances = rfe.estimator_.feature_importances_
            if len(importances) > 0 and importances.max() > 0.90:
                leakage_idx = np.argmax(importances)
                if leakage_idx < len(selected_numerical_features):
                    leakage_warning_col = selected_numerical_features[leakage_idx]

        # Map selected features to their importances
        importance_map = {}
        if hasattr(rfe, 'estimator_') and hasattr(rfe.estimator_, 'feature_importances_'):
            importances = rfe.estimator_.feature_importances_
            for idx, col in enumerate(selected_numerical_features):
                if idx < len(importances):
                    importance_map[col] = importances[idx]

        # Log reasoning to reporter
        if self.reporter:
            for idx, col in enumerate(numerical_cols):
                rank = rfe.ranking_[idx]
                if rank == 1:
                    importance = importance_map.get(col, 0.0)
                    msg = f'RFE Rank: 1 (Selected Tier - Gini Importance: {importance:.2%}). Tree ensemble MDI indicates mathematically superior split-optimization synergy with other variables.'
                    if col == leakage_warning_col:
                        msg = f'🚨 TARGET LEAKAGE WARNING: RFE Rank: 1 (Selected Tier - Gini Importance: {importance:.2%}). This feature alone dominates >90% of the ensemble\'s Gini importance. It is almost certainly a direct proxy for the target variable!'
                    self.reporter.log_event(col, 'kept', msg, 'RFE')
                else:
                    self.reporter.log_event(col, 'dropped', f'Dropped: RFE Rank {rank}. Pruned by Random Forest Ensemble. Mean Decrease in Impurity (MDI) proves it contributes no meaningful split-optimization (Gini reduction), even when evaluated synergistically.', 'RFE')

            for col in non_numerical_features:
                 self.reporter.log_event(col, 'kept', 'Not numerical, skipped by RFE.', 'RFE')

        return self
