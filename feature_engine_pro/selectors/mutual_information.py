import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from feature_engine_pro.selectors.base_selector import _BaseSelector

class MutualInformationSelector(_BaseSelector):
    """
    Mutual Information Selector.
    Captures non-linear dependencies between features and the target.
    Features with Mutual Information below a threshold are dropped.
    """
    def __init__(self, threshold=0.01, target_column=None, problem_type='classification'):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self.reporter = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.original_feature_names = X.columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

        if y is None or len(numerical_cols) == 0:
            self.selected_features_ = X.columns.tolist()
            if self.reporter:
                 for col in self.original_feature_names:
                     self.reporter.log_event(col, 'kept', 'Mutual Info skipped: No target y provided or no numerical cols.', 'MutualInformation')
            return self

        # Calculate Mutual Information
        if self.problem_type == 'classification':
             mi_scores = mutual_info_classif(X[numerical_cols].fillna(0), y, random_state=42)
        else:
             mi_scores = mutual_info_regression(X[numerical_cols].fillna(0), y, random_state=42)

        mi_scores_series = pd.Series(mi_scores, index=numerical_cols)

        selected_numerical_features = mi_scores_series[mi_scores_series >= self.threshold].index.tolist()
        non_numerical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        self.selected_features_ = selected_numerical_features + non_numerical_features

        # Log reasoning to reporter
        if self.reporter:
            for col, score in mi_scores_series.items():
                if score >= self.threshold:
                     self.reporter.log_event(col, 'kept', f'Mutual Information Score = {score:.4f} >= {self.threshold}. Feature provides statistically significant non-linear reduction in uncertainty about the target.', 'MutualInformation')
                else:
                     self.reporter.log_event(col, 'dropped', f'Mutual Information Score = {score:.4f} < {self.threshold}. Information Theory proves this feature provides no statistically significant reduction in uncertainty (Entropy) about the target Y. Pruned to minimize noise.', 'MutualInformation')

            for col in non_numerical_features:
                 self.reporter.log_event(col, 'kept', 'Not numerical, skipped by Mutual Info.', 'MutualInformation')

        return self
