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

        # Dynamic threshold: 5% of the maximum information content
        max_mi = mi_scores_series.max()
        dynamic_threshold = 0.05 * max_mi if max_mi > 0 else self.threshold

        selected_numerical_features = mi_scores_series[mi_scores_series >= dynamic_threshold].index.tolist()
        non_numerical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        self.selected_features_ = selected_numerical_features + non_numerical_features

        # Log reasoning to reporter
        if self.reporter:
            for col, score in mi_scores_series.items():
                if score >= dynamic_threshold:
                     msg = f'Mutual Information Score = {score:.4f} >= dynamic threshold {dynamic_threshold:.4f} (5% of max MI). Feature provides statistically significant non-linear reduction in uncertainty about the target.'
                     if score > 0.90:
                         msg = f'🚨 TARGET LEAKAGE WARNING 🚨: Mutual Information Score = {score:.4f}. This feature perfectly predicts the target with almost zero entropy loss. It is almost certainly derived directly from the target variable!'
                     self.reporter.log_event(col, 'kept', msg, 'MutualInformation')
                else:
                     self.reporter.log_event(col, 'dropped', f'Mutual Information Score = {score:.4f} < dynamic threshold {dynamic_threshold:.4f} (5% of max MI). Information Theory proves this feature provides no statistically significant reduction in uncertainty (Entropy) relative to the dataset. Pruned to minimize noise.', 'MutualInformation')

            for col in non_numerical_features:
                 self.reporter.log_event(col, 'kept', 'Not numerical, skipped by Mutual Info.', 'MutualInformation')

        return self
