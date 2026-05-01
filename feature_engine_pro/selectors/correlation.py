import pandas as pd
import numpy as np
from feature_engine_pro.selectors.base_selector import _BaseSelector

class CorrelationSelector(_BaseSelector):
    """
    Removes multicollinear features by keeping the one most correlated with the target.
    Can also use Variance Inflation Factor (VIF) as a secondary filter.
    """
    def __init__(self, threshold=0.8, target_column=None, problem_type='classification', method='pearson'):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self.method = method
        self.reporter = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
<<<<<<< Updated upstream

        self.original_feature_names = X.columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

=======

        self.original_feature_names = X.columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

>>>>>>> Stashed changes
        if len(numerical_cols) < 2:
            self.selected_features = X.columns.tolist()
            if self.reporter:
                for col in self.original_feature_names:
                    self.reporter.log_event(col, 'kept', 'Not enough numerical columns to compute correlation', 'CorrelationSelector')
            return self
<<<<<<< Updated upstream

        # Compute correlation matrix
        corr_matrix = X[numerical_cols].corr(method=self.method).abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

=======

        # Compute correlation matrix
        corr_matrix = X[numerical_cols].corr(method=self.method).abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

>>>>>>> Stashed changes
        # Calculate target correlation to decide which to keep
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            elif not isinstance(y, pd.Series):
                y = pd.Series(y)
            # Ensure index alignment
            y.index = X.index
            target_corr = X[numerical_cols].apply(lambda col: col.corr(y, method=self.method)).abs()
        else:
            target_corr = pd.Series(1, index=numerical_cols)  # Fallback: treat equally
<<<<<<< Updated upstream

        to_drop = set()

=======

        to_drop = set()

>>>>>>> Stashed changes
        # Identify highly correlated pairs
        for col in upper_tri.columns:
            high_corr_cols = upper_tri[col][upper_tri[col] > self.threshold].index.tolist()
            for correlated_col in high_corr_cols:
                if col not in to_drop and correlated_col not in to_drop:
                    # Drop the one less correlated with the target
                    if target_corr[col] > target_corr[correlated_col]:
                        drop_feature = correlated_col
                        keep_feature = col
                    else:
                        drop_feature = col
                        keep_feature = correlated_col
<<<<<<< Updated upstream

                    to_drop.add(drop_feature)

                    if self.reporter:
                        corr_val = upper_tri.loc[correlated_col, col]
                        self.reporter.log_event(
                            drop_feature,
                            'dropped',
                            f'Correlated {corr_val:.2f} with {keep_feature}. Kept {keep_feature} because it has higher correlation to target.',
=======

                    to_drop.add(drop_feature)

                    if self.reporter:
                        corr_val = upper_tri.loc[correlated_col, col]
                        self.reporter.log_event(
                            drop_feature,
                            'dropped',
                            f'Correlated {corr_val:.2f} with {keep_feature}. Kept {keep_feature} because it has higher correlation to target.',
>>>>>>> Stashed changes
                            'CorrelationSelector'
                        )

        self.selected_features = [c for c in self.original_feature_names if c not in to_drop]
<<<<<<< Updated upstream

        if self.reporter:
            for c in self.selected_features:
                self.reporter.log_event(c, 'kept', f'No collinearity above {self.threshold} threshold found', 'CorrelationSelector')

=======

        if self.reporter:
            for c in self.selected_features:
                self.reporter.log_event(c, 'kept', f'No collinearity above {self.threshold} threshold found', 'CorrelationSelector')

>>>>>>> Stashed changes
        return self
