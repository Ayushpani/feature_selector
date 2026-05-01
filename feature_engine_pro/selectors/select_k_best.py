"""
SelectKBestSelector — Statistical Hypothesis Testing for Feature Selection

Uses classical statistical tests to identify features with significant
relationships to the target variable.

Methods:
- Classification: ANOVA F-test (f_classif), Chi-squared test (chi2)
- Regression: F-test (f_regression)

Applies Bonferroni correction for multiple hypothesis testing to control
false positive rate.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, f_regression, chi2

from feature_engine_pro.selectors.base_selector import _BaseSelector
from feature_engine_pro.logger import get_logger


class StatisticalSelector(_BaseSelector):
    """
    Selects features using statistical hypothesis tests.

    Parameters
    ----------
    significance_level : float, default=0.05
        P-value threshold for feature significance.
    target_column : str, optional
    problem_type : str, default='classification'
    test_method : str, default='auto'
        'auto' (picks best test), 'f_test', 'chi2'.
    bonferroni_correction : bool, default=True
        Apply Bonferroni correction for multiple testing.
    """
    def __init__(self, significance_level=0.05, target_column=None,
                 problem_type='classification', test_method='auto',
                 bonferroni_correction=True):
        super().__init__(target_column, problem_type)
        self.significance_level = significance_level
        self.test_method = test_method
        self.bonferroni_correction = bonferroni_correction
        self.test_results_ = None

    def fit(self, X, y=None):
        """Run statistical tests and select significant features."""
        X = self._validate_input(X, context='fit')
        self.original_feature_names = X.columns.tolist()

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        non_numerical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        if y is None or len(numerical_cols) == 0:
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            for col in self.original_feature_names:
                self._log_to_reporter(col, 'kept', 'Statistical test skipped: no target or no numerical columns.', 'StatisticalTest')
            self.is_fitted_ = True
            return self

        X_num = X[numerical_cols].fillna(0)

        # Select test function
        if self.test_method == 'auto':
            if self.problem_type == 'classification':
                test_func = f_classif
                test_name = 'ANOVA F-test'
            else:
                test_func = f_regression
                test_name = 'F-regression'
        elif self.test_method == 'chi2':
            test_func = chi2
            test_name = 'Chi-squared'
            # Chi-squared requires non-negative values
            X_num = X_num.clip(lower=0)
        elif self.test_method == 'f_test':
            if self.problem_type == 'classification':
                test_func = f_classif
                test_name = 'ANOVA F-test'
            else:
                test_func = f_regression
                test_name = 'F-regression'
        else:
            raise ValueError(f"Unknown test method: {self.test_method}")

        try:
            f_scores, p_values = test_func(X_num, y)
        except Exception as e:
            self._logger.error(
                f"[StatisticalTest] Test '{test_name}' failed: {e}. Keeping all features."
            )
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            self.is_fitted_ = True
            return self

        # Handle NaN p-values (can happen with constant features)
        p_values = np.where(np.isnan(p_values), 1.0, p_values)
        f_scores = np.where(np.isnan(f_scores), 0.0, f_scores)

        # Bonferroni correction
        effective_alpha = self.significance_level
        if self.bonferroni_correction:
            effective_alpha = self.significance_level / len(numerical_cols)

        self.test_results_ = pd.DataFrame({
            'feature': numerical_cols,
            'f_score': f_scores,
            'p_value': p_values,
            'significant': p_values < effective_alpha,
        }).set_index('feature')

        selected_numerical = self.test_results_[self.test_results_['significant']].index.tolist()

        # Guard: if nothing passes, keep top 50% by f-score
        if not selected_numerical:
            self._logger.warning(
                f"[StatisticalTest] No features passed significance test "
                f"(α={effective_alpha:.6f}). Keeping top 50% by F-score."
            )
            n_keep = max(1, len(numerical_cols) // 2)
            selected_numerical = (
                self.test_results_.nlargest(n_keep, 'f_score').index.tolist()
            )

        self.selected_features_ = selected_numerical + non_numerical_cols
        self.dropped_features_ = [c for c in numerical_cols if c not in selected_numerical]

        # Log to reporter
        for col in numerical_cols:
            row = self.test_results_.loc[col]
            if col in selected_numerical:
                self._log_to_reporter(
                    col, 'kept',
                    f'{test_name}: F={row["f_score"]:.2f}, p={row["p_value"]:.2e} '
                    f'(significant at α={effective_alpha:.4f}'
                    f'{" with Bonferroni" if self.bonferroni_correction else ""}).',
                    'StatisticalTest'
                )
            else:
                self._log_to_reporter(
                    col, 'dropped',
                    f'{test_name}: F={row["f_score"]:.2f}, p={row["p_value"]:.2e} '
                    f'(not significant at α={effective_alpha:.4f}'
                    f'{" with Bonferroni" if self.bonferroni_correction else ""}).',
                    'StatisticalTest'
                )

        for col in non_numerical_cols:
            self._log_to_reporter(col, 'kept', 'Non-numerical, skipped.', 'StatisticalTest')

        self._logger.info(
            f"[StatisticalTest] {test_name}: {len(selected_numerical)} of {len(numerical_cols)} "
            f"features significant (α={effective_alpha:.6f}"
            f"{', Bonferroni-corrected' if self.bonferroni_correction else ''})."
        )

        self.is_fitted_ = True
        return self
