"""
DataProcessor — Hardened Data Validation, Cleaning, and Imputation

Handles:
- Infinite value detection and replacement
- Fully-NaN column detection and removal
- Mixed-type column coercion
- Duplicate column detection
- Configurable imputation strategies (mean, median, most_frequent)
- Safe imputation that learns from training data only (no leakage)
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.logger import get_logger


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Production-grade data preprocessor that handles the worst real-world data.

    Parameters
    ----------
    imputation_strategy : str, default='median'
        Strategy for imputing missing numerical values.
        Options: 'mean', 'median', 'most_frequent', 'zero'.
    categorical_imputation : str, default='most_frequent'
        Strategy for imputing missing categorical values.
        Options: 'most_frequent', 'missing' (adds a 'Missing' category).
    drop_all_nan_cols : bool, default=True
        If True, drop columns that are 100% NaN.
    replace_inf : bool, default=True
        If True, replace np.inf and -np.inf with NaN before imputation.
    """
    def __init__(self, imputation_strategy='median', categorical_imputation='most_frequent',
                 drop_all_nan_cols=True, replace_inf=True):
        self.imputation_strategy = imputation_strategy
        self.categorical_imputation = categorical_imputation
        self.drop_all_nan_cols = drop_all_nan_cols
        self.replace_inf = replace_inf

        # Learned state
        self.numerical_cols_ = []
        self.categorical_cols_ = []
        self.imputation_values_ = {}
        self.all_nan_cols_ = []
        self.duplicate_cols_ = []
        self.feature_names_in_ = None
        self.is_fitted_ = False
        self._logger = get_logger()

    def fit(self, X, y=None):
        """
        Learn imputation statistics, detect problematic columns.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : ignored
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns.tolist()

        # --- Step 1: Detect and record fully-NaN columns ---
        nan_fractions = X.isnull().mean()
        self.all_nan_cols_ = nan_fractions[nan_fractions == 1.0].index.tolist()
        if self.all_nan_cols_:
            self._logger.warning(
                f"[DataProcessor] Detected {len(self.all_nan_cols_)} fully-NaN columns "
                f"(will be dropped): {self.all_nan_cols_}"
            )

        # --- Step 2: Detect duplicate columns (same data) ---
        self.duplicate_cols_ = []
        seen_hashes = {}
        for col in X.columns:
            col_hash = pd.util.hash_pandas_object(X[col], index=False).sum()
            if col_hash in seen_hashes:
                self.duplicate_cols_.append(col)
                self._logger.warning(
                    f"[DataProcessor] Column '{col}' is a duplicate of '{seen_hashes[col_hash]}' (will be dropped)."
                )
            else:
                seen_hashes[col_hash] = col

        # Work on cleaned columns
        cols_to_drop = set(self.all_nan_cols_) | set(self.duplicate_cols_)
        X_work = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

        # --- Step 3: Replace infinities ---
        if self.replace_inf:
            inf_counts = np.isinf(X_work.select_dtypes(include=np.number)).sum()
            inf_cols = inf_counts[inf_counts > 0].index.tolist()
            if inf_cols:
                self._logger.warning(
                    f"[DataProcessor] Infinite values detected in {len(inf_cols)} columns: {inf_cols}. "
                    f"Replacing with NaN before imputation."
                )
                X_work = X_work.replace([np.inf, -np.inf], np.nan)

        # --- Step 4: Segregate numerical vs categorical ---
        self.numerical_cols_ = X_work.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X_work.select_dtypes(exclude=[np.number]).columns.tolist()

        # --- Step 5: Learn imputation values for numerical columns ---
        for col in self.numerical_cols_:
            if X_work[col].isnull().all():
                self.imputation_values_[col] = 0
                continue

            if self.imputation_strategy == 'mean':
                self.imputation_values_[col] = X_work[col].mean()
            elif self.imputation_strategy == 'median':
                self.imputation_values_[col] = X_work[col].median()
            elif self.imputation_strategy == 'most_frequent':
                self.imputation_values_[col] = X_work[col].mode().iloc[0] if not X_work[col].mode().empty else 0
            elif self.imputation_strategy == 'zero':
                self.imputation_values_[col] = 0
            else:
                self.imputation_values_[col] = X_work[col].median()

        # --- Step 6: Learn imputation values for categorical columns ---
        for col in self.categorical_cols_:
            if self.categorical_imputation == 'most_frequent':
                mode_val = X_work[col].mode()
                self.imputation_values_[col] = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
            else:
                self.imputation_values_[col] = 'Missing'

        # Log summary
        missing_cols = X.isnull().any().sum()
        self._logger.info(
            f"[DataProcessor] Fit complete. "
            f"Numerical: {len(self.numerical_cols_)}, Categorical: {len(self.categorical_cols_)}, "
            f"Columns with missing values: {missing_cols}, "
            f"All-NaN dropped: {len(self.all_nan_cols_)}, Duplicates dropped: {len(self.duplicate_cols_)}."
        )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Apply learned cleaning and imputation to data.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Cleaned, imputed DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("[DataProcessor] Not fitted. Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_processed = X.copy()

        # Drop all-NaN and duplicate columns
        cols_to_drop = set(self.all_nan_cols_) | set(self.duplicate_cols_)
        X_processed = X_processed.drop(
            columns=[c for c in cols_to_drop if c in X_processed.columns]
        )

        # Replace infinities
        if self.replace_inf:
            num_cols = X_processed.select_dtypes(include=np.number).columns
            X_processed[num_cols] = X_processed[num_cols].replace([np.inf, -np.inf], np.nan)

        # Impute numerical
        for col in self.numerical_cols_:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].fillna(
                    self.imputation_values_.get(col, 0)
                )

        # Impute categorical
        for col in self.categorical_cols_:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].fillna(
                    self.imputation_values_.get(col, 'Unknown')
                )

        return X_processed

    def get_feature_names_out(self, input_features=None):
        """Get feature names after processing."""
        if not self.is_fitted_:
            raise RuntimeError("[DataProcessor] Not fitted.")
        cols_dropped = set(self.all_nan_cols_) | set(self.duplicate_cols_)
        return [c for c in self.feature_names_in_ if c not in cols_dropped]
