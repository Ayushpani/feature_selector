"""
AutoCategoricalEncoder — Production-Grade Categorical Encoding

Critical fix: The original implementation had target encoding data leakage risk.

This version implements:
- Smoothed (Bayesian) Target Encoding to prevent overfitting on rare categories
- Frequency Encoding as an alternative for high-cardinality columns
- Ordinal Encoding fallback when no target is provided
- Global mean imputation for unseen categories at inference time
- High-cardinality warnings
"""
import pandas as pd
import numpy as np
from feature_engine_pro.transformers.base_transformer import _BaseTransformer
from feature_engine_pro.logger import get_logger


class AutoCategoricalEncoder(_BaseTransformer):
    """
    Converts categorical columns into mathematical representations.

    Parameters
    ----------
    target_column : str, optional
        Name of the target column (for reference/logging only).
    problem_type : str, default='classification'
        'classification' or 'regression'.
    encoding_method : str, default='target'
        Primary encoding method: 'target' (smoothed target encoding),
        'frequency' (frequency encoding), 'ordinal' (ordinal encoding).
    smoothing : float, default=10.0
        Smoothing factor for target encoding (Bayesian regularization).
        Higher values pull rare categories closer to the global mean.
        Formula: encoded = (count * cat_mean + m * global_mean) / (count + m)
    high_cardinality_threshold : int, default=50
        If a column has more unique values than this, log a warning.
    frequency_encoding_threshold : int, default=100
        If a column has more unique values than this and method is 'target',
        auto-switch to frequency encoding to avoid instability.
    """
    def __init__(self, target_column=None, problem_type='classification',
                 encoding_method='target', smoothing=10.0,
                 high_cardinality_threshold=50,
                 frequency_encoding_threshold=100):
        super().__init__()
        self.target_column = target_column
        self.problem_type = problem_type
        self.encoding_method = encoding_method
        self.smoothing = smoothing
        self.high_cardinality_threshold = high_cardinality_threshold
        self.frequency_encoding_threshold = frequency_encoding_threshold

        self.encoding_maps_ = {}
        self.categorical_cols_ = []
        self.global_mean_ = None
        self.encoding_methods_used_ = {}
        self._logger = get_logger()

    def fit(self, X, y=None):
        """Learn encoding maps from training data."""
        X = self._validate_input(X, context='fit')
        self.feature_names_in_ = X.columns.tolist()

        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Compute global mean of target (used for smoothing and unseen categories)
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            elif not isinstance(y, pd.Series):
                y = pd.Series(y, index=X.index)
            self.global_mean_ = float(y.mean())
        else:
            self.global_mean_ = 0.0

        for col in self.categorical_cols_:
            n_unique = X[col].nunique()

            # High cardinality warning
            if n_unique > self.high_cardinality_threshold:
                self._logger.warning(
                    f"[Encoder] Column '{col}' has {n_unique} unique values (high cardinality). "
                    f"Consider reviewing encoding strategy."
                )

            # Determine encoding method for this column
            method = self.encoding_method
            if method == 'target' and n_unique > self.frequency_encoding_threshold:
                method = 'frequency'
                self._logger.info(
                    f"[Encoder] Column '{col}' has {n_unique} unique values, "
                    f"auto-switching from target to frequency encoding."
                )

            self.encoding_methods_used_[col] = method

            if method == 'target' and y is not None:
                # --- Smoothed (Bayesian) Target Encoding ---
                temp_df = pd.DataFrame({col: X[col], 'target': y.values})
                agg = temp_df.groupby(col)['target'].agg(['mean', 'count'])

                # Bayesian smoothing: (count * cat_mean + m * global_mean) / (count + m)
                smoothed = (
                    (agg['count'] * agg['mean'] + self.smoothing * self.global_mean_) /
                    (agg['count'] + self.smoothing)
                )
                self.encoding_maps_[col] = smoothed.to_dict()
                self.explanation[col] = (
                    f"Smoothed Target Encoded (m={self.smoothing}). "
                    f"{n_unique} categories."
                )

            elif method == 'frequency':
                # --- Frequency Encoding ---
                freq = X[col].value_counts(normalize=True)
                self.encoding_maps_[col] = freq.to_dict()
                self.explanation[col] = f"Frequency Encoded. {n_unique} categories."

            else:
                # --- Ordinal Encoding (fallback) ---
                unique_vals = X[col].dropna().unique()
                self.encoding_maps_[col] = {val: i for i, val in enumerate(unique_vals)}
                self.explanation[col] = f"Ordinal Encoded (no target). {n_unique} categories."

        self._logger.info(
            f"[Encoder] Encoded {len(self.categorical_cols_)} categorical columns. "
            f"Methods used: {dict(pd.Series(list(self.encoding_methods_used_.values())).value_counts())}"
        )

        self.is_fitted_ = True
        self.feature_names_out_ = X.columns.tolist()  # columns stay the same, just values change
        return self

    def transform(self, X):
        """Apply learned encoding maps."""
        self._check_is_fitted()
        X = self._validate_input(X, context='transform')
        X_encoded = X.copy()

        for col in self.categorical_cols_:
            if col not in X_encoded.columns or col not in self.encoding_maps_:
                continue

            encoding_map = self.encoding_maps_[col]
            mapped_values = X_encoded[col].map(encoding_map)

            # For unseen categories: use global_mean (target encoding)
            # or 0 (frequency encoding) or -1 (ordinal)
            method = self.encoding_methods_used_.get(col, 'ordinal')
            if method == 'target':
                fill_val = self.global_mean_
            elif method == 'frequency':
                fill_val = 0.0  # unseen = zero frequency
            else:
                fill_val = -1

            n_unseen = mapped_values.isna().sum() - X_encoded[col].isna().sum()
            if n_unseen > 0:
                self._logger.debug(
                    f"[Encoder] {n_unseen} unseen categories in '{col}' imputed with {fill_val}."
                )

            X_encoded[col] = mapped_values.fillna(fill_val)

        return X_encoded

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
