"""
OutlierHandler — Robust Outlier Detection and Treatment

Outliers can catastrophically affect variance, correlation, mutual information,
and model training. This transformer detects and treats outliers using
training-data-learned bounds to prevent data leakage.

Detection Methods:
- IQR (Interquartile Range) — robust, works for skewed data
- Z-Score — best for normally distributed data
- MAD (Median Absolute Deviation) — robust alternative to Z-Score

Treatment Strategies:
- 'clip' (Winsorize) — cap values at bounds
- 'nan' — replace outliers with NaN for downstream imputation
- 'drop' — remove rows with outliers (use with caution)
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.logger import get_logger


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detects and treats outliers in numerical features.

    Parameters
    ----------
    method : str, default='iqr'
        Detection method: 'iqr', 'zscore', or 'mad'.
    treatment : str, default='clip'
        Treatment strategy: 'clip', 'nan', or 'drop'.
    iqr_factor : float, default=1.5
        Multiplier for IQR bounds. Use 3.0 for extreme outliers only.
    zscore_threshold : float, default=3.0
        Z-score threshold for outlier detection.
    mad_threshold : float, default=3.5
        Modified Z-score threshold (MAD-based).
    """
    def __init__(self, method='iqr', treatment='clip', iqr_factor=1.5,
                 zscore_threshold=3.0, mad_threshold=3.5):
        self.method = method
        self.treatment = treatment
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold
        self.mad_threshold = mad_threshold

        # Learned bounds
        self.bounds_ = {}
        self.numerical_cols_ = []
        self.is_fitted_ = False
        self.outlier_counts_ = {}
        self._logger = get_logger()

    def fit(self, X, y=None):
        """Learn outlier bounds from training data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        for col in self.numerical_cols_:
            series = X[col].dropna()
            if series.empty:
                continue

            if self.method == 'iqr':
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_factor * iqr
                upper = q3 + self.iqr_factor * iqr

            elif self.method == 'zscore':
                mean = series.mean()
                std = series.std()
                if std == 0:
                    lower, upper = mean, mean
                else:
                    lower = mean - self.zscore_threshold * std
                    upper = mean + self.zscore_threshold * std

            elif self.method == 'mad':
                median = series.median()
                mad = np.median(np.abs(series - median))
                if mad == 0:
                    lower, upper = median, median
                else:
                    # Modified Z-score
                    modified_z_limit = self.mad_threshold * 1.4826 * mad
                    lower = median - modified_z_limit
                    upper = median + modified_z_limit
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")

            self.bounds_[col] = {'lower': lower, 'upper': upper}

            # Count outliers for reporting
            n_outliers = ((series < lower) | (series > upper)).sum()
            self.outlier_counts_[col] = n_outliers

        total_outliers = sum(self.outlier_counts_.values())
        cols_with_outliers = sum(1 for v in self.outlier_counts_.values() if v > 0)
        self._logger.info(
            f"[OutlierHandler] Fit complete (method={self.method}). "
            f"Found {total_outliers} outlier values across {cols_with_outliers} columns."
        )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply outlier treatment using learned bounds."""
        if not self.is_fitted_:
            raise RuntimeError("[OutlierHandler] Not fitted. Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        if self.treatment == 'drop':
            mask = pd.Series(True, index=X_out.index)
            for col, bounds in self.bounds_.items():
                if col in X_out.columns:
                    col_mask = (X_out[col] >= bounds['lower']) & (X_out[col] <= bounds['upper'])
                    # Keep NaN rows (they are missing, not outliers)
                    col_mask = col_mask | X_out[col].isna()
                    mask = mask & col_mask
            n_dropped = (~mask).sum()
            if n_dropped > 0:
                self._logger.info(f"[OutlierHandler] Dropping {n_dropped} outlier rows.")
            X_out = X_out[mask].reset_index(drop=True)

        else:
            for col, bounds in self.bounds_.items():
                if col not in X_out.columns:
                    continue

                lower, upper = bounds['lower'], bounds['upper']

                if self.treatment == 'clip':
                    X_out[col] = X_out[col].clip(lower=lower, upper=upper)
                elif self.treatment == 'nan':
                    outlier_mask = (X_out[col] < lower) | (X_out[col] > upper)
                    X_out.loc[outlier_mask, col] = np.nan

        return X_out

    def get_outlier_summary(self):
        """Return a summary of outlier counts per feature."""
        return pd.DataFrame({
            'feature': list(self.outlier_counts_.keys()),
            'outlier_count': list(self.outlier_counts_.values()),
            'lower_bound': [self.bounds_.get(c, {}).get('lower', None) for c in self.outlier_counts_],
            'upper_bound': [self.bounds_.get(c, {}).get('upper', None) for c in self.outlier_counts_],
        })
