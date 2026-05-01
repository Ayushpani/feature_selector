"""
LogTransformer — Automatic Skewness Correction

Highly skewed features degrade correlation and mutual information calculations
and can bias linear models. This transformer automatically detects skewed
numerical features and applies log1p transformation to normalize their
distribution.

Critical for: financial data, count data, heavy-tailed distributions.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.logger import get_logger


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Automatically detects and corrects skewed numerical features.

    Parameters
    ----------
    skewness_threshold : float, default=1.0
        Features with |skewness| > threshold will be transformed.
    method : str, default='log1p'
        Transformation method: 'log1p' (natural log of 1+x),
        'sqrt' (square root), 'boxcox' (Box-Cox, requires positive values).
    handle_negative : str, default='shift'
        How to handle negative values: 'shift' (add min+1 to make all positive),
        'abs' (use absolute value), 'skip' (skip columns with negatives).
    """
    def __init__(self, skewness_threshold=1.0, method='log1p', handle_negative='shift'):
        self.skewness_threshold = skewness_threshold
        self.method = method
        self.handle_negative = handle_negative

        self.skewed_cols_ = []
        self.skewness_values_ = {}
        self.shift_values_ = {}
        self.is_fitted_ = False
        self._logger = get_logger()

    def fit(self, X, y=None):
        """Detect skewed numerical columns."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

        self.skewed_cols_ = []
        self.skewness_values_ = {}
        self.shift_values_ = {}

        for col in numerical_cols:
            series = X[col].dropna()
            if len(series) < 10:  # Need sufficient data for skewness
                continue

            # Exclude binary columns (only 2 unique values)
            if series.nunique() <= 2:
                continue

            skewness = float(stats.skew(series, nan_policy='omit'))
            self.skewness_values_[col] = skewness

            if abs(skewness) > self.skewness_threshold:
                min_val = series.min()

                if min_val < 0:
                    if self.handle_negative == 'skip':
                        self._logger.debug(
                            f"[LogTransformer] Skipping '{col}' (skew={skewness:.2f}): "
                            f"contains negative values."
                        )
                        continue
                    elif self.handle_negative == 'shift':
                        self.shift_values_[col] = abs(min_val) + 1
                    elif self.handle_negative == 'abs':
                        self.shift_values_[col] = 0  # Will use abs()
                elif min_val == 0 and self.method == 'boxcox':
                    self.shift_values_[col] = 1  # Box-Cox needs strictly positive

                self.skewed_cols_.append(col)

        if self.skewed_cols_:
            self._logger.info(
                f"[LogTransformer] Detected {len(self.skewed_cols_)} skewed features "
                f"(|skew| > {self.skewness_threshold}): {self.skewed_cols_[:10]}"
                f"{'...' if len(self.skewed_cols_) > 10 else ''}"
            )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply skewness-correcting transformations."""
        if not self.is_fitted_:
            raise RuntimeError("[LogTransformer] Not fitted. Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        for col in self.skewed_cols_:
            if col not in X_out.columns:
                continue

            series = X_out[col].copy()

            # Handle negatives
            if col in self.shift_values_:
                shift = self.shift_values_[col]
                if self.handle_negative == 'abs':
                    series = series.abs()
                else:
                    series = series + shift

            # Apply transformation
            if self.method == 'log1p':
                X_out[col] = np.log1p(series.clip(lower=0))
            elif self.method == 'sqrt':
                X_out[col] = np.sqrt(series.clip(lower=0))
            elif self.method == 'boxcox':
                try:
                    positive_mask = series > 0
                    if positive_mask.sum() > 2:
                        transformed, _ = stats.boxcox(series[positive_mask])
                        X_out.loc[positive_mask, col] = transformed
                except (ValueError, RuntimeWarning):
                    # Fallback to log1p if Box-Cox fails
                    X_out[col] = np.log1p(series.clip(lower=0))

        return X_out

    def get_skewness_report(self):
        """Return a summary of skewness values and decisions."""
        return pd.DataFrame({
            'feature': list(self.skewness_values_.keys()),
            'skewness': list(self.skewness_values_.values()),
            'transformed': [c in self.skewed_cols_ for c in self.skewness_values_],
        }).sort_values('skewness', key=abs, ascending=False)
