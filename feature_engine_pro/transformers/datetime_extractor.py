"""
DatetimeExtractor — Advanced Temporal Feature Engineering

Detects datetime columns and expands them into rich numerical features.

Improvements over v1:
- Cyclic encoding (sin/cos) for month, dayofweek, hour to preserve cyclical relationships
- Parse threshold to avoid false-positive datetime detection
- Timezone normalization to UTC
- Additional features: hour, minute, quarter, is_month_start, is_month_end, days_since_epoch
- Robust handling of partial/failed parsing
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.logger import get_logger


class DatetimeExtractor(BaseEstimator, TransformerMixin):
    """
    Automatically detects datetime columns and expands them into
    useful numerical features with cyclic encoding for periodic components.

    Parameters
    ----------
    drop_original : bool, default=True
        Drop the original datetime column after expansion.
    parse_threshold : float, default=0.8
        Minimum fraction of successfully parsed values to treat a column as datetime.
        Prevents false positives on mixed-content object columns.
    cyclic_encode : bool, default=True
        Use sin/cos encoding for cyclical features (month, dayofweek, hour).
    extract_time : bool, default=True
        Extract hour/minute if the datetime has time components.
    """
    def __init__(self, drop_original=True, parse_threshold=0.8,
                 cyclic_encode=True, extract_time=True):
        self.drop_original = drop_original
        self.parse_threshold = parse_threshold
        self.cyclic_encode = cyclic_encode
        self.extract_time = extract_time
        self.datetime_cols_ = []
        self.has_time_component_ = {}
        self.is_fitted_ = False
        self._logger = get_logger()

    def fit(self, X, y=None):
        """Detect datetime columns."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.datetime_cols_ = []
        self.has_time_component_ = {}

        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.datetime_cols_.append(col)
                # Check if time component is non-trivial
                self.has_time_component_[col] = (
                    X[col].dropna().dt.hour.nunique() > 1 or
                    X[col].dropna().dt.minute.nunique() > 1
                )
            elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
                # Attempt parsing and measure success rate
                non_null = X[col].dropna()
                if len(non_null) == 0:
                    continue
                try:
                    parsed = pd.to_datetime(non_null, errors='coerce')
                    success_rate = parsed.notna().mean()
                    if success_rate >= self.parse_threshold:
                        self.datetime_cols_.append(col)
                        self.has_time_component_[col] = (
                            parsed.dropna().dt.hour.nunique() > 1 or
                            parsed.dropna().dt.minute.nunique() > 1
                        )
                        self._logger.debug(
                            f"[DatetimeExtractor] Column '{col}' detected as datetime "
                            f"({success_rate:.0%} parse success)."
                        )
                    elif success_rate > 0.3:
                        self._logger.debug(
                            f"[DatetimeExtractor] Column '{col}' partially parses as datetime "
                            f"({success_rate:.0%}) but below threshold {self.parse_threshold:.0%}. Skipping."
                        )
                except (ValueError, TypeError, OverflowError):
                    pass

        if self.datetime_cols_:
            self._logger.info(
                f"[DatetimeExtractor] Detected {len(self.datetime_cols_)} datetime columns: "
                f"{self.datetime_cols_}"
            )

        self.is_fitted_ = True
        return self

    def _cyclic_encode(self, series, period):
        """Encode a cyclical feature as sin/cos pair."""
        sin_vals = np.sin(2 * np.pi * series / period)
        cos_vals = np.cos(2 * np.pi * series / period)
        return sin_vals, cos_vals

    def transform(self, X):
        """Expand datetime columns into numerical features."""
        if not self.is_fitted_:
            raise RuntimeError("[DatetimeExtractor] Not fitted. Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        for col in self.datetime_cols_:
            if col not in X_out.columns:
                continue

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_out[col]):
                X_out[col] = pd.to_datetime(X_out[col], errors='coerce', utc=True)
                # Remove timezone info after normalization for clean numerical extraction
                if hasattr(X_out[col].dt, 'tz') and X_out[col].dt.tz is not None:
                    X_out[col] = X_out[col].dt.tz_localize(None)
            else:
                # Normalize timezone if present
                if hasattr(X_out[col].dt, 'tz') and X_out[col].dt.tz is not None:
                    X_out[col] = X_out[col].dt.tz_convert('UTC').dt.tz_localize(None)

            dt = X_out[col].dt

            # Core temporal features
            X_out[f'{col}_year'] = dt.year.fillna(-1).astype(int)

            # Cyclic encoding for periodic features
            month = dt.month.fillna(0)
            dow = dt.dayofweek.fillna(0)

            if self.cyclic_encode:
                sin_m, cos_m = self._cyclic_encode(month, 12)
                X_out[f'{col}_month_sin'] = sin_m
                X_out[f'{col}_month_cos'] = cos_m

                sin_d, cos_d = self._cyclic_encode(dow, 7)
                X_out[f'{col}_dow_sin'] = sin_d
                X_out[f'{col}_dow_cos'] = cos_d
            else:
                X_out[f'{col}_month'] = month.astype(int)
                X_out[f'{col}_dayofweek'] = dow.astype(int)

            X_out[f'{col}_day'] = dt.day.fillna(-1).astype(int)
            X_out[f'{col}_quarter'] = dt.quarter.fillna(-1).astype(int)
            X_out[f'{col}_is_weekend'] = (dow >= 5).astype(int)
            X_out[f'{col}_is_month_start'] = dt.is_month_start.fillna(False).astype(int)
            X_out[f'{col}_is_month_end'] = dt.is_month_end.fillna(False).astype(int)

            # Days since epoch — a single continuous feature for temporal ordering
            epoch = pd.Timestamp('1970-01-01')
            X_out[f'{col}_days_since_epoch'] = (X_out[col] - epoch).dt.total_seconds().fillna(0) / 86400

            # Time components (hour, minute) if present
            if self.extract_time and self.has_time_component_.get(col, False):
                hour = dt.hour.fillna(0)
                minute = dt.minute.fillna(0)

                if self.cyclic_encode:
                    sin_h, cos_h = self._cyclic_encode(hour, 24)
                    X_out[f'{col}_hour_sin'] = sin_h
                    X_out[f'{col}_hour_cos'] = cos_h
                else:
                    X_out[f'{col}_hour'] = hour.astype(int)
                    X_out[f'{col}_minute'] = minute.astype(int)

            if self.drop_original:
                X_out.drop(columns=[col], inplace=True)

        return X_out
