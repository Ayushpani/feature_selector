import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeExtractor(BaseEstimator, TransformerMixin):
    """
    Automatically detects datetime columns and expands them into
    useful numerical features (year, month, day, dayofweek, is_weekend).
    It then drops the original datetime columns.
    """
    def __init__(self, drop_original=True):
        self.drop_original = drop_original
        self.datetime_cols_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Detect datetime columns or columns that look like datetime
        self.datetime_cols_ = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                 self.datetime_cols_.append(col)
            elif X[col].dtype == 'object':
                 try:
                     # Try parsing the first valid row to see if it's a date
                     first_valid = X[col].dropna().iloc[0]
                     pd.to_datetime(first_valid)
                     self.datetime_cols_.append(col)
                 except (ValueError, TypeError, IndexError):
                     pass

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        for col in self.datetime_cols_:
            if col in X_out.columns:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(X_out[col]):
                     X_out[col] = pd.to_datetime(X_out[col], errors='coerce')

                X_out[f'{col}_year'] = X_out[col].dt.year.fillna(-1)
                X_out[f'{col}_month'] = X_out[col].dt.month.fillna(-1)
                X_out[f'{col}_day'] = X_out[col].dt.day.fillna(-1)
                X_out[f'{col}_dayofweek'] = X_out[col].dt.dayofweek.fillna(-1)
                X_out[f'{col}_is_weekend'] = (X_out[f'{col}_dayofweek'] >= 5).astype(int)

                if self.drop_original:
                    X_out.drop(columns=[col], inplace=True)

        return X_out
