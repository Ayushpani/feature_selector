import pandas as pd
import numpy as np
from feature_engine_pro.transformers.base_transformer import _BaseTransformer

class AutoCategoricalEncoder(_BaseTransformer):
    """
    Automatically converts categorical (text) columns into mathematical representations.
    For classification/regression, it defaults to Target Encoding (or Ordinal if target is missing)
    so that downstream correlation and mathematical filters can process them.
    """
    def __init__(self, target_column=None, problem_type='classification'):
        super().__init__()
        self.target_column = target_column
        self.problem_type = problem_type
        self.encoding_maps = {}
        self.categorical_cols = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in self.categorical_cols:
            if y is not None:
                # Target Encoding
                temp_df = pd.DataFrame({col: X[col], 'target': y})
                mean_target = temp_df.groupby(col)['target'].mean()
                self.encoding_maps[col] = mean_target.to_dict()
                self.explanation[col] = f"Target Encoded based on '{self.problem_type}' relationship."
            else:
                # Fallback to Ordinal Encoding if no target
                unique_vals = X[col].unique()
                self.encoding_maps[col] = {val: i for i, val in enumerate(unique_vals)}
                self.explanation[col] = "Ordinal Encoded (No target provided)."
        return self

    def transform(self, X):
        X_encoded = X.copy()
        if not isinstance(X_encoded, pd.DataFrame):
            X_encoded = pd.DataFrame(X_encoded)

        for col in self.categorical_cols:
            if col in X_encoded.columns and col in self.encoding_maps:
                # Map values, fill unknown categories with median/mean of the encoded values
                mapped_values = X_encoded[col].map(self.encoding_maps[col])
                fill_val = np.nanmean(list(self.encoding_maps[col].values())) if self.encoding_maps[col] else 0
                X_encoded[col] = mapped_values.fillna(fill_val)

        return X_encoded

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
