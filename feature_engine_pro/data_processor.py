import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Handles initial data validation, imputation of missing values,
    and segregation of numerical vs categorical columns to prepare
    the data for the mathematical selection pipeline.
    """
    def __init__(self, imputation_strategy='mean'):
        self.imputation_strategy = imputation_strategy
        self.numerical_cols = []
        self.categorical_cols = []
        self.imputation_values_ = {}

    def fit(self, X, y=None):
        """
        Learns imputation statistics from the training data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Learn imputation values
        for col in self.numerical_cols:
            if self.imputation_strategy == 'mean':
                self.imputation_values_[col] = X[col].mean()
            elif self.imputation_strategy == 'median':
                self.imputation_values_[col] = X[col].median()
            else:
                self.imputation_values_[col] = 0

        for col in self.categorical_cols:
             self.imputation_values_[col] = 'Missing_Category'

        return self

    def transform(self, X):
        """
        Applies learned imputation to the data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_processed = X.copy()

        # Handle missing values securely using learned statistics
        for col in self.numerical_cols:
            if col in X_processed.columns:
                 X_processed[col] = X_processed[col].fillna(self.imputation_values_.get(col, 0))

        for col in self.categorical_cols:
            if col in X_processed.columns:
                 X_processed[col] = X_processed[col].fillna(self.imputation_values_.get(col, 'Missing_Category'))

        return X_processed
