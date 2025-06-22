import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import logging

from feature_engine_pro.selectors.base_selector import _BaseSelector

class VarianceThresholdSelector(_BaseSelector):
    """
    A filter method that removes features with variance below a certain threshold.
    Features with zero variance (i.e., constant features) are always removed if threshold is 0.0.
    """
    def __init__(self, threshold=0.0, target_column=None, problem_type='classification'):
        """
        Initializes the VarianceThresholdSelector.

        :param threshold: float, default=0.0
            Features with a training-set variance lower than this threshold will be removed.
            The default of 0.0 means only features with zero variance (constant features) are removed.
        :param target_column: str, optional
            The name of the target column. Not directly used by VarianceThreshold,
            but kept for consistency with other selectors.
        :param problem_type: str, optional
            The type of machine learning problem ('classification' or 'regression').
            Not directly used by VarianceThreshold, but kept for consistency.
        """
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self._selector = VarianceThreshold(threshold=self.threshold)

    def fit(self, X, y=None):
        """
        Fits the VarianceThreshold selector to the data.

        This method identifies which numerical features have a variance below
        the specified threshold and marks them for removal. Non-numerical features
        are always retained by this selector.

        :param X: pandas.DataFrame
            The input DataFrame containing features.
        :param y: pandas.Series or numpy.ndarray, optional
            The target variable. Not used by VarianceThreshold, but included
            for compatibility with the _BaseSelector interface.
        :return: self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        self.original_feature_names = X.columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        if not numerical_cols:
            self.selected_features = X.columns.tolist() # Retain all if no numerical
            self.explanation = {
                'method_used': 'Variance Threshold',
                'threshold': self.threshold,
                'original_feature_count': X.shape[1],
                'selected_feature_count': len(self.selected_features),
                'features_info': {col: 'Not numerical, retained.' for col in X.columns},
                'removed_features': [],
                'selected_features': self.selected_features,
                'summary': 'No numerical features found for variance thresholding. All original features retained.'
            }
            logging.info("No numerical columns found for VarianceThresholdSelector. All features retained.")
            return self

        # Fit only on numerical columns as variance threshold is for numerical data
        try:
            self._selector.fit(X[numerical_cols])
        except Exception as e:
            logging.error(f"Error fitting VarianceThresholdSelector: {e}. Retaining all features.")
            self.selected_features = X.columns.tolist()
            self.explanation = {
                'method_used': 'Variance Threshold',
                'threshold': self.threshold,
                'original_feature_count': X.shape[1],
                'selected_feature_count': len(self.selected_features),
                'features_info': {col: f"Error during fit: {e}. Retained." for col in X.columns},
                'removed_features': [],
                'selected_features': self.selected_features,
                'summary': f"VarianceThresholdSelector failed to fit. All features retained. Error: {e}"
            }
            return self

        # Get selected numerical features
        selected_numerical_mask = self._selector.get_support()
        selected_numerical_features = [col for idx, col in enumerate(numerical_cols) if selected_numerical_mask[idx]]

        # Combine with non-numerical features which are always kept by this selector
        non_numerical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        self.selected_features = selected_numerical_features + non_numerical_features

        # Prepare explanation
        removed_features = [col for col in numerical_cols if col not in selected_numerical_features]
        self.explanation = {
            'method_used': 'Variance Threshold',
            'threshold': self.threshold,
            'original_feature_count': X.shape[1],
            'selected_feature_count': len(self.selected_features),
            'feature_scores': {col: X[col].var() for col in numerical_cols}, # Store actual variances
            'features_info': {},
            'removed_features': removed_features,
            'selected_features': self.selected_features,
            'summary': f"Features with variance below {self.threshold} were removed ({len(removed_features)} features). Non-numerical features were retained."
        }
        for col in X.columns:
            if col in selected_numerical_features:
                self.explanation['features_info'][col] = f"Selected (Variance: {X[col].var():.4f} >= {self.threshold})"
            elif col in numerical_cols and col in removed_features:
                 self.explanation['features_info'][col] = f"Removed (Variance: {X[col].var():.4f} < {self.threshold})"
            else: # Non-numerical features
                self.explanation['features_info'][col] = "Not numerical, retained."

        logging.info(f"VarianceThresholdSelector fitted. Selected {len(self.selected_features)} out of {X.shape[1]} features.")
        return self

