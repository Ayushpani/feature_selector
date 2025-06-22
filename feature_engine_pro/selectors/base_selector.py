from abc import ABC, abstractmethod
import pandas as pd

class _BaseSelector(ABC):
    """
    Abstract Base Class for all feature selection methods.
    Defines the interface that all selectors must implement.
    """
    def __init__(self, target_column=None, problem_type='classification'):
        self.target_column = target_column
        self.problem_type = problem_type # 'classification' or 'regression'
        self.selected_features = None
        self.explanation = {}
        self.original_feature_names = None

    @abstractmethod
    def fit(self, X, y=None):
        """Fits the selector to the data."""
        pass

    def transform(self, X):
        """Transforms the data by selecting features."""
        if self.selected_features is None:
            raise RuntimeError("Selector not fitted. Call .fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        
        # Ensure all selected features exist in the input DataFrame
        # If any selected feature is missing in the input X, filter them out to prevent KeyError
        valid_selected_features = [f for f in self.selected_features if f in X.columns]
        if len(valid_selected_features) < len(self.selected_features):
            missing_features = set(self.selected_features) - set(valid_selected_features)
            # Log this warning for debugging, but proceed with available features
            # A more robust system might raise an error or ask for user input.
            # For this context, we will simply filter and warn.
            import logging
            logging.warning(f"Transforming with missing features that were selected during fit: {missing_features}. These will be excluded from the transformed DataFrame.")
            self.selected_features = valid_selected_features # Update selected features to reflect what's actually present

        return X[self.selected_features].copy()

    def fit_transform(self, X, y=None):
        """Fits and transforms the data."""
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self):
        """Returns the list of selected feature names."""
        return self.selected_features

    def get_explanation(self):
        """Returns a dictionary explaining the selection process."""
        return self.explanation
