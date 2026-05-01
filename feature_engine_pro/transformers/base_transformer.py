"""
Base Transformer — Hardened Abstract Foundation

All transformers inherit from this class. It provides:
- Input validation (DataFrame type, empty checks, duplicate columns)
- Fitted-state tracking with check_is_fitted()
- sklearn get_feature_names_out() compatibility
- Structured logging
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted as _sklearn_check_is_fitted

from feature_engine_pro.logger import get_logger


class _BaseTransformer(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract Base Class for all Feature Engine Pro transformers.

    Provides defensive input validation, fitted-state management,
    and sklearn pipeline compatibility.
    """
    def __init__(self):
        self.explanation = {}
        self.is_fitted_ = False
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self._logger = get_logger()

    def _validate_input(self, X, context='fit'):
        """
        Validate input data with clear, actionable error messages.

        Parameters
        ----------
        X : any
            The input to validate.
        context : str
            'fit' or 'transform', used in error messages.

        Returns
        -------
        pd.DataFrame
            The validated (and possibly converted) DataFrame.
        """
        if X is None:
            raise ValueError(
                f"[{self.__class__.__name__}] X cannot be None during {context}()."
            )

        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
                self._logger.debug(
                    f"[{self.__class__.__name__}] Converted input to DataFrame during {context}()."
                )
            except Exception as e:
                raise TypeError(
                    f"[{self.__class__.__name__}] Cannot convert input to DataFrame: {e}"
                )

        if X.empty:
            raise ValueError(
                f"[{self.__class__.__name__}] Received an empty DataFrame during {context}(). "
                f"Shape: {X.shape}."
            )

        # Check for duplicate column names
        dupes = X.columns[X.columns.duplicated()].tolist()
        if dupes:
            raise ValueError(
                f"[{self.__class__.__name__}] DataFrame has duplicate column names: {dupes}. "
                f"Please deduplicate before passing to the pipeline."
            )

        return X

    def _check_is_fitted(self):
        """Raise an error if the transformer has not been fitted."""
        if not self.is_fitted_:
            raise RuntimeError(
                f"[{self.__class__.__name__}] This transformer has not been fitted yet. "
                f"Call .fit() before .transform()."
            )

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the transformer. Subclasses must implement this."""
        pass

    @abstractmethod
    def transform(self, X):
        """Transform the data. Subclasses must implement this."""
        pass

    def get_feature_names_out(self, input_features=None):
        """
        Get the output feature names after transformation.

        Sklearn pipeline compatibility.
        """
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_)
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_)
        raise RuntimeError(
            f"[{self.__class__.__name__}] Feature names not available. "
            f"Fit the transformer first."
        )
