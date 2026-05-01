from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class _BaseTransformer(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract Base Class for all transformers.
    """
    def __init__(self):
        self.explanation = {}

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass
