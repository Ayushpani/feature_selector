"""
PolynomialFeatureGenerator — Smart Interaction Feature Engineering

Generates degree-2 interaction features for the most promising feature pairs.
Uses pre-screening (correlation with target) to avoid combinatorial explosion.

Critical for: Capturing multiplicative relationships invisible to linear methods.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.logger import get_logger


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates interaction features for the top-K most promising features.

    Parameters
    ----------
    max_interactions : int, default=50
        Maximum number of interaction features to create.
    top_k_features : int, default=15
        Number of top features (by target correlation) to consider for interactions.
    include_squares : bool, default=True
        Include squared features (x^2) in addition to pairwise interactions (x*y).
    min_target_correlation : float, default=0.05
        Minimum absolute correlation with target for a feature to be considered.
    """
    def __init__(self, max_interactions=50, top_k_features=15,
                 include_squares=True, min_target_correlation=0.05):
        self.max_interactions = max_interactions
        self.top_k_features = top_k_features
        self.include_squares = include_squares
        self.min_target_correlation = min_target_correlation

        self.selected_features_ = []
        self.interaction_pairs_ = []
        self.square_features_ = []
        self.is_fitted_ = False
        self._logger = get_logger()

    def fit(self, X, y=None):
        """Pre-screen features and select interaction pairs."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

        if y is None or len(numerical_cols) < 2:
            self._logger.debug(
                "[PolynomialFeatures] Skipping: no target or < 2 numerical columns."
            )
            self.is_fitted_ = True
            return self

        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)

        # Pre-screen: rank features by absolute correlation with target
        target_corrs = {}
        for col in numerical_cols:
            try:
                corr = abs(X[col].corr(y))
                if not np.isnan(corr) and corr >= self.min_target_correlation:
                    target_corrs[col] = corr
            except Exception:
                pass

        # Select top K features
        sorted_features = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
        self.selected_features_ = [f[0] for f in sorted_features[:self.top_k_features]]

        if len(self.selected_features_) < 2:
            self._logger.debug(
                "[PolynomialFeatures] Fewer than 2 features passed pre-screening. Skipping."
            )
            self.is_fitted_ = True
            return self

        # Generate interaction pairs
        self.interaction_pairs_ = list(combinations(self.selected_features_, 2))

        # Generate squared features
        if self.include_squares:
            self.square_features_ = self.selected_features_[:]

        # Cap total new features
        total = len(self.interaction_pairs_) + len(self.square_features_)
        if total > self.max_interactions:
            # Prioritize interactions over squares
            max_pairs = self.max_interactions - len(self.square_features_)
            if max_pairs < 0:
                self.square_features_ = self.square_features_[:self.max_interactions]
                self.interaction_pairs_ = []
            else:
                self.interaction_pairs_ = self.interaction_pairs_[:max_pairs]

        self._logger.info(
            f"[PolynomialFeatures] Will generate {len(self.interaction_pairs_)} interactions "
            f"and {len(self.square_features_)} squared features "
            f"from top {len(self.selected_features_)} features."
        )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Generate interaction features."""
        if not self.is_fitted_:
            raise RuntimeError("[PolynomialFeatures] Not fitted. Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        # Interaction features (x * y)
        for col_a, col_b in self.interaction_pairs_:
            if col_a in X_out.columns and col_b in X_out.columns:
                name = f"{col_a}__x__{col_b}"
                X_out[name] = X_out[col_a] * X_out[col_b]

        # Squared features (x^2)
        for col in self.square_features_:
            if col in X_out.columns:
                name = f"{col}__squared"
                X_out[name] = X_out[col] ** 2

        return X_out
