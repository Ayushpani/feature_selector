"""
CorrelationSelector — Advanced Multicollinearity Resolution

Improvements over v1:
- Spearman support and 'auto' method selection
- Constant-column crash protection (zero std → NaN correlation handling)
- Hierarchical clustering for intelligent grouped deletion
- Optional VIF (Variance Inflation Factor) secondary filter
- Robust reporter integration
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from feature_engine_pro.selectors.base_selector import _BaseSelector
from feature_engine_pro.logger import get_logger


class CorrelationSelector(_BaseSelector):
    """
    Removes multicollinear features by keeping the one most correlated with the target.

    Parameters
    ----------
    threshold : float, default=0.85
        Correlation threshold above which features are considered redundant.
    target_column : str, optional
    problem_type : str, default='classification'
    method : str, default='auto'
        Correlation method: 'pearson', 'spearman', or 'auto'.
        'auto' uses Spearman if any feature has |skewness| > 2, else Pearson.
    use_clustering : bool, default=True
        If True, use hierarchical clustering to group correlated features
        and keep the best representative from each cluster (more intelligent
        than greedy pairwise deletion).
    vif_threshold : float or None, default=None
        If set, apply VIF-based secondary filter. Features with VIF > threshold
        (typically 5-10) indicate severe multicollinearity.
    """
    def __init__(self, threshold=0.85, target_column=None, problem_type='classification',
                 method='auto', use_clustering=True, vif_threshold=None):
        super().__init__(target_column, problem_type)
        self.threshold = threshold
        self.method = method
        self.use_clustering = use_clustering
        self.vif_threshold = vif_threshold
        self.correlation_matrix_ = None
        self.actual_method_ = None

    def _compute_vif(self, X_numerical):
        """Compute Variance Inflation Factor for each feature."""
        from numpy.linalg import LinAlgError
        vif_values = {}
        cols = X_numerical.columns.tolist()
        X_np = X_numerical.values

        for i, col in enumerate(cols):
            try:
                y_i = X_np[:, i]
                X_others = np.delete(X_np, i, axis=1)
                if X_others.shape[1] == 0:
                    vif_values[col] = 1.0
                    continue

                # Add intercept
                X_others_c = np.column_stack([np.ones(X_others.shape[0]), X_others])
                try:
                    beta = np.linalg.lstsq(X_others_c, y_i, rcond=None)[0]
                    y_pred = X_others_c @ beta
                    ss_res = np.sum((y_i - y_pred) ** 2)
                    ss_tot = np.sum((y_i - y_i.mean()) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                    vif_values[col] = vif
                except LinAlgError:
                    vif_values[col] = float('inf')
            except Exception:
                vif_values[col] = float('inf')

        return vif_values

    def fit(self, X, y=None):
        """Fit: identify and resolve multicollinearity."""
        X = self._validate_input(X, context='fit')
        self.original_feature_names = X.columns.tolist()

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        non_numerical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        if len(numerical_cols) < 2:
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            for col in self.original_feature_names:
                self._log_to_reporter(col, 'kept', 'Fewer than 2 numerical columns.', 'CorrelationSelector')
            self.is_fitted_ = True
            return self

        # --- Determine correlation method ---
        if self.method == 'auto':
            from scipy.stats import skew
            max_skew = max(abs(skew(X[col].dropna())) for col in numerical_cols if X[col].dropna().shape[0] > 2)
            self.actual_method_ = 'spearman' if max_skew > 2 else 'pearson'
            self._logger.debug(
                f"[CorrelationSelector] Auto-selected method: {self.actual_method_} "
                f"(max skewness: {max_skew:.2f})."
            )
        else:
            self.actual_method_ = self.method

        # --- Compute correlation matrix (handling zero-std columns) ---
        # Drop constant columns first to avoid NaN correlations
        non_constant_cols = [
            col for col in numerical_cols
            if X[col].std() > 0 and X[col].nunique() > 1
        ]
        constant_cols = [col for col in numerical_cols if col not in non_constant_cols]

        if constant_cols:
            self._logger.debug(
                f"[CorrelationSelector] {len(constant_cols)} constant columns excluded from correlation."
            )

        if len(non_constant_cols) < 2:
            self.selected_features_ = X.columns.tolist()
            self.dropped_features_ = []
            self.is_fitted_ = True
            return self

        corr_matrix = X[non_constant_cols].corr(method=self.actual_method_).abs()
        self.correlation_matrix_ = corr_matrix

        # Fill NaN with 0 (can happen with constant-ish columns)
        corr_matrix = corr_matrix.fillna(0)

        # --- Target correlation for tie-breaking ---
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            elif not isinstance(y, pd.Series):
                y = pd.Series(y, index=X.index)
            y_aligned = y.copy()
            y_aligned.index = X.index
            target_corr = X[non_constant_cols].apply(
                lambda col: abs(col.corr(y_aligned, method=self.actual_method_))
            ).fillna(0)
        else:
            target_corr = pd.Series(1.0, index=non_constant_cols)

        to_drop = set()

        if self.use_clustering and len(non_constant_cols) >= 3:
            # --- Hierarchical Clustering Approach ---
            # Convert correlation to distance
            distance_matrix = 1 - corr_matrix.values
            np.fill_diagonal(distance_matrix, 0)
            distance_matrix = np.clip(distance_matrix, 0, 2)

            # Ensure symmetry
            distance_matrix = (distance_matrix + distance_matrix.T) / 2

            try:
                condensed = squareform(distance_matrix, checks=False)
                Z = linkage(condensed, method='complete')

                # Cut threshold: features with correlation > self.threshold are in same cluster
                cut_distance = 1 - self.threshold
                clusters = fcluster(Z, t=cut_distance, criterion='distance')

                cluster_map = {}
                for col, cluster_id in zip(non_constant_cols, clusters):
                    cluster_map.setdefault(cluster_id, []).append(col)

                # From each cluster, keep the feature most correlated with target
                for cluster_id, members in cluster_map.items():
                    if len(members) == 1:
                        self._log_to_reporter(
                            members[0], 'kept',
                            f'No collinearity cluster (unique group). '
                            f'Method: {self.actual_method_}.',
                            'CorrelationSelector'
                        )
                        continue

                    # Keep the one with highest target correlation
                    best = max(members, key=lambda c: target_corr.get(c, 0))
                    for col in members:
                        if col != best:
                            to_drop.add(col)
                            best_corr = corr_matrix.loc[col, best] if col in corr_matrix.index and best in corr_matrix.columns else 0
                            self._log_to_reporter(
                                col, 'dropped',
                                f'Clustered with {len(members)} features (corr ≈ {best_corr:.2f}). '
                                f'Kept {best} (higher target corr: {target_corr.get(best, 0):.3f} vs {target_corr.get(col, 0):.3f}).',
                                'CorrelationSelector'
                            )
                        else:
                            self._log_to_reporter(
                                col, 'kept',
                                f'Best representative in cluster of {len(members)} features. '
                                f'Target correlation: {target_corr.get(col, 0):.3f}.',
                                'CorrelationSelector'
                            )
            except Exception as e:
                self._logger.warning(
                    f"[CorrelationSelector] Clustering failed ({e}), falling back to greedy."
                )
                to_drop = self._greedy_drop(corr_matrix, target_corr, non_constant_cols)
        else:
            # --- Greedy Pairwise Approach (fallback or < 3 features) ---
            to_drop = self._greedy_drop(corr_matrix, target_corr, non_constant_cols)

        # --- Optional VIF secondary filter ---
        if self.vif_threshold is not None:
            remaining_numerical = [c for c in non_constant_cols if c not in to_drop]
            if len(remaining_numerical) >= 2:
                vif_values = self._compute_vif(X[remaining_numerical].dropna())
                for col, vif in vif_values.items():
                    if vif > self.vif_threshold and col not in to_drop:
                        to_drop.add(col)
                        self._log_to_reporter(
                            col, 'dropped',
                            f'VIF = {vif:.1f} exceeds threshold {self.vif_threshold}.',
                            'CorrelationSelector'
                        )

        # Non-numerical columns always pass
        for col in non_numerical_cols:
            self._log_to_reporter(col, 'kept', 'Non-numerical, skipped.', 'CorrelationSelector')

        self.selected_features_ = [c for c in self.original_feature_names if c not in to_drop]
        self.dropped_features_ = list(to_drop)

        self._logger.info(
            f"[CorrelationSelector] Kept {len(self.selected_features_)}, "
            f"dropped {len(self.dropped_features_)} features "
            f"(method={self.actual_method_}, threshold={self.threshold}"
            f"{', VIF=' + str(self.vif_threshold) if self.vif_threshold else ''})."
        )

        self.is_fitted_ = True
        return self

    def _greedy_drop(self, corr_matrix, target_corr, cols):
        """Greedy pairwise deletion — fallback method."""
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = set()

        for col in upper_tri.columns:
            high_corr_cols = upper_tri[col][upper_tri[col] > self.threshold].index.tolist()
            for correlated_col in high_corr_cols:
                if col not in to_drop and correlated_col not in to_drop:
                    if target_corr.get(col, 0) > target_corr.get(correlated_col, 0):
                        drop_feature, keep_feature = correlated_col, col
                    else:
                        drop_feature, keep_feature = col, correlated_col

                    to_drop.add(drop_feature)
                    corr_val = upper_tri.loc[correlated_col, col] if correlated_col in upper_tri.index else 0
                    self._log_to_reporter(
                        drop_feature, 'dropped',
                        f'Correlated {corr_val:.2f} with {keep_feature}. '
                        f'Kept {keep_feature} (higher target correlation).',
                        'CorrelationSelector'
                    )

        for c in cols:
            if c not in to_drop:
                self._log_to_reporter(
                    c, 'kept',
                    f'No collinearity above {self.threshold}.',
                    'CorrelationSelector'
                )

        return to_drop
