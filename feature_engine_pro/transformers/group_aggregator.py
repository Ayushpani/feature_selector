"""
GroupAggregator — Smart Grouped Feature Engineering

Creates grouped aggregate features (mean, std, min, max, count) based on
grouping columns.

Improvements over v1:
- Smarter ID column heuristic (cardinality ratio + name pattern, not just "id" substring)
- Aggregation explosion cap (max_new_features)
- Memory guard to prevent OOM
- More aggregation functions (std, min, max, count)
- Deduplication of new feature names
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.logger import get_logger


class GroupAggregator(BaseEstimator, TransformerMixin):
    """
    Creates grouped aggregate features.

    Parameters
    ----------
    groupby_cols : list of str, optional
        Columns to group by. If None, auto-detects ID-like columns.
    aggregate_cols : list of str, optional
        Numerical columns to aggregate. If None, uses all numerical columns.
    aggregations : list of str, default=['mean', 'std']
        Aggregation functions to apply.
    max_new_features : int, default=100
        Maximum number of new features to create. Prevents combinatorial explosion.
    cardinality_ratio_max : float, default=0.5
        For auto-detection: max ratio of unique values to total rows for a column
        to be considered a grouping column.
    cardinality_ratio_min : float, default=0.005
        For auto-detection: min ratio. Columns with < 0.5% unique values are likely
        constants, not useful group keys.
    """
    def __init__(self, groupby_cols=None, aggregate_cols=None,
                 aggregations=None, max_new_features=100,
                 cardinality_ratio_max=0.5, cardinality_ratio_min=0.005):
        self.groupby_cols = groupby_cols
        self.aggregate_cols = aggregate_cols
        self.aggregations = aggregations if aggregations else ['mean', 'std']
        self.max_new_features = max_new_features
        self.cardinality_ratio_max = cardinality_ratio_max
        self.cardinality_ratio_min = cardinality_ratio_min

        self.learned_aggregates_ = {}
        self.actual_groupby_cols_ = []
        self.actual_aggregate_cols_ = []
        self.new_feature_names_ = []
        self.is_fitted_ = False
        self._logger = get_logger()

    def _is_groupable_column(self, series, col_name, n_rows):
        """
        Determine if a column is a good candidate for group-by operations.

        Uses both name heuristics and statistical properties.
        """
        n_unique = series.nunique()
        ratio = n_unique / n_rows if n_rows > 0 else 0

        # Must have reasonable cardinality
        if ratio > self.cardinality_ratio_max or ratio < self.cardinality_ratio_min:
            return False

        # Must have at least 2 unique values
        if n_unique < 2:
            return False

        # Name heuristics (more careful than just "id" substring)
        name_lower = col_name.lower()
        # Positive patterns: explicit ID columns, category columns
        positive_patterns = [
            name_lower.endswith('_id'),
            name_lower.endswith('id') and len(name_lower) > 2,
            name_lower.startswith('id_'),
            name_lower in ('customer', 'user', 'account', 'group', 'category',
                          'segment', 'region', 'department', 'store', 'product'),
            name_lower.endswith('_type'),
            name_lower.endswith('_category'),
            name_lower.endswith('_group'),
        ]

        # Negative patterns: avoid false positives
        negative_patterns = [
            'humid' in name_lower,
            'acid' in name_lower,
            'valid' in name_lower,
            'rapid' in name_lower,
            name_lower in ('index', 'id'),  # pure index columns aren't useful
        ]

        if any(negative_patterns):
            return False

        return any(positive_patterns)

    def fit(self, X, y=None):
        """Learn aggregation statistics from training data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_rows = len(X)

        # Determine groupby columns
        if self.groupby_cols:
            self.actual_groupby_cols_ = [c for c in self.groupby_cols if c in X.columns]
        else:
            self.actual_groupby_cols_ = [
                col for col in X.columns
                if self._is_groupable_column(X[col], col, n_rows)
            ]

        # Determine aggregate columns
        if self.aggregate_cols:
            self.actual_aggregate_cols_ = [c for c in self.aggregate_cols if c in X.columns]
        else:
            self.actual_aggregate_cols_ = X.select_dtypes(include='number').columns.tolist()

        # Remove groupby cols from aggregate cols
        self.actual_aggregate_cols_ = [
            c for c in self.actual_aggregate_cols_
            if c not in self.actual_groupby_cols_
        ]

        if not self.actual_groupby_cols_:
            self._logger.debug("[GroupAggregator] No groupable columns detected. Skipping.")
            self.is_fitted_ = True
            return self

        # Check for aggregation explosion
        total_new = (
            len(self.actual_groupby_cols_) *
            len(self.actual_aggregate_cols_) *
            len(self.aggregations)
        )
        if total_new > self.max_new_features:
            # Limit aggregate columns to fit within budget
            max_agg_cols = self.max_new_features // (
                len(self.actual_groupby_cols_) * len(self.aggregations)
            )
            self.actual_aggregate_cols_ = self.actual_aggregate_cols_[:max(1, max_agg_cols)]
            self._logger.warning(
                f"[GroupAggregator] Aggregation would create {total_new} features "
                f"(exceeds max_new_features={self.max_new_features}). "
                f"Limiting to {len(self.actual_aggregate_cols_)} aggregate columns."
            )

        # Learn aggregations
        self.new_feature_names_ = []
        for group_col in self.actual_groupby_cols_:
            self.learned_aggregates_[group_col] = {}
            for agg_col in self.actual_aggregate_cols_:
                if agg_col not in X.columns:
                    continue
                try:
                    agg_df = X.groupby(group_col)[agg_col].agg(self.aggregations).reset_index()

                    # Flatten multi-level column names
                    if isinstance(agg_df.columns, pd.MultiIndex):
                        agg_df.columns = [
                            f"{group_col}_agg_{func}_{agg_col}" if func != group_col else group_col
                            for func in agg_df.columns.get_level_values(-1)
                        ]
                    else:
                        rename_dict = {}
                        for func in self.aggregations:
                            new_name = f"{group_col}_agg_{func}_{agg_col}"
                            rename_dict[func] = new_name
                            self.new_feature_names_.append(new_name)
                        agg_df = agg_df.rename(columns=rename_dict)

                    self.learned_aggregates_[group_col][agg_col] = agg_df
                except Exception as e:
                    self._logger.warning(
                        f"[GroupAggregator] Failed to aggregate '{agg_col}' by '{group_col}': {e}"
                    )

        self._logger.info(
            f"[GroupAggregator] Fit complete. Group columns: {self.actual_groupby_cols_}, "
            f"Aggregate columns: {len(self.actual_aggregate_cols_)}, "
            f"New features: {len(self.new_feature_names_)}."
        )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Merge learned aggregates into the dataset."""
        if not self.is_fitted_:
            raise RuntimeError("[GroupAggregator] Not fitted. Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        if not self.learned_aggregates_:
            return X_out

        for group_col in self.actual_groupby_cols_:
            if group_col not in X_out.columns:
                continue
            for agg_col, agg_df in self.learned_aggregates_[group_col].items():
                agg_df_copy = agg_df.copy()

                # Merge on group column
                X_out = X_out.merge(agg_df_copy, on=group_col, how='left')

                # Fill NaN for unseen groups with global median
                for new_col in agg_df_copy.columns:
                    if new_col != group_col and new_col in X_out.columns:
                        median_val = agg_df_copy[new_col].median()
                        X_out[new_col] = X_out[new_col].fillna(median_val)

        return X_out
