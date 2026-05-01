import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GroupAggregator(BaseEstimator, TransformerMixin):
    """
    Creates grouped aggregate features (mean, sum) based on specified grouping columns.
    For example, calculating the 'mean_transaction_amount' grouped by 'customer_id'.
    """
    def __init__(self, groupby_cols=None, aggregate_cols=None, aggregations=None):
        """
        :param groupby_cols: list of columns to group by. If None, inference will be attempted (e.g. looking for ID columns).
        :param aggregate_cols: list of numerical columns to aggregate over.
        :param aggregations: list of functions to apply (e.g. ['mean', 'max', 'min']). Defaults to ['mean'].
        """
        self.groupby_cols = groupby_cols
        self.aggregate_cols = aggregate_cols
        self.aggregations = aggregations if aggregations else ['mean']
        self.learned_aggregates_ = {}
        self.actual_groupby_cols_ = []
        self.actual_aggregate_cols_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.actual_groupby_cols_ = self.groupby_cols if self.groupby_cols else []
        self.actual_aggregate_cols_ = self.aggregate_cols if self.aggregate_cols else X.select_dtypes(include='number').columns.tolist()

        # Simple inference: if groupby_cols is None, try to find columns with "id" in the name
        if not self.groupby_cols:
             for col in X.columns:
                 if 'id' in col.lower() and X[col].nunique() < len(X) * 0.5: # heuristic for categorical IDs
                     self.actual_groupby_cols_.append(col)

        if not self.actual_groupby_cols_:
             # If we still have none, we can't do anything
             return self

        # Filter aggregate cols so we don't group by and aggregate the same column
        self.actual_aggregate_cols_ = [c for c in self.actual_aggregate_cols_ if c not in self.actual_groupby_cols_]

        # Learn the aggregations from the training set
        for group_col in self.actual_groupby_cols_:
            self.learned_aggregates_[group_col] = {}
            for agg_col in self.actual_aggregate_cols_:
                 if agg_col in X.columns:
                     agg_df = X.groupby(group_col)[agg_col].agg(self.aggregations).reset_index()
                     self.learned_aggregates_[group_col][agg_col] = agg_df

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        if not self.learned_aggregates_:
             return X_out

        # Merge learned aggregates into the target dataset based on the group_col
        for group_col in self.actual_groupby_cols_:
            if group_col in X_out.columns:
                for agg_col, agg_df in self.learned_aggregates_[group_col].items():
                     # Rename the agg columns to something like "customer_id_mean_amount"
                     rename_dict = {func: f"{group_col}_{func}_{agg_col}" for func in self.aggregations}
                     agg_df_renamed = agg_df.rename(columns=rename_dict)

                     X_out = X_out.merge(agg_df_renamed, on=group_col, how='left')

                     # Fill NaN for unseen groups with the median of the aggregate feature
                     for new_col in rename_dict.values():
                         X_out[new_col] = X_out[new_col].fillna(agg_df_renamed[new_col].median())

        return X_out
