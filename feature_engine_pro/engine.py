import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine_pro.data_processor import DataProcessor
from feature_engine_pro.transformers.categorical_encoder import AutoCategoricalEncoder
<<<<<<< Updated upstream
from feature_engine_pro.selectors.variance_threshold import VarianceThresholdSelector
from feature_engine_pro.selectors.correlation import CorrelationSelector
=======
from feature_engine_pro.transformers.datetime_extractor import DatetimeExtractor
from feature_engine_pro.transformers.group_aggregator import GroupAggregator
from feature_engine_pro.selectors.variance_threshold import VarianceThresholdSelector
from feature_engine_pro.selectors.correlation import CorrelationSelector
from feature_engine_pro.selectors.mutual_information import MutualInformationSelector
from feature_engine_pro.selectors.rfe import RFESelector
>>>>>>> Stashed changes
from feature_engine_pro.reporter import Reporter

class FeatureEngine(BaseEstimator, TransformerMixin):
    """
    The orchestrator that runs the entire end-to-end mathematical feature selection pipeline.
    It runs preprocessing, encoding, and selection stages, then generates a comprehensive report.
    """
<<<<<<< Updated upstream
    def __init__(self, target_column=None, problem_type='classification',
                 variance_threshold=0.01, correlation_threshold=0.85):
=======
    def __init__(self, target_column=None, problem_type='classification',
                 variance_threshold=0.01, correlation_threshold=0.85,
                 mi_threshold=0.01, rfe_n_features=None):
        # We store these explicitly for scikit-learn's get_params/set_params to work
>>>>>>> Stashed changes
        self.target_column = target_column
        self.problem_type = problem_type
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
<<<<<<< Updated upstream

        # Initialize Core Utilities
        self.reporter = Reporter()
        self.data_processor = DataProcessor()
        self.encoder = AutoCategoricalEncoder(target_column=target_column, problem_type=problem_type)

        # Initialize Pipeline Selectors
        self.variance_selector = VarianceThresholdSelector(threshold=variance_threshold, target_column=target_column, problem_type=problem_type)
        self.correlation_selector = CorrelationSelector(threshold=correlation_threshold, target_column=target_column, problem_type=problem_type)

        # Inject reporter
        self.variance_selector.reporter = self.reporter
        self.correlation_selector.reporter = self.reporter

=======
        self.mi_threshold = mi_threshold
        self.rfe_n_features = rfe_n_features

>>>>>>> Stashed changes
        self.selected_features_ = None

    def fit(self, X, y=None):
        """
        Fits the entire pipeline on the training data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
<<<<<<< Updated upstream

        print("Starting Feature Engine Pipeline (Fit)...")

        # Step 1: Pre-process and clean
        self.data_processor.fit(X, y)
        X_clean = self.data_processor.transform(X)

        # Step 2: Categorical Encoding
        self.encoder.fit(X_clean, y)
        X_encoded = self.encoder.transform(X_clean)

        # Step 3: Math Stage 1 - Variance
        self.variance_selector.fit(X_encoded, y)
        X_stage1 = self.variance_selector.transform(X_encoded)

        # Step 4: Math Stage 2 - Correlation & Multicollinearity
        self.correlation_selector.fit(X_stage1, y)

        self.selected_features_ = self.correlation_selector.selected_features_
        print(f"Pipeline Fit Complete. Selected {len(self.selected_features_)} features out of {X.shape[1]}.")

=======

        print("Starting Feature Engine Pipeline (Fit)...")

        # Instantiate dependencies dynamically during fit so they pick up
        # any updated thresholds from GridSearchCV set_params
        self.reporter = Reporter()
        self.data_processor = DataProcessor()
        self.datetime_extractor = DatetimeExtractor()
        self.group_aggregator = GroupAggregator()
        self.encoder = AutoCategoricalEncoder(target_column=self.target_column, problem_type=self.problem_type)

        self.variance_selector = VarianceThresholdSelector(
            threshold=self.variance_threshold,
            target_column=self.target_column,
            problem_type=self.problem_type
        )
        self.correlation_selector = CorrelationSelector(
            threshold=self.correlation_threshold,
            target_column=self.target_column,
            problem_type=self.problem_type
        )
        self.mi_selector = MutualInformationSelector(
            threshold=self.mi_threshold,
            target_column=self.target_column,
            problem_type=self.problem_type
        )
        self.rfe_selector = RFESelector(
            n_features_to_select=self.rfe_n_features,
            target_column=self.target_column,
            problem_type=self.problem_type
        )

        self.variance_selector.reporter = self.reporter
        self.correlation_selector.reporter = self.reporter
        self.mi_selector.reporter = self.reporter
        self.rfe_selector.reporter = self.reporter

        # Step 1: Feature Engineering - Datetime Expansion & Group Aggregations
        self.datetime_extractor.fit(X, y)
        X_dt = self.datetime_extractor.transform(X)
        self.group_aggregator.fit(X_dt, y)
        X_eng = self.group_aggregator.transform(X_dt)

        # Step 2: Pre-process and clean
        self.data_processor.fit(X_eng, y)
        X_clean = self.data_processor.transform(X_eng)

        # Step 3: Categorical Encoding
        self.encoder.fit(X_clean, y)
        X_encoded = self.encoder.transform(X_clean)

        # Capture pre-correlation matrix for reporter
        self.reporter.capture_correlation_matrix(X_encoded)

        # Step 4: Math Stage 1 - Variance
        self.variance_selector.fit(X_encoded, y)
        X_stage1 = self.variance_selector.transform(X_encoded)

        # Step 4: Math Stage 2 - Correlation & Multicollinearity
        self.correlation_selector.fit(X_stage1, y)
        X_stage2 = self.correlation_selector.transform(X_stage1)

        # Step 5: Math Stage 3 - Mutual Information (Non-linear)
        self.mi_selector.fit(X_stage2, y)
        X_stage3 = self.mi_selector.transform(X_stage2)

        # Step 6: Math Stage 4 - Recursive Feature Elimination (Model-based)
        self.rfe_selector.fit(X_stage3, y)

        self.selected_features_ = self.rfe_selector.selected_features_
        print(f"Pipeline Fit Complete. Selected {len(self.selected_features_)} features out of {X.shape[1]}.")

>>>>>>> Stashed changes
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
<<<<<<< Updated upstream

        if self.selected_features_ is None:
            raise ValueError("FeatureEngine has not been fitted yet.")

        # Apply data processing and encoding using learned states
        X_clean = self.data_processor.transform(X)
        X_encoded = self.encoder.transform(X_clean)

=======

        if self.selected_features_ is None:
            raise ValueError("FeatureEngine has not been fitted yet.")

        # Apply data processing and encoding using learned states
        X_dt = self.datetime_extractor.transform(X)
        X_eng = self.group_aggregator.transform(X_dt)
        X_clean = self.data_processor.transform(X_eng)
        X_encoded = self.encoder.transform(X_clean)

>>>>>>> Stashed changes
        # Keep only the features that survived the selection funnel
        cols_to_keep = [col for col in self.selected_features_ if col in X_encoded.columns]
        return X_encoded[cols_to_keep]

    def generate_report(self, filepath="feature_engine_report.html"):
        """
        Triggers the reporter to generate the HTML audit trail.
        """
<<<<<<< Updated upstream
        self.reporter.generate_html_report(filepath=filepath)

=======
        if not hasattr(self, 'reporter'):
             raise ValueError("You must call .fit() before generating a report.")
        self.reporter.generate_html_report(filepath=filepath)

>>>>>>> Stashed changes
    def print_summary(self):
        """
        Prints the summary audit trail to the console.
        """
<<<<<<< Updated upstream
=======
        if not hasattr(self, 'reporter'):
             raise ValueError("You must call .fit() before printing a summary.")
>>>>>>> Stashed changes
        self.reporter.print_report()
