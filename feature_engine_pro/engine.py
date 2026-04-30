import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine_pro.data_processor import DataProcessor
from feature_engine_pro.transformers.categorical_encoder import AutoCategoricalEncoder
from feature_engine_pro.selectors.variance_threshold import VarianceThresholdSelector
from feature_engine_pro.selectors.correlation import CorrelationSelector
from feature_engine_pro.reporter import Reporter

class FeatureEngine(BaseEstimator, TransformerMixin):
    """
    The orchestrator that runs the entire end-to-end mathematical feature selection pipeline.
    It runs preprocessing, encoding, and selection stages, then generates a comprehensive report.
    """
    def __init__(self, target_column=None, problem_type='classification',
                 variance_threshold=0.01, correlation_threshold=0.85):
        self.target_column = target_column
        self.problem_type = problem_type
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

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

        self.selected_features_ = None

    def fit(self, X, y=None):
        """
        Fits the entire pipeline on the training data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

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

        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.selected_features_ is None:
            raise ValueError("FeatureEngine has not been fitted yet.")

        # Apply data processing and encoding using learned states
        X_clean = self.data_processor.transform(X)
        X_encoded = self.encoder.transform(X_clean)

        # Keep only the features that survived the selection funnel
        cols_to_keep = [col for col in self.selected_features_ if col in X_encoded.columns]
        return X_encoded[cols_to_keep]

    def generate_report(self, filepath="feature_engine_report.html"):
        """
        Triggers the reporter to generate the HTML audit trail.
        """
        self.reporter.generate_html_report(filepath=filepath)

    def print_summary(self):
        """
        Prints the summary audit trail to the console.
        """
        self.reporter.print_report()
