"""
FeatureEngine — The Industry-Grade Orchestrator

The main entry point for the Feature Engine Pro pipeline. This class
orchestrates the entire end-to-end pipeline from raw messy data to
optimally selected, clean features.

Pipeline Order:
1. Datetime Expansion (cyclic encoding)
2. Group Aggregation
3. Outlier Handling (IQR/Z-Score/MAD)
4. Data Cleaning & Imputation
5. Log/Skewness Correction
6. Categorical Encoding (smoothed target encoding)
7. Polynomial Interaction Features
8. Variance Threshold Filter
9. Correlation / Multicollinearity Filter
10. Mutual Information Filter
11. Statistical Hypothesis Tests
12. Recursive Feature Elimination (RFECV)
13. SHAP-based Selection (optional)
14. Performance Evaluation (before/after proof)
"""
import time
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine_pro.data_processor import DataProcessor
from feature_engine_pro.transformers.datetime_extractor import DatetimeExtractor
from feature_engine_pro.transformers.group_aggregator import GroupAggregator
from feature_engine_pro.transformers.outlier_handler import OutlierHandler
from feature_engine_pro.transformers.log_transform import LogTransformer
from feature_engine_pro.transformers.categorical_encoder import AutoCategoricalEncoder
from feature_engine_pro.transformers.polynomial_features import PolynomialFeatureGenerator
from feature_engine_pro.selectors.variance_threshold import VarianceThresholdSelector
from feature_engine_pro.selectors.correlation import CorrelationSelector
from feature_engine_pro.selectors.mutual_information import MutualInformationSelector
from feature_engine_pro.selectors.select_k_best import StatisticalSelector
from feature_engine_pro.selectors.rfe import RFESelector
from feature_engine_pro.selectors.shap_selector import SHAPSelector, _shap_available
from feature_engine_pro.evaluator import Evaluator
from feature_engine_pro.reporter import Reporter
from feature_engine_pro.logger import get_logger


class FeatureEngine(BaseEstimator, TransformerMixin):
    """
    The orchestrator that runs the entire end-to-end mathematical
    feature selection pipeline.

    Parameters
    ----------
    target_column : str, optional
        Name of the target column (for reference only).
    problem_type : str, default='classification'
        'classification' or 'regression'.
    mode : str, default='balanced'
        Pipeline aggressiveness preset:
        - 'fast': Skip SHAP, polynomial features, use greedy correlation, RFE (not RFECV).
        - 'balanced': Full pipeline with reasonable defaults.
        - 'thorough': Enable SHAP, RFECV, multiple MI runs, VIF, clustered correlation.
    variance_threshold : float, default=0.01
        Minimum variance to keep a feature.
    correlation_threshold : float, default=0.85
        Correlation above which features are considered redundant.
    mi_threshold : float, default=0.01
        Minimum mutual information score.
    rfe_n_features : int or None, default=None
        Target number of features for RFE. None = auto-determine.
    outlier_method : str, default='iqr'
        Outlier detection method: 'iqr', 'zscore', 'mad'.
    outlier_treatment : str, default='clip'
        Outlier treatment: 'clip', 'nan', 'drop'.
    enable_polynomial : bool, default=True
        Generate polynomial interaction features.
    enable_shap : bool, default='auto'
        Enable SHAP-based selection. 'auto' = True in 'thorough' mode.
    enable_evaluation : bool, default=True
        Run before/after model performance evaluation.
    verbosity : int, default=1
        0=silent, 1=summary, 2=detailed.
    """
    def __init__(self, target_column=None, problem_type='classification',
                 mode='balanced',
                 variance_threshold=0.01, correlation_threshold=0.85,
                 mi_threshold=0.01, rfe_n_features=None,
                 outlier_method='iqr', outlier_treatment='clip',
                 enable_polynomial=True, enable_shap='auto',
                 enable_evaluation=True, verbosity=1):
        self.target_column = target_column
        self.problem_type = problem_type
        self.mode = mode
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold
        self.rfe_n_features = rfe_n_features
        self.outlier_method = outlier_method
        self.outlier_treatment = outlier_treatment
        self.enable_polynomial = enable_polynomial
        self.enable_shap = enable_shap
        self.enable_evaluation = enable_evaluation
        self.verbosity = verbosity

        self.selected_features_ = None
        self.column_journey_ = {}
        self.pipeline_runtime_ = None

    def _resolve_mode(self):
        """Resolve mode presets into component configurations."""
        config = {}

        if self.mode == 'fast':
            config['use_clustering'] = False
            config['vif_threshold'] = None
            config['mi_repeats'] = 1
            config['rfe_use_cv'] = False
            config['use_shap'] = False
            config['use_polynomial'] = False
            config['use_statistical'] = False

        elif self.mode == 'balanced':
            config['use_clustering'] = True
            config['vif_threshold'] = None
            config['mi_repeats'] = 3
            config['rfe_use_cv'] = True
            config['use_shap'] = False
            config['use_polynomial'] = self.enable_polynomial
            config['use_statistical'] = True

        elif self.mode == 'thorough':
            config['use_clustering'] = True
            config['vif_threshold'] = 10.0
            config['mi_repeats'] = 5
            config['rfe_use_cv'] = True
            config['use_shap'] = True
            config['use_polynomial'] = True
            config['use_statistical'] = True

        else:
            raise ValueError(f"Unknown mode: '{self.mode}'. Choose 'fast', 'balanced', or 'thorough'.")

        # Override SHAP if explicitly set
        if self.enable_shap is True:
            config['use_shap'] = True
        elif self.enable_shap is False:
            config['use_shap'] = False

        return config

    def fit(self, X, y=None):
        """
        Fits the entire pipeline on the training data.
        """
        start_time = time.time()
        logger = get_logger(verbosity=self.verbosity)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.empty:
            raise ValueError("[FeatureEngine] Input DataFrame is empty.")

        config = self._resolve_mode()

        logger.info(f"[FeatureEngine] Starting pipeline (mode={self.mode})...")
        logger.info(f"[FeatureEngine] Input shape: {X.shape}")

        self.column_journey_ = {'input': X.columns.tolist()}

        # --- Initialize Reporter ---
        self.reporter_ = Reporter()

        # --- Step 1: Feature Engineering — Datetime Expansion ---
        self.datetime_extractor_ = DatetimeExtractor()
        self.datetime_extractor_.fit(X, y)
        X = self.datetime_extractor_.transform(X)
        self.column_journey_['after_datetime'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After datetime extraction: {X.shape}")

        # --- Step 2: Feature Engineering — Group Aggregation ---
        self.group_aggregator_ = GroupAggregator()
        self.group_aggregator_.fit(X, y)
        X = self.group_aggregator_.transform(X)
        self.column_journey_['after_group_agg'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After group aggregation: {X.shape}")

        # --- Step 3: Outlier Handling ---
        self.outlier_handler_ = OutlierHandler(
            method=self.outlier_method, treatment=self.outlier_treatment
        )
        self.outlier_handler_.fit(X, y)
        X = self.outlier_handler_.transform(X)
        # If treatment is 'drop', y needs to be re-aligned
        if self.outlier_treatment == 'drop' and y is not None:
            if isinstance(y, pd.Series):
                y = y.iloc[X.index] if hasattr(X, 'index') else y[:len(X)]
            else:
                y = pd.Series(y)[:len(X)]
        self.column_journey_['after_outlier'] = X.columns.tolist()

        # --- Step 4: Data Cleaning & Imputation ---
        self.data_processor_ = DataProcessor()
        self.data_processor_.fit(X, y)
        X = self.data_processor_.transform(X)
        self.column_journey_['after_cleaning'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After cleaning & imputation: {X.shape}")

        # --- Step 5: Log/Skewness Correction ---
        self.log_transformer_ = LogTransformer()
        self.log_transformer_.fit(X, y)
        X = self.log_transformer_.transform(X)
        self.column_journey_['after_log_transform'] = X.columns.tolist()

        # --- Step 6: Categorical Encoding ---
        self.encoder_ = AutoCategoricalEncoder(
            target_column=self.target_column, problem_type=self.problem_type
        )
        self.encoder_.fit(X, y)
        X = self.encoder_.transform(X)
        self.column_journey_['after_encoding'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After encoding: {X.shape}")

        # --- Step 7: Polynomial Features (optional) ---
        self.poly_generator_ = None
        if config['use_polynomial']:
            self.poly_generator_ = PolynomialFeatureGenerator()
            self.poly_generator_.fit(X, y)
            X = self.poly_generator_.transform(X)
            self.column_journey_['after_polynomial'] = X.columns.tolist()
            logger.info(f"[FeatureEngine] After polynomial features: {X.shape}")

        # Save pre-selection state for evaluation
        X_pre_selection = X.copy()

        # Capture correlation matrix for reporter
        self.reporter_.capture_correlation_matrix(X)

        # =========================================
        # MATHEMATICAL SELECTION FUNNEL
        # =========================================

        # --- Stage 1: Variance Threshold ---
        self.variance_selector_ = VarianceThresholdSelector(
            threshold=self.variance_threshold,
            target_column=self.target_column,
            problem_type=self.problem_type
        )
        self.variance_selector_.reporter = self.reporter_
        self.variance_selector_.fit(X, y)
        X = self.variance_selector_.transform(X)
        self.column_journey_['after_variance'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After variance filter: {X.shape}")

        # --- Stage 2: Correlation / Multicollinearity ---
        self.correlation_selector_ = CorrelationSelector(
            threshold=self.correlation_threshold,
            target_column=self.target_column,
            problem_type=self.problem_type,
            use_clustering=config['use_clustering'],
            vif_threshold=config['vif_threshold']
        )
        self.correlation_selector_.reporter = self.reporter_
        self.correlation_selector_.fit(X, y)
        X = self.correlation_selector_.transform(X)
        self.column_journey_['after_correlation'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After correlation filter: {X.shape}")

        # --- Stage 3: Mutual Information ---
        self.mi_selector_ = MutualInformationSelector(
            threshold=self.mi_threshold,
            target_column=self.target_column,
            problem_type=self.problem_type,
            n_repeats=config['mi_repeats']
        )
        self.mi_selector_.reporter = self.reporter_
        self.mi_selector_.fit(X, y)
        X = self.mi_selector_.transform(X)
        self.column_journey_['after_mi'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After mutual information filter: {X.shape}")

        # --- Stage 4: Statistical Tests (optional) ---
        self.statistical_selector_ = None
        if config['use_statistical']:
            self.statistical_selector_ = StatisticalSelector(
                target_column=self.target_column,
                problem_type=self.problem_type
            )
            self.statistical_selector_.reporter = self.reporter_
            self.statistical_selector_.fit(X, y)
            X = self.statistical_selector_.transform(X)
            self.column_journey_['after_statistical'] = X.columns.tolist()
            logger.info(f"[FeatureEngine] After statistical tests: {X.shape}")

        # --- Stage 5: RFE ---
        self.rfe_selector_ = RFESelector(
            n_features_to_select=self.rfe_n_features,
            target_column=self.target_column,
            problem_type=self.problem_type,
            use_cv=config['rfe_use_cv']
        )
        self.rfe_selector_.reporter = self.reporter_
        self.rfe_selector_.fit(X, y)
        X = self.rfe_selector_.transform(X)
        self.column_journey_['after_rfe'] = X.columns.tolist()
        logger.info(f"[FeatureEngine] After RFE: {X.shape}")

        # --- Stage 6: SHAP (optional) ---
        self.shap_selector_ = None
        if config['use_shap'] and _shap_available():
            self.shap_selector_ = SHAPSelector(
                target_column=self.target_column,
                problem_type=self.problem_type
            )
            self.shap_selector_.reporter = self.reporter_
            self.shap_selector_.fit(X, y)
            X = self.shap_selector_.transform(X)
            self.column_journey_['after_shap'] = X.columns.tolist()
            logger.info(f"[FeatureEngine] After SHAP selection: {X.shape}")

        self.selected_features_ = X.columns.tolist()
        self.column_journey_['final'] = self.selected_features_

        # --- Evaluation: Before/After Performance Comparison ---
        self.evaluator_ = None
        self.evaluation_results_ = None
        if self.enable_evaluation and y is not None:
            try:
                self.evaluator_ = Evaluator(problem_type=self.problem_type)
                self.evaluation_results_ = self.evaluator_.evaluate(
                    X_pre_selection, X, y
                )
                self.reporter_.evaluation_results = self.evaluation_results_
            except Exception as e:
                logger.warning(f"[FeatureEngine] Evaluation failed: {e}")

        self.pipeline_runtime_ = round(time.time() - start_time, 2)
        self.reporter_.pipeline_runtime = self.pipeline_runtime_
        self.reporter_.column_journey = self.column_journey_

        logger.info(
            f"[FeatureEngine] ✅ Pipeline complete in {self.pipeline_runtime_}s. "
            f"Selected {len(self.selected_features_)} of "
            f"{len(self.column_journey_['input'])} original features "
            f"({100 - round(len(self.selected_features_) / max(len(self.column_journey_['input']), 1) * 100, 1)}% reduction)."
        )

        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.selected_features_ is None:
            raise ValueError("[FeatureEngine] Not fitted. Call .fit() first.")

        # Apply all transformers in order
        X = self.datetime_extractor_.transform(X)
        X = self.group_aggregator_.transform(X)
        X = self.outlier_handler_.transform(X)
        X = self.data_processor_.transform(X)
        X = self.log_transformer_.transform(X)
        X = self.encoder_.transform(X)

        if self.poly_generator_ is not None:
            X = self.poly_generator_.transform(X)

        # Keep only the features that survived selection
        cols_to_keep = [col for col in self.selected_features_ if col in X.columns]
        return X[cols_to_keep]

    def generate_report(self, filepath="feature_engine_report.html"):
        """
        Generate the professional HTML audit report.
        """
        if not hasattr(self, 'reporter_'):
            raise ValueError("[FeatureEngine] Call .fit() before generating a report.")
        self.reporter_.generate_html_report(filepath=filepath)

    def print_summary(self):
        """Print the audit trail to console."""
        if not hasattr(self, 'reporter_'):
            raise ValueError("[FeatureEngine] Call .fit() before printing a summary.")
        self.reporter_.print_report()

    def save(self, filepath):
        """Serialize the fitted pipeline to disk."""
        import joblib
        if self.selected_features_ is None:
            raise ValueError("[FeatureEngine] Not fitted. Call .fit() first.")
        joblib.dump(self, filepath)
        logger = get_logger(verbosity=self.verbosity)
        logger.info(f"[FeatureEngine] Pipeline saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load a fitted pipeline from disk."""
        import joblib
        engine = joblib.load(filepath)
        logger = get_logger()
        logger.info(f"[FeatureEngine] Pipeline loaded from {filepath}")
        return engine

    def get_column_journey(self):
        """Get the full column tracking history."""
        if not self.column_journey_:
            raise ValueError("[FeatureEngine] Not fitted.")
        return self.column_journey_

    def get_selected_features(self):
        """Get the list of final selected features."""
        if self.selected_features_ is None:
            raise ValueError("[FeatureEngine] Not fitted.")
        return self.selected_features_
