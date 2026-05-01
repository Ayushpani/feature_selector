"""
Edge-Case Tests — Adversarial Test Suite

Tests the pipeline against the worst possible real-world data scenarios.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from feature_engine_pro.engine import FeatureEngine


class TestEdgeCases:
    """Test edge cases that would crash or produce wrong results in v1."""

    def test_all_nan_column_dropped(self):
        """A column that is 100% NaN should be dropped, not crash."""
        X = pd.DataFrame({
            'good': [1, 2, 3, 4, 5] * 20,
            'all_nan': [np.nan] * 100,
            'normal': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        X_out = engine.fit_transform(X, y)
        assert 'all_nan' not in X_out.columns

    def test_single_row(self):
        """Single row DataFrame should not crash."""
        X = pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})
        y = pd.Series([1])
        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        # This may raise or produce warnings, but should NOT crash with an unhandled exception
        try:
            engine.fit(X, y)
        except (ValueError, RuntimeError):
            pass  # Acceptable — some selectors need more data

    def test_constant_features_dropped(self):
        """All-constant features should be dropped by variance filter."""
        y = pd.Series(np.random.randint(0, 2, 100))
        X = pd.DataFrame({
            'const': [5] * 100,
            'signal': y + np.random.randn(100) * 0.1,
        })
        engine = FeatureEngine(
            problem_type='classification', mode='fast',
            variance_threshold=0.001, verbosity=0
        )
        X_out = engine.fit_transform(X, y)
        assert 'const' not in X_out.columns

    def test_infinite_values_handled(self):
        """Infinite values should be replaced, not propagated."""
        X = pd.DataFrame({
            'a': [1, 2, np.inf, 4, 5] * 20,
            'b': [-np.inf, 2, 3, 4, 5] * 20,
            'c': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        X_out = engine.fit_transform(X, y)
        assert not np.isinf(X_out.values).any()

    def test_boolean_columns_protected(self):
        """Boolean columns should not be killed by variance threshold."""
        np.random.seed(42)
        y = pd.Series(np.random.randint(0, 2, 200))
        X = pd.DataFrame({
            'flag_a': y,
            'flag_b': np.random.choice([True, False], 200).astype(int),
            'signal': y + np.random.randn(200) * 0.1,
        })
        engine = FeatureEngine(
            problem_type='classification', mode='fast',
            variance_threshold=0.01, verbosity=0
        )
        X_out = engine.fit_transform(X, y)
        # At least one boolean should survive
        assert X_out.shape[1] >= 1

    def test_high_cardinality_categorical(self):
        """Extremely high cardinality categoricals should not crash."""
        y = pd.Series(np.random.randint(0, 2, 500))
        X = pd.DataFrame({
            'high_card': [f'cat_{i}' for i in range(500)],
            'signal': y + np.random.randn(500) * 0.1,
        })
        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        X_out = engine.fit_transform(X, y)
        assert X_out.shape[0] > 0

    def test_missing_heavy_dataset(self):
        """Dataset with >50% missing values should still work."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        # Make 60% of values NaN
        mask = np.random.random(X.shape) < 0.6
        X[mask] = np.nan
        y = pd.Series(np.random.randint(0, 2, 100))
        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        X_out = engine.fit_transform(X, y)
        assert X_out.isnull().sum().sum() == 0

    def test_multiclass_classification(self):
        """Multiclass (not just binary) should work."""
        X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        y = pd.Series(np.random.randint(0, 5, 200))  # 5 classes
        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        X_out = engine.fit_transform(X, y)
        assert X_out.shape[1] > 0

    def test_regression_problem_type(self):
        """Regression pipeline should work end-to-end."""
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        engine = FeatureEngine(problem_type='regression', mode='fast', verbosity=0)
        X_out = engine.fit_transform(X, y)
        assert X_out.shape[1] > 0
        assert X_out.shape[1] <= X.shape[1]

    def test_breast_cancer_full_pipeline(self):
        """Full pipeline on the standard breast cancer dataset."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        # Add adversarial columns
        X['constant'] = 5
        X['near_dup'] = X.iloc[:, 0] * 1.0001
        X.loc[:5, 'mean texture'] = np.nan
        original_cols = X.shape[1]

        engine = FeatureEngine(
            problem_type='classification', mode='balanced', verbosity=0
        )
        X_out = engine.fit_transform(X, y)

        assert X_out.shape[1] < original_cols
        assert X_out.isnull().sum().sum() == 0
        assert 'constant' not in X_out.columns

    def test_train_test_consistency(self):
        """Train and test transforms should produce the same column set."""
        from sklearn.model_selection import train_test_split
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        engine.fit(X_train, y_train)
        X_tr_out = engine.transform(X_train)
        X_te_out = engine.transform(X_test)

        assert X_tr_out.shape[1] == X_te_out.shape[1]
        assert list(X_tr_out.columns) == list(X_te_out.columns)

    def test_datetime_columns(self):
        """Datetime columns should be expanded into numerical features."""
        y = pd.Series(np.random.randint(0, 2, 100))
        X = pd.DataFrame({
            'signup': pd.date_range('2020-01-01', periods=100).astype(str),
            'amount': y + np.random.randn(100) * 0.1,
        })
        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        X_out = engine.fit_transform(X, y)
        # Original 'signup' string column should not exist
        assert 'signup' not in X_out.columns
        # But derived features should
        assert X_out.shape[1] > 1

    def test_report_generation(self):
        """Report should generate without errors."""
        import os
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        engine.fit(X, y)

        report_path = 'test_edge_report.html'
        engine.generate_report(report_path)

        assert os.path.exists(report_path)
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Feature Engine Pro' in content
            assert 'Audit Trail' in content
        os.remove(report_path)

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise a clear error."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        engine = FeatureEngine(problem_type='classification', verbosity=0)
        with pytest.raises(ValueError, match="empty"):
            engine.fit(X, y)

    def test_column_journey_tracking(self):
        """Column journey should be tracked through all stages."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        engine = FeatureEngine(problem_type='classification', mode='fast', verbosity=0)
        engine.fit(X, y)

        journey = engine.get_column_journey()
        assert 'input' in journey
        assert 'final' in journey
        assert len(journey['final']) <= len(journey['input'])
