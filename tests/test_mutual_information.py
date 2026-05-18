import pytest
import pandas as pd
import numpy as np
from feature_engine_pro.selectors.mutual_information import MutualInformationSelector
from feature_engine_pro.reporter import Reporter

def test_mutual_information_dynamic_threshold_and_leakage():
    # Arrange
    np.random.seed(42)
    # Using continuous variables so MI can easily exceed 0.90 when perfectly correlated
    X = pd.DataFrame({
        'pure_noise': np.random.randn(100),
        'weak_signal': np.random.randn(100),
        'leakage': np.zeros(100)
    })
    y = pd.Series(np.random.randn(100) * 10) # Continuous target with high variance
    
    # Force leakage to be perfect predictor
    X.loc[:, 'leakage'] = y.values
    # Force weak signal to be slightly predictive
    X.loc[0:50, 'weak_signal'] = y.values[0:51]

    # Act
    selector = MutualInformationSelector(threshold=0.01, problem_type='regression')
    selector.reporter = Reporter()
    selector.fit(X, y)
    
    # Assert
    assert 'leakage' in selector.selected_features_
    assert 'pure_noise' not in selector.selected_features_
    
    # Verify leakage warning was logged
    leakage_log = [log for log in selector.reporter.logs if log['feature'] == 'leakage']
    assert len(leakage_log) == 1
    assert 'TARGET LEAKAGE WARNING' in leakage_log[0]['reason']
