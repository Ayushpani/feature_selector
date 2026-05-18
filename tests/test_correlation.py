import pytest
import pandas as pd
from feature_engine_pro.selectors.correlation import CorrelationSelector

def test_correlation_drops_lower_target_correlation():
    # Arrange
    X = pd.DataFrame({
        'f1': [1.5, 1.5, 3.5, 3.5, 5.0],
        'f2': [1.0, 2.0, 3.0, 4.0, 5.0], # Perfectly correlated with y
        'independent': [1, 0, 1, 0, 0]
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Act
    selector = CorrelationSelector(threshold=0.90)
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    
    # Assert
    assert 'independent' in X_transformed.columns
    assert 'f2' in X_transformed.columns
    assert 'f1' not in X_transformed.columns
