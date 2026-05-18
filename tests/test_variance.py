import pytest
import pandas as pd
import numpy as np
from prunex.selectors.variance_threshold import VarianceThresholdSelector

def test_variance_threshold_drops_zero_variance():
    # Arrange
    X = pd.DataFrame({
        'high_var': [1, 100, 10, 50, 20],
        'zero_var': [5, 5, 5, 5, 5],
        'small_but_real_var': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    })
    y = pd.Series([1, 0, 1, 0, 1])
    
    # Act
    selector = VarianceThresholdSelector(threshold=0.01)
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    
    # Assert
    assert 'high_var' in X_transformed.columns
    assert 'small_but_real_var' in X_transformed.columns  # MinMaxScaler scales this up!
    assert 'zero_var' not in X_transformed.columns
