import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from feature_engine_pro.engine import FeatureEngine

def test_engine_end_to_end():
    # Arrange
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    engine = FeatureEngine(
        target_column='target',
        problem_type='classification',
        variance_threshold=0.01,
        correlation_threshold=0.85,
        mi_threshold=0.01,
        rfe_n_features=10
    )
    
    # Act
    engine.fit(X, y)
    X_transformed = engine.transform(X)
    
    # Assert
    assert X_transformed.shape[1] == 10
    assert hasattr(engine, 'reporter_')
    assert engine.reporter_.pipeline_runtime > 0
