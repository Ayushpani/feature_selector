import pandas as pd
import numpy as np
from feature_engine_pro.engine import FeatureEngine
from sklearn.datasets import load_breast_cancer

def test_engine_breast_cancer():
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Introduce some artificial bad columns to test our engine
    X['constant_col'] = 5  # Zero variance
    X['duplicate_mean_radius'] = X['mean radius'] * 1.0001  # Perfect correlation
    X['category_col'] = np.where(y == 1, 'Type_A', 'Type_B')  # Categorical string

    # Intentionally missing values
    X.loc[0:10, 'mean texture'] = np.nan
    X.loc[0:5, 'category_col'] = np.nan

    original_shape = X.shape[1]

    # Initialize Engine
    engine = FeatureEngine(target_column='target', variance_threshold=0.01, correlation_threshold=0.90)

    # Run pipeline
    X_selected = engine.fit_transform(X, y)

    # Validate
    assert 'constant_col' not in X_selected.columns, "Engine failed to drop zero-variance column"
    assert 'duplicate_mean_radius' not in X_selected.columns or 'mean radius' not in X_selected.columns, "Engine failed to drop highly correlated columns"
    assert X_selected.isnull().sum().sum() == 0, "Engine failed to impute missing values securely"

    # Ensure categorical was converted to numeric
    assert pd.api.types.is_numeric_dtype(X_selected['category_col']), "Engine failed to encode categorical text"

    # Output report test
    engine.print_summary()
    engine.generate_report("test_report.html")

    print(f"\nOriginal columns: {original_shape}")
    print(f"Final columns: {X_selected.shape[1]}")
    assert X_selected.shape[1] < original_shape, "Dimensionality was not reduced!"
