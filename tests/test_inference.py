import pandas as pd
from prunex.engine import FeatureEngine
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

def test_engine_inference():
    """Simulates inference time (no data leakage, transform only on test)."""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    engine = FeatureEngine(target_column='target', variance_threshold=0.0001, correlation_threshold=0.85)

    # Fit ONLY on training data
    engine.fit(X_train, y_train)

    # Transform BOTH train and test data
    X_train_selected = engine.transform(X_train)
    X_test_selected = engine.transform(X_test)

    assert X_train_selected.shape[1] == X_test_selected.shape[1], "Train and test feature counts mismatch!"
    assert all(X_train_selected.columns == X_test_selected.columns), "Train and test feature names mismatch!"
