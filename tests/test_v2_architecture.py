import pandas as pd
import numpy as np
from feature_engine_pro.engine import FeatureEngine
from sklearn.datasets import load_diabetes

def test_full_pipeline_v2():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Inject a datetime string column to test DatetimeExtractor
    X['signup_date'] = pd.date_range(start='2020-01-01', periods=len(X)).astype(str)

    # Inject an ID column to test GroupAggregator
    X['user_id'] = np.random.randint(0, 10, size=len(X))

    original_shape = X.shape[1]

    # Initialize Engine with tight thresholds to force dropping
    engine = FeatureEngine(
        target_column='target',
        problem_type='regression',
        variance_threshold=0.001,
        correlation_threshold=0.70,
        mi_threshold=0.01,
        rfe_n_features=5
    )

    # Run fit
    engine.fit(X, y)

    # Run transform
    X_out = engine.transform(X)

    # The datetime should be expanded, groups aggregated, and then math filtered down to rfe_n_features
    assert X_out.shape[1] <= 5 + 1 # 5 numerical + maybe user_id if kept, depending on randomness, but definitely strictly reduced

    # Test Reporter Visualization Output
    engine.generate_report("test_report_v2.html")

    # Basic check to ensure the report was generated
    import os
    assert os.path.exists("test_report_v2.html")
    with open("test_report_v2.html", "r") as f:
        html_content = f.read()
        assert "<img" in html_content, "Images were not embedded in the HTML report!"
        assert "Funnel Chart" in html_content, "Funnel chart missing!"

    print("Full Pipeline V2 Test Passed.")
