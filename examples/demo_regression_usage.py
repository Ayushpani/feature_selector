"""
Demo Regression Usage - Full Messy Real-World Regression Example

Demonstrates the complete Feature Engine Pro pipeline on a messy, real-world regression dataset.
It features datetime extraction, group aggregation, missing values, high-cardinality categorical
encoding (target encoding), non-linear selection, and target leakage guards.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from feature_engine_pro.engine import FeatureEngine


def main():
    print("=" * 60)
    print("  Feature Engine Pro v2.0 - Messy Regression Demo")
    print("=" * 60)

    # 1. Load Regression Dataset (California Housing)
    raw_data = fetch_california_housing()
    X = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    y = pd.Series(raw_data.target)

    # 2. Inject Messy Real-World Anomalies
    np.random.seed(42)
    
    # A. Stage 1: Datetime Expansion Column
    X['transaction_timestamp'] = pd.date_range('2022-01-01', periods=len(X), freq='h').astype(str)
    
    # B. Stage 1: Group Aggregation Column (User ID)
    user_ids = [f"user_{i}" for i in np.random.randint(1, 100, size=len(X))]
    X['user_id'] = user_ids
    
    # C. Stage 2: High-cardinality target-encoded categorical feature
    neighborhoods = [f"district_{i}" for i in np.random.randint(1, 20, size=len(X))]
    X['neighborhood'] = neighborhoods
    
    # D. Missing Values (Numerical & Categorical)
    # 5% missing in HouseAge
    mask_age = np.random.rand(len(X)) < 0.05
    X.loc[mask_age, 'HouseAge'] = np.nan
    # 2% missing in neighborhood
    mask_neigh = np.random.rand(len(X)) < 0.02
    X.loc[mask_neigh, 'neighborhood'] = np.nan

    # E. Redundant/Collinear Feature
    X['collinear_MedInc'] = X['MedInc'] * 0.99 + np.random.normal(0, 0.01, size=len(X))

    # F. Zero Variance Feature
    X['constant_feature'] = 9.99

    # G. Pure Noise Feature
    X['random_noise'] = np.random.normal(0, 1, size=len(X))

    print(f"\nOriginal messy dataset: {X.shape[0]} rows x {X.shape[1]} columns")
    print("Injected Anomalies:")
    print(" - Stage 1: Datetime Column ('transaction_timestamp')")
    print(" - Stage 1: Group ID ('user_id') for Aggregations")
    print(" - Stage 2: Missing values injected into 'HouseAge' and 'neighborhood'")
    print(" - Stage 2: High-cardinality categorical column ('neighborhood') for Target Encoding")
    print(" - Stage 3: Zero-variance ('constant_feature') and collinear ('collinear_MedInc') columns")
    print(" - Stage 3: Pure noise column ('random_noise')")

    # 3. Train/Test Split (CRITICAL for preventing data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Initialize Feature Engine for Regression
    # We default mi_threshold to 'dynamic' (using our new flexible API!)
    engine = FeatureEngine(
        target_column='target',
        problem_type='regression',
        variance_threshold=0.01,
        correlation_threshold=0.85,
        mi_threshold='dynamic',
        rfe_n_features=8,
        enable_evaluation=True,
        verbosity=1,
    )

    # 5. Fit the entire mathematical funnel on training data only
    print("\nFitting Feature Engine Pro Pipeline...")
    engine.fit(X_train, y_train)

    # 6. Transform both train and test sets
    X_train_clean = engine.transform(X_train)
    X_test_clean = engine.transform(X_test)

    print(f"\nFinal dataset selected features: {X_train_clean.shape[1]} columns")
    print(f"Selected features list: {engine.get_selected_features()}")

    # 7. Generate beautiful interactive HTML and PDF Audit Reports
    html_path = "regression_report.html"
    pdf_path = "regression_report.pdf"
    
    print(f"\nGenerating visual HTML and PDF audit trail reports at '{html_path}' and '{pdf_path}'...")
    engine.reporter_.generate_pdf_report(
        html_filepath=html_path,
        pdf_filepath=pdf_path
    )
    print("[SUCCESS] Reports compiled successfully.")

    # 8. Print Console Summary
    print("\n--- Pipeline Summary ---")
    engine.print_summary()

    print("\n" + "=" * 60)
    print("  Regression Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
