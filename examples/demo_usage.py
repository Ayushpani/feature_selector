"""
Demo Usage - Full Working Example

Demonstrates the complete PruneX pipeline on the Breast Cancer dataset.
Generates an interactive HTML Audit Report and a high-fidelity PDF.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from prunex.engine import FeatureEngine


def main():
    """Run a full demonstration of PruneX."""
    print("=" * 60)
    print("  PruneX v1.0 - Demo")
    print("=" * 60)

    # 1. Load Data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # 2. Add challenging columns to demonstrate the pipeline
    X['constant_feature'] = 5                              # Zero variance
    X['near_duplicate'] = X['mean radius'] * 1.0001        # Highly correlated
    X['category_col'] = np.where(y == 1, 'Benign', 'Malignant')  # Categorical
    X['signup_date'] = pd.date_range('2020-01-01', periods=len(X)).astype(str)  # Datetime
    X.loc[:10, 'mean texture'] = np.nan                    # Missing values
    X.loc[:5, 'category_col'] = np.nan                     # Missing categorical

    print(f"\\nOriginal dataset: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Missing values: {X.isnull().sum().sum()}")

    # 3. Train/Test Split (CRITICAL for no data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Initialize Feature Engine
    engine = FeatureEngine(
        target_column='target',
        problem_type='classification',
        mode='balanced',           # Options: 'fast', 'balanced', 'thorough'
        variance_threshold=0.01,
        correlation_threshold=0.85,
        mi_threshold=0.01,
        rfe_n_features=None,       # Let RFECV auto-determine
        enable_evaluation=True,
        verbosity=1,               # 0=silent, 1=summary, 2=detailed
    )

    # 5. Fit on training data only
    engine.fit(X_train, y_train)

    # 6. Transform both train and test
    X_train_clean = engine.transform(X_train)
    X_test_clean = engine.transform(X_test)

    print(f"\\nFinal dataset: {X_train_clean.shape[1]} features selected")
    print(f"Selected features: {engine.get_selected_features()}")

    # 7. Verify no data leakage
    assert X_train_clean.shape[1] == X_test_clean.shape[1]
    assert list(X_train_clean.columns) == list(X_test_clean.columns)
    print("\\n[OK] Train/test column consistency verified (no data leakage)")

    # 8. Generate the professional reports (HTML and PDF)
    # The generate_pdf_report method will also call generate_html_report internally
    engine.reporter_.generate_pdf_report(
        html_filepath="demo_report_v2.html", 
        pdf_filepath="demo_report_v2.pdf"
    )
    print("\\n[REPORT] Open demo_report_v2.html or demo_report_v2.pdf to see the audit report.")

    # 9. Print console summary
    engine.print_summary()

    # 10. Save the fitted pipeline
    engine.save("fitted_pipeline.joblib")
    print("\\n[SAVED] Pipeline saved to fitted_pipeline.joblib")

    print("\\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
