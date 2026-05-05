# Feature Engine Pro

Feature Engine Pro is an advanced, deterministically-driven Python library designed for automated feature engineering and mathematically rigorous feature selection.

In real-world machine learning environments, datasets frequently contain hundreds or thousands of columns. Navigating this high dimensionality manually is prone to error and bias. Feature Engine Pro solves this by providing a multi-stage, Scikit-Learn compatible mathematical funnel that autonomously selects only the features that positively impact model performance.

Crucially, this library resolves the "black box" problem of automated data pipelines by generating a comprehensive HTML Audit Report, detailing the exact mathematical reasoning behind every feature kept or dropped.

## Installation

You can install Feature Engine Pro directly from source:

```bash
pip install .
```

For PDF report generation support, install with the `pdf` extra:

```bash
pip install ".[pdf]"
playwright install chromium
```

## Core Philosophy

1. **Deterministic and Mathematical:** Relies entirely on robust statistical techniques (Variance, Pearson/Spearman correlation, Information Theory, Recursive Feature Elimination) rather than non-deterministic or costly LLM-based agent swarms.
2. **Transparent "Audit Trail":** Never wonder why a feature disappeared. The Engine logs every action and compiles a visual report.
3. **Scikit-Learn Native:** Designed to slot perfectly into existing `sklearn.pipeline.Pipeline` architectures, complete with `fit()`, `transform()`, and `GridSearchCV` compatibility to prevent data leakage.
4. **End-to-End Execution:** Automatically handles missing values, encodes complex text/categorical variables, extracts temporal features, and reduces dimensionality in a single execution.

## Pipeline Architecture

Feature Engine Pro processes high-dimensional data through a sequence of modular stages:

### Stage 1: Automated Feature Engineering
* **Datetime Expansion:** Detects temporal columns and extracts granular numerical representations (year, month, day, day-of-week, weekend flags).
* **Group Aggregation:** Autonomously detects ID-based columns and engineers aggregated statistics (mean, sum) to capture group-level behavior.

### Stage 2: Data Pre-Processing & Encoding
* **Secure Imputation:** Learns missing value distributions (mean, median) during `.fit()` and safely applies them during `.transform()`.
* **Target Encoding:** Converts high-cardinality categorical string columns into continuous numerical data by mapping them against the target variable.

### Stage 3: The Mathematical Selection Funnel
* **Variance Filter:** Eliminates zero-variance constants and low-variance features that carry no signal.
* **Collinearity Filter:** Identifies heavily correlated feature pairs. It evaluates both features against the target variable and intelligently drops the redundant feature providing the least predictive power.
* **Mutual Information:** Applies Information Theory to identify and preserve features with complex, non-linear dependencies on the target.
* **Recursive Feature Elimination (RFE):** Uses tree-based ensemble estimators (Random Forest) and feature importance ranking to iteratively prune the weakest remaining columns.

## Installation

*(Note: Package is currently in pre-release development phase)*

```bash
pip install feature-engine-pro
```

## Quick Start Guide

The entire framework can be instantiated and run with a few lines of code.

```python
import pandas as pd
from feature_engine_pro.engine import FeatureEngine
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv("high_dimensional_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 2. Split Data (Crucial for preventing data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize Feature Engine
engine = FeatureEngine(
    target_column="target",
    problem_type="classification",
    variance_threshold=0.01,
    correlation_threshold=0.85,
    mi_threshold=0.01,
    rfe_n_features=25
)

# 4. Fit the pipeline to training data
engine.fit(X_train, y_train)

# 5. Transform both train and test sets
X_train_clean = engine.transform(X_train)
X_test_clean = engine.transform(X_test)

# 6. Generate the Audit Report
engine.generate_report(filepath="feature_audit_report.html")
```

## Advanced Usage: GridSearchCV

Because `FeatureEngine` inherits from `BaseEstimator` and `TransformerMixin`, it natively supports hyperparameter tuning to find the optimal mathematical thresholds for your specific dataset.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

pipeline = Pipeline([
    ('feature_engine', FeatureEngine(problem_type='classification')),
    ('classifier', GradientBoostingClassifier())
])

param_grid = {
    'feature_engine__correlation_threshold': [0.75, 0.85, 0.95],
    'feature_engine__mi_threshold': [0.01, 0.05],
    'classifier__learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## The Audit Report

Calling `.generate_report("report.html")` produces a standalone HTML document containing:
* A summary count of features kept vs. dropped.
* A visual Bar Chart Funnel illustrating the reduction at each pipeline stage.
* A pre-filtering Correlation Heatmap to visualize dataset collinearity.
* A comprehensive Tabular Audit Trail detailing the exact mathematical reason a specific column was eliminated (e.g., *"[CorrelationSelector] Dropped: Correlated 0.92 with feature_X. Kept feature_X because it has higher correlation to target."*).

## Contributing

Contributions to mathematical optimization, expanding the suite of transformers, or improving computational efficiency for massive datasets are welcome. Please ensure all pull requests maintain Scikit-Learn compatibility and do not introduce data leakage.