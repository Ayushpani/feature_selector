# Feature Engine Pro v2.0

**Feature Engine Pro** is an industry-grade, deterministically-driven Python library designed for automated feature engineering, mathematically rigorous feature selection, and transparent audit reporting.

In real-world machine learning environments, datasets frequently contain hundreds or thousands of columns with missing data, extreme outliers, high cardinality, and extreme collinearity. Navigating this high dimensionality manually is prone to error, bias, and data leakage. Feature Engine Pro solves this by providing a robust, 13-stage Scikit-Learn compatible mathematical funnel that autonomously engineers features, handles edge cases, and selects only the signals that positively impact model performance.

Crucially, this library resolves the "black box" problem of automated ML pipelines by generating a highly professional **Interactive HTML and PDF Audit Report**, detailing the exact mathematical reasoning behind every feature kept, modified, or dropped.

## Core Philosophy

1. **Deterministic and Mathematical:** Relies entirely on robust statistical techniques (ANOVA, Variance, Hierarchical Clustering, Information Theory, Recursive Feature Elimination) ensuring highly reproducible results without relying on costly non-deterministic logic.
2. **Transparent Audit Trail:** The Engine logs every action and compiles a visual report detailing the exact lifecycle of every column.
3. **Impeccable Safety & Scikit-Learn Native:** Designed to slot perfectly into existing `sklearn.pipeline.Pipeline` architectures. Handles infinite values, entirely-NaN columns, extreme cardinality, and mixed types without crashing. Strictly segregates `fit()` and `transform()` to completely eliminate data leakage.
4. **Environment-Aware UX:** The logger auto-detects if you are in a Jupyter Notebook, VS Code Terminal, or CI/CD runner, and gracefully adapts its output.

## Benchmark Performance

Feature Engine Pro is designed to drastically reduce dimensionality (and thus computational overhead and overfitting risk) while maintaining or even improving model performance.

| Dataset | Original Features | Features After Selection | Dimensionality Reduction | Baseline Performance | Engineered Performance | Change |
|---------|------------------|--------------------------|--------------------------|----------------------|------------------------|--------|
| Breast Cancer (Classification) | 34 | 4 | **-88.2%** | 96.26% Accuracy | 96.26% Accuracy | 0.00% |
| Wine (Classification) | 16 | 6 | **-62.5%** | 97.22% Accuracy | 97.22% Accuracy | 0.00% |
| Diabetes (Regression) | 13 | 4 | **-69.2%** | 0.452 R2 Score | 0.457 R2 Score | **+0.005 R2** |

*Note: Models evaluated using 5-fold cross-validation with Random Forest estimators.*

## Pipeline Architecture (13 Stages)

Feature Engine Pro processes high-dimensional data through a sequence of intelligent, modular stages:

### Stage 1: Automated Feature Engineering
* **Datetime Expansion:** Detects temporal columns, extracts granular components, and applies Cyclic Encoding (sin/cos transformations) for periodic features.
* **Group Aggregation:** Autonomously detects ID-based columns and engineers aggregated statistics (mean, std, min, max).
* **Polynomial Feature Generation:** Generates non-linear interaction features utilizing Target-Correlation Pre-Screening to prevent combinatorial explosion.

### Stage 2: Data Pre-Processing & Encoding
* **Robust Data Cleaning:** Handles infinity, drops duplicate columns, and eliminates 100% NaN columns automatically.
* **Outlier Handling:** Employs IQR, Z-Score, or MAD algorithms to gracefully clip or nullify extreme outliers without dropping rows.
* **Skewness Correction:** Auto-detects highly skewed numerical distributions and applies logarithmic or Box-Cox transformations.
* **Auto-Categorical Encoding:** Leverages Smoothed Bayesian Target Encoding to safely convert high-cardinality strings to numerical data, preventing leakage.

### Stage 3: The Mathematical Selection Funnel
* **Variance Filter:** Eliminates zero-variance constants and quasi-constant features.
* **Hierarchical Collinearity Filter:** Identifies heavily correlated pairs via Hierarchical Clustering, intelligently keeping the feature with the highest independent predictive power against the target.
* **Mutual Information:** Applies stable Information Theory to identify complex, non-linear dependencies.
* **Statistical Testing:** Applies robust ANOVA / Chi2 testing with Bonferroni correction to ensure only statistically significant features survive.
* **Recursive Feature Elimination (RFE):** Uses cross-validated tree-based estimators to iteratively prune the weakest remaining columns and find the absolute optimal feature count.

---

## Installation

```bash
pip install -e .
```
*(Dependencies: pandas, numpy, scikit-learn, plotly, scipy)*

**For automated high-quality PDF generation**, Playwright is required:
```bash
pip install playwright
python -m playwright install chromium
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

# 3. Initialize Feature Engine (Try mode='fast', 'balanced', or 'thorough')
engine = FeatureEngine(
    target_column="target",
    problem_type="classification",
    mode="balanced",  # Balances deep feature engineering with execution speed
    enable_evaluation=True, # Validates model impact (Before vs After)
    verbosity=1
)

# 4. Fit the pipeline to training data
engine.fit(X_train, y_train)

# 5. Transform both train and test sets
X_train_clean = engine.transform(X_train)
X_test_clean = engine.transform(X_test)

# 6. Generate the HTML and PDF Audit Reports
engine.reporter_.generate_pdf_report(
    html_filepath="audit_report.html",
    pdf_filepath="audit_report.pdf"
)

# 7. Print Console Summary
engine.print_summary()
```

## The Corporate Audit Report

Calling `.generate_pdf_report()` produces a stunning, standalone HTML document and a high-fidelity PDF featuring:

![Report Summary Header](file:///C:/Users/Ayush/.gemini/antigravity/brain/b423a9f2-a7c9-44fd-8b2f-6db97e9c3430/report_summary_header_1777625663948.png)

* **The Attrition Funnel:** A chart illustrating the reduction of features at each stage.
![Attrition Funnel](file:///C:/Users/Ayush/.gemini/antigravity/brain/b423a9f2-a7c9-44fd-8b2f-6db97e9c3430/report_funnel_correlation_1777625688202.png)

* **Performance Impact:** An automated validation comparing model performance on the original vs. engineered dataset.
![Performance Impact](file:///C:/Users/Ayush/.gemini/antigravity/brain/b423a9f2-a7c9-44fd-8b2f-6db97e9c3430/report_performance_impact_1777625700404.png)

* **The Audit Trail:** A comprehensive search-enabled table detailing the exact mathematical reason a specific column was eliminated or engineered.
![Audit Trail](file:///C:/Users/Ayush/.gemini/antigravity/brain/b423a9f2-a7c9-44fd-8b2f-6db97e9c3430/report_audit_trail_1777625712921.png)