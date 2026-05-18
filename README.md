# PruneX

PruneX is an advanced, deterministically-driven Python library designed for automated feature engineering and mathematically rigorous feature selection.

In real-world machine learning environments, datasets frequently contain hundreds or thousands of columns. Navigating this high dimensionality manually is prone to error and bias. PruneX solves this by providing a multi-stage, Scikit-Learn compatible mathematical funnel that autonomously selects only the features that positively impact model performance.

Crucially, this library resolves the "black box" problem of automated data pipelines by generating a comprehensive HTML Audit Report, detailing the exact mathematical reasoning behind every feature kept or dropped.

## Core Philosophy

1. **Deterministic and Mathematical:** Relies entirely on robust statistical techniques (Variance, Pearson/Spearman correlation, Information Theory, Recursive Feature Elimination) rather than non-deterministic or costly LLM-based agent swarms.
2. **Transparent "Audit Trail":** Never wonder why a feature disappeared. The Engine logs every action and compiles a visual report.
3. **Scikit-Learn Native:** Designed to slot perfectly into existing `sklearn.pipeline.Pipeline` architectures, complete with `fit()`, `transform()`, and `GridSearchCV` compatibility to prevent data leakage.
4. **End-to-End Execution:** Automatically handles missing values, encodes complex text/categorical variables, extracts temporal features, and reduces dimensionality in a single execution.

## Pipeline Architecture

PruneX processes high-dimensional data through a sequence of modular stages:

### Stage 1: Automated Feature Engineering
* **Datetime Expansion:** Detects temporal columns and extracts granular numerical representations (year, month, day, day-of-week, weekend flags).
* **Group Aggregation:** Autonomously detects ID-based columns and engineers aggregated statistics (mean, sum) to capture group-level behavior.

### Stage 2: Data Pre-Processing & Encoding
* **Secure Imputation:** Learns missing value distributions (mean, median) during `.fit()` and safely applies them during `.transform()`.
* **Target Encoding:** Converts high-cardinality categorical string columns into continuous numerical data by mapping them against the target variable.

### Stage 3: The Mathematical Selection Funnel

PruneX produces a statistically rigorous, reproducible feature set not through heuristics, manual guesswork, or non-deterministic LLM agents, but by forcing every feature through **four rigorous mathematical gates**. This specific sequence is engineered to iteratively strip away noise, resolve collinearity, and isolate the pure predictive signal. This systematic funnel stabilizes model coefficients, prevents overfitting, and maximizes generalization.

#### 1. Variance Filter (The Signal Verification Gate)
Before evaluating a feature against the target, the feature must first possess internal variance. A feature without variance contains no information. We define population variance as:

$$ \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 $$

**Why this produces a superior outcome:** Features where $\sigma^2 \approx 0$ are virtually constants. In Information Theory, a constant feature has zero Shannon Entropy ($H(X) = 0$). Mathematically, such a feature cannot improve the split criteria in decision trees (Gini/Entropy reduction is impossible) and causes singular matrix errors in linear regression gradient descent. By stripping these immediately, we reduce the dimensionality space and computational overhead without losing a single bit of predictive signal.

#### 2. Collinearity Filter (The Redundancy Gate)
Once we have features with variance, we must ensure they are independent. Linear models suffer from drastically inflated standard errors when predictor variables are highly correlated (Multicollinearity). The Collinearity Filter evaluates every pair of features using the Pearson Correlation Coefficient:

$$ r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}} $$

**Why this produces a superior outcome:** When $|r_{xy}| > \text{threshold}$ (e.g., $0.85$), the Engine flags the collinear pair. Instead of dropping one randomly, it computes the correlation of both features against the *target variable* $y$. The feature with the lower target correlation is dropped. This mathematical deterministic approach ensures that we resolve the collinearity (stabilizing our eventual model) while strictly preserving the feature that holds the strongest predictive relationship to the outcome.

#### 3. Mutual Information (The Non-Linear Gate)
While Pearson correlation perfectly captures linear relationships, it fails to detect non-linear dependencies. **Mutual Information (MI)** solves this by utilizing Information Theory to measure the true reduction in uncertainty about the target $Y$ given the feature $X$. It is defined using marginal and joint probability distributions:

$$ I(X; Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right) $$

**Why this produces a superior outcome:** The engine uses **Dynamic Thresholding** (e.g., pruning features that hold `< 5%` of the dataset's maximum information). This ensures that every feature passing this gate contains genuine, measurable predictive power relative to the specific dataset's information density. **It also acts as our first Target Leakage Guard**, immediately flagging any feature whose MI score approaches perfect entropy loss ($> 0.90$).

#### 4. Recursive Feature Elimination (The Synergy Gate)
The final stage is an aggressive, iterative pruning process. The previous three gates evaluated features in isolation or pairs. RFE evaluates them *together* to account for multi-feature synergy. The Engine trains an internal ensemble model (e.g., Random Forest), which minimizes Gini Impurity at every split.

The importance of a feature $X_m$ is determined by its **Mean Decrease in Impurity (MDI)** across all trees $t$ in the forest:

$$ \text{Importance}(X_m) = \frac{1}{N_T} \sum_{t} \sum_{v \in S_{X_m}} \frac{N_v}{N} \Delta i(v, t) $$

**Why this produces a superior outcome:** By recursively eliminating the lowest-ranked features, the model prevents "masking" effects where a feature looks weak initially but is highly predictive when combined with another. The Engine also monitors the Gini distribution—if a single feature begins consuming $>90\%$ of the model's split decisions, the Engine fires a **🚨 TARGET LEAKAGE WARNING 🚨** directly into the audit report.

## Installation

For local development or testing, install the package in editable mode from the repository root:

```bash
# Clone the repository
git clone https://github.com/your-username/feature-selector.git
cd feature-selector

# Install in editable mode with development dependencies
pip install -e .
```

*Note: The library will automatically handle headless browser dependencies (Playwright/Chromium) in the background the first time you compile a high-fidelity PDF report.*

## Quick Start Guide

The entire framework can be instantiated and run with a few lines of code.

```python
import pandas as pd
from prunex.engine import FeatureEngine
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
    mi_threshold="dynamic",     # Dynamic mode: automatically selects features above 5% of max MI
    rfe_n_features=None         # Automatically determine optimal RFE feature subset size
)

# 4. Fit the pipeline to training data
engine.fit(X_train, y_train)

# 5. Transform both train and test sets
X_train_clean = engine.transform(X_train)
X_test_clean = engine.transform(X_test)

# 6. Generate the Audit Report (HTML and high-fidelity PDF)
engine.generate_report(filepath="feature_audit_report.html")
engine.reporter_.generate_pdf_report(html_filepath="feature_audit_report.html", pdf_filepath="feature_audit_report.pdf")
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

## Working Examples

The repository includes two complete end-to-end working demonstrations showcasing the library's versatility:

1. **Classification Demo (`examples/demo_usage.py`):** Runs the pipeline on the classic breast cancer dataset with injected collinear and categorical features.
2. **Messy Real-World Regression Demo (`examples/demo_regression_usage.py`):** Showcases high-dimensional regression feature selection on the California Housing dataset. This advanced demo simulates messy real-world conditions:
   * **Stage 1 (Preprocessing & Expansion):** Automatic temporal feature extraction from timestamps, secure mean/median missing value imputation, and **Group Aggregation** (calculating user-level grouped averages for all numerical columns).
   * **Stage 2 (Encoding):** **Target Encoding** of high-cardinality neighborhood categoricals to handle categorical predictors in linear and non-linear selectors.
   * **Stage 3 & 4 (Selection Funnel):** Complete variance, collinearity, dynamic mutual information, and recursive feature elimination (RFE) pruning.
   * **Safety Guardrails:** Robust warning when `rfe_n_features` exceeds remaining features, and automatic target leakage identification.

To run either demo locally:
```bash
# Classification demo
python examples/demo_usage.py

# Messy real-world regression demo
python examples/demo_regression_usage.py
```

## The Audit Report

Calling `.generate_report("report.html")` produces a standalone HTML document containing:
* A summary count of features kept vs. dropped.
* A visual Bar Chart Funnel illustrating the reduction at each pipeline stage.
* A pre-filtering Correlation Heatmap to visualize dataset collinearity.
* A comprehensive Tabular Audit Trail detailing the exact mathematical reason a specific column was eliminated (e.g., *"[CorrelationSelector] Dropped: Correlated 0.92 with feature_X. Kept feature_X because it has higher correlation to target."*).

## Contributing

Contributions to mathematical optimization, expanding the suite of transformers, or improving computational efficiency for massive datasets are welcome. Please ensure all pull requests maintain Scikit-Learn compatibility and do not introduce data leakage.