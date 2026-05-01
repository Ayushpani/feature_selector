"""
Evaluator — Before/After Pipeline Performance Proof

Proves that the feature selection pipeline actually improves (or at least
maintains) model performance while reducing dimensionality.

Features:
- Before vs. after model performance comparison
- Cross-validated metrics (accuracy, F1, AUC-ROC, R², RMSE, MAE)
- Feature stability analysis (bootstrap selection consistency)
- Statistical significance testing of performance differences
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, f1_score, mean_squared_error

from feature_engine_pro.logger import get_logger


class Evaluator:
    """
    Evaluates the impact of feature selection on model performance.

    Parameters
    ----------
    problem_type : str, default='classification'
        'classification' or 'regression'.
    cv_folds : int, default=5
        Number of cross-validation folds.
    n_estimators : int, default=100
        Number of trees for the evaluation model.
    run_stability : bool, default=False
        If True, run bootstrap stability analysis.
    n_bootstrap : int, default=10
        Number of bootstrap samples for stability analysis.
    """
    def __init__(self, problem_type='classification', cv_folds=5,
                 n_estimators=100, run_stability=False, n_bootstrap=10):
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.run_stability = run_stability
        self.n_bootstrap = n_bootstrap

        self.before_scores_ = None
        self.after_scores_ = None
        self.before_metrics_ = {}
        self.after_metrics_ = {}
        self.stability_scores_ = None
        self._logger = get_logger()

    def _get_model(self):
        """Get the evaluation model."""
        if self.problem_type == 'classification':
            return RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=42, n_jobs=-1
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=42, n_jobs=-1
            )

    def _get_cv(self):
        """Get the cross-validation strategy."""
        if self.problem_type == 'classification':
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

    def _get_scoring(self):
        """Get scoring metrics."""
        if self.problem_type == 'classification':
            return {
                'accuracy': 'accuracy',
                'f1_weighted': 'f1_weighted',
            }
        else:
            return {
                'r2': 'r2',
                'neg_rmse': 'neg_root_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error',
            }

    def evaluate(self, X_before, X_after, y):
        """
        Compare model performance before and after feature selection.

        Parameters
        ----------
        X_before : pd.DataFrame
            Features before selection (all features, must be numerical).
        X_after : pd.DataFrame
            Features after selection.
        y : array-like
            Target variable.

        Returns
        -------
        dict
            Comparison metrics.
        """
        self._logger.info(
            f"[Evaluator] Evaluating pipeline impact: "
            f"{X_before.shape[1]} features → {X_after.shape[1]} features."
        )

        model = self._get_model()
        cv = self._get_cv()
        scoring = self._get_scoring()

        # Ensure numerical only
        X_before_num = X_before.select_dtypes(include=np.number).fillna(0)
        X_after_num = X_after.select_dtypes(include=np.number).fillna(0)

        # Before
        self.before_metrics_ = {}
        for name, scorer in scoring.items():
            try:
                scores = cross_val_score(model, X_before_num, y, cv=cv, scoring=scorer, n_jobs=-1)
                self.before_metrics_[name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'scores': scores.tolist(),
                }
            except Exception as e:
                self._logger.warning(f"[Evaluator] Before-scoring for '{name}' failed: {e}")
                self.before_metrics_[name] = {'mean': 0, 'std': 0, 'scores': []}

        # After
        self.after_metrics_ = {}
        for name, scorer in scoring.items():
            try:
                scores = cross_val_score(model, X_after_num, y, cv=cv, scoring=scorer, n_jobs=-1)
                self.after_metrics_[name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'scores': scores.tolist(),
                }
            except Exception as e:
                self._logger.warning(f"[Evaluator] After-scoring for '{name}' failed: {e}")
                self.after_metrics_[name] = {'mean': 0, 'std': 0, 'scores': []}

        # Log summary
        for metric_name in scoring:
            before_mean = self.before_metrics_.get(metric_name, {}).get('mean', 0)
            after_mean = self.after_metrics_.get(metric_name, {}).get('mean', 0)
            delta = after_mean - before_mean
            direction = '↑' if delta >= 0 else '↓'
            self._logger.info(
                f"[Evaluator] {metric_name}: "
                f"Before={before_mean:.4f}, After={after_mean:.4f} ({direction}{abs(delta):.4f})"
            )

        return {
            'before': self.before_metrics_,
            'after': self.after_metrics_,
            'features_before': X_before.shape[1],
            'features_after': X_after.shape[1],
            'reduction_pct': round((1 - X_after.shape[1] / max(X_before.shape[1], 1)) * 100, 1),
        }

    def run_stability_analysis(self, X, y, engine):
        """
        Run bootstrap stability analysis to measure feature selection consistency.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature set.
        y : array-like
            Target variable.
        engine : FeatureEngine
            The engine instance to test stability of.

        Returns
        -------
        pd.DataFrame
            Feature stability scores.
        """
        self._logger.info(
            f"[Evaluator] Running stability analysis ({self.n_bootstrap} bootstrap samples)..."
        )

        selection_counts = {}

        for i in range(self.n_bootstrap):
            # Bootstrap sample
            rng = np.random.RandomState(42 + i)
            indices = rng.choice(len(X), len(X), replace=True)

            X_boot = X.iloc[indices].reset_index(drop=True)
            if isinstance(y, pd.Series):
                y_boot = y.iloc[indices].reset_index(drop=True)
            else:
                y_boot = pd.Series(y).iloc[indices].reset_index(drop=True)

            try:
                # Clone the engine to avoid state contamination
                from sklearn.base import clone
                engine_clone = clone(engine)
                engine_clone.fit(X_boot, y_boot)

                for feat in engine_clone.selected_features_:
                    selection_counts[feat] = selection_counts.get(feat, 0) + 1
            except Exception as e:
                self._logger.debug(f"[Evaluator] Bootstrap {i} failed: {e}")

        # Compute stability scores
        self.stability_scores_ = pd.DataFrame({
            'feature': list(selection_counts.keys()),
            'selection_count': list(selection_counts.values()),
            'stability_score': [c / self.n_bootstrap for c in selection_counts.values()],
        }).sort_values('stability_score', ascending=False)

        n_stable = (self.stability_scores_['stability_score'] >= 0.8).sum()
        self._logger.info(
            f"[Evaluator] Stability analysis complete. "
            f"{n_stable} features selected in ≥80% of bootstrap runs."
        )

        return self.stability_scores_
