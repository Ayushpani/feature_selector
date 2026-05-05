import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from feature_engine_pro.engine import FeatureEngine
import warnings
warnings.filterwarnings('ignore')

datasets = {
    'Breast Cancer': (load_breast_cancer, 'classification'),
    'Wine': (load_wine, 'classification'),
    'Diabetes': (load_diabetes, 'regression'),
    'Iris': (load_iris, 'classification')
}

print("Evaluating datasets...")

results = []

for name, (loader, prob_type) in datasets.items():
    print(f"\n--- {name} ---")
    data = loader(as_frame=True)
    X = data.data
    y = data.target
    
    # Add noisy columns to show how we clean them up
    np.random.seed(42)
    X['noise_1'] = np.random.randn(len(X))
    X['noise_2'] = np.random.randn(len(X))
    X['constant'] = 5.0
    if name == 'Breast Cancer':
        X['dup_mean_radius'] = X['mean radius'] * 1.0001
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline Model
    model = RandomForestClassifier(random_state=42) if prob_type == 'classification' else RandomForestRegressor(random_state=42)
    baseline_score = np.mean(cross_val_score(model, X_train.fillna(0), y_train, cv=5, scoring='accuracy' if prob_type=='classification' else 'r2'))
    
    # Engineered Model
    engine = FeatureEngine(problem_type=prob_type, mode='balanced', verbosity=0)
    try:
        engine.fit(X_train, y_train)
        X_train_clean = engine.transform(X_train)
        engineered_score = np.mean(cross_val_score(model, X_train_clean, y_train, cv=5, scoring='accuracy' if prob_type=='classification' else 'r2'))
        
        improvement = engineered_score - baseline_score
        
        results.append({
            'Dataset': name,
            'Original Features': X.shape[1],
            'Engineered Features': X_train_clean.shape[1],
            'Baseline Score': baseline_score,
            'Engineered Score': engineered_score,
            'Improvement': improvement
        })
        print(f"Original: {X.shape[1]} -> Clean: {X_train_clean.shape[1]}")
        print(f"Baseline: {baseline_score:.4f} -> Engineered: {engineered_score:.4f} (Diff: {improvement:+.4f})")
    except Exception as e:
        print(f"Failed: {e}")

print("\nSummary Table:")
df = pd.DataFrame(results)
print(df.to_string(index=False))
