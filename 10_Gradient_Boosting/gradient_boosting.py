"""
Gradient Boosting Implementation
================================

This module provides examples using XGBoost, LightGBM, and CatBoost.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Note: Install these libraries separately if needed
# pip install xgboost lightgbm catboost

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def compare_gradient_boosting_libraries(X_train, X_test, y_train, y_test, task='classification'):
    """
    Compare XGBoost, LightGBM, and CatBoost.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    task : str
        'classification' or 'regression'
        
    Returns:
    --------
    results : dict
        Performance metrics for each library
    """
    results = {}
    
    # Scikit-learn Gradient Boosting
    if task == 'classification':
        sklearn_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                                   max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        y_pred = sklearn_model.predict(X_test)
        results['Sklearn GB'] = accuracy_score(y_test, y_pred)
    else:
        sklearn_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                  max_depth=3, random_state=42)
        sklearn_model.fit(X_train, y_train)
        y_pred = sklearn_model.predict(X_test)
        results['Sklearn GB'] = r2_score(y_test, y_pred)
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        if task == 'classification':
            xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                         max_depth=3, random_state=42)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            results['XGBoost'] = accuracy_score(y_test, y_pred)
        else:
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=3, random_state=42)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            results['XGBoost'] = r2_score(y_test, y_pred)
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        if task == 'classification':
            lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                          max_depth=3, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            y_pred = lgb_model.predict(X_test)
            results['LightGBM'] = accuracy_score(y_test, y_pred)
        else:
            lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                                         max_depth=3, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            y_pred = lgb_model.predict(X_test)
            results['LightGBM'] = r2_score(y_test, y_pred)
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        if task == 'classification':
            cat_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1,
                                             depth=3, random_state=42, verbose=False)
            cat_model.fit(X_train, y_train)
            y_pred = cat_model.predict(X_test)
            results['CatBoost'] = accuracy_score(y_test, y_pred)
        else:
            cat_model = cb.CatBoostRegressor(iterations=100, learning_rate=0.1,
                                            depth=3, random_state=42, verbose=False)
            cat_model.fit(X_train, y_train)
            y_pred = cat_model.predict(X_test)
            results['CatBoost'] = r2_score(y_test, y_pred)
    
    return results


def plot_feature_importance_gb(model, feature_names, title='Feature Importance'):
    """
    Plot feature importance from gradient boosting model.
    
    Parameters:
    -----------
    model : fitted model
        Model with feature_importances_ attribute
    feature_names : list
        Feature names
    title : str
        Plot title
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


def plot_learning_curve_gb(model, X_train, y_train, X_test, y_test, metric='accuracy'):
    """
    Plot learning curve showing performance vs number of estimators.
    
    Parameters:
    -----------
    model : gradient boosting model
        Model with staged_predict method
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    metric : str
        'accuracy' or 'mse'
    """
    train_scores = []
    test_scores = []
    
    for y_pred_train, y_pred_test in zip(model.staged_predict(X_train),
                                          model.staged_predict(X_test)):
        if metric == 'accuracy':
            train_scores.append(accuracy_score(y_train, y_pred_train))
            test_scores.append(accuracy_score(y_test, y_pred_test))
        else:
            train_scores.append(mean_squared_error(y_train, y_pred_train))
            test_scores.append(mean_squared_error(y_test, y_pred_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='Training Score', linewidth=2)
    plt.plot(test_scores, label='Test Score', linewidth=2)
    plt.xlabel('Number of Estimators')
    plt.ylabel(metric.capitalize())
    plt.title('Learning Curve - Gradient Boosting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_library_comparison(results):
    """Plot comparison of gradient boosting libraries."""
    plt.figure(figsize=(10, 6))
    libraries = list(results.keys())
    scores = list(results.values())
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(libraries)]
    bars = plt.bar(libraries, scores, color=colors)
    
    plt.xlabel('Library', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Gradient Boosting Libraries Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_classification
    from sklearn.metrics import classification_report
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scikit-learn Gradient Boosting
    print("=" * 50)
    print("Scikit-learn Gradient Boosting")
    print("=" * 50)
    
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                         max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Compare libraries
    print("\n" + "=" * 50)
    print("Comparing Gradient Boosting Libraries")
    print("=" * 50)
    
    results = compare_gradient_boosting_libraries(X_train, X_test, y_train, y_test, 
                                                  task='classification')
    for library, score in results.items():
        print(f"{library}: {score:.4f}")
    
    # Visualizations
    plot_feature_importance_gb(gb_model, iris.feature_names, 
                              'Feature Importance - Gradient Boosting')
    
    plot_learning_curve_gb(gb_model, X_train, y_train, X_test, y_test, metric='accuracy')
    
    if len(results) > 1:
        plot_library_comparison(results)
    
    # XGBoost specific features (if available)
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 50)
        print("XGBoost Specific Features")
        print("=" * 50)
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        # Plot XGBoost tree
        try:
            fig, ax = plt.subplots(figsize=(20, 10))
            xgb.plot_tree(xgb_model, num_trees=0, ax=ax)
            plt.title('XGBoost Tree Visualization (First Tree)')
            plt.tight_layout()
            plt.show()
        except:
            print("Tree visualization requires graphviz")
        
        # Plot XGBoost feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(xgb_model, ax=ax, importance_type='weight')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.show()
