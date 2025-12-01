"""
Ensemble Methods Implementation
===============================

This module provides implementation of various ensemble methods.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, 
                              AdaBoostClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def create_voting_ensemble(X_train, y_train, voting='hard'):
    """
    Create voting ensemble classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    voting : str
        'hard' or 'soft' voting
        
    Returns:
    --------
    voting_clf : VotingClassifier
        Fitted voting classifier
    """
    # Define base models
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = DecisionTreeClassifier(random_state=42)
    clf3 = KNeighborsClassifier()
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('dt', clf2), ('knn', clf3)],
        voting=voting
    )
    
    voting_clf.fit(X_train, y_train)
    return voting_clf


def create_bagging_ensemble(X_train, y_train, n_estimators=10):
    """
    Create bagging ensemble.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_estimators : int
        Number of base estimators
        
    Returns:
    --------
    bagging_clf : BaggingClassifier
        Fitted bagging classifier
    """
    base_clf = DecisionTreeClassifier(random_state=42)
    bagging_clf = BaggingClassifier(
        base_clf,
        n_estimators=n_estimators,
        max_samples=0.8,
        max_features=0.8,
        random_state=42
    )
    
    bagging_clf.fit(X_train, y_train)
    return bagging_clf


def create_boosting_ensemble(X_train, y_train, n_estimators=50):
    """
    Create AdaBoost ensemble.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_estimators : int
        Number of estimators
        
    Returns:
    --------
    boosting_clf : AdaBoostClassifier
        Fitted AdaBoost classifier
    """
    base_clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    boosting_clf = AdaBoostClassifier(
        base_clf,
        n_estimators=n_estimators,
        learning_rate=1.0,
        random_state=42
    )
    
    boosting_clf.fit(X_train, y_train)
    return boosting_clf


def create_stacking_ensemble(X_train, y_train):
    """
    Create stacking ensemble.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
        
    Returns:
    --------
    stacking_clf : StackingClassifier
        Fitted stacking classifier
    """
    # Base models (Level 0)
    estimators = [
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC(probability=True, random_state=42))
    ]
    
    # Meta model (Level 1)
    final_estimator = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5
    )
    
    stacking_clf.fit(X_train, y_train)
    return stacking_clf


def compare_ensemble_methods(X_train, X_test, y_train, y_test):
    """
    Compare different ensemble methods.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    results : dict
        Accuracy scores for each method
    """
    results = {}
    
    # Single Decision Tree (baseline)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    results['Decision Tree'] = accuracy_score(y_test, dt.predict(X_test))
    
    # Voting (Hard)
    voting_hard = create_voting_ensemble(X_train, y_train, voting='hard')
    results['Voting (Hard)'] = accuracy_score(y_test, voting_hard.predict(X_test))
    
    # Voting (Soft)
    voting_soft = create_voting_ensemble(X_train, y_train, voting='soft')
    results['Voting (Soft)'] = accuracy_score(y_test, voting_soft.predict(X_test))
    
    # Bagging
    bagging = create_bagging_ensemble(X_train, y_train, n_estimators=10)
    results['Bagging'] = accuracy_score(y_test, bagging.predict(X_test))
    
    # Boosting (AdaBoost)
    boosting = create_boosting_ensemble(X_train, y_train, n_estimators=50)
    results['AdaBoost'] = accuracy_score(y_test, boosting.predict(X_test))
    
    # Stacking
    stacking = create_stacking_ensemble(X_train, y_train)
    results['Stacking'] = accuracy_score(y_test, stacking.predict(X_test))
    
    return results


def plot_ensemble_comparison(results):
    """
    Plot comparison of ensemble methods.
    
    Parameters:
    -----------
    results : dict
        Accuracy scores for each method
    """
    plt.figure(figsize=(12, 6))
    methods = list(results.keys())
    accuracies = list(results.values())
    
    colors = ['lightcoral', 'skyblue', 'lightblue', 'lightgreen', 'lightyellow', 'plum']
    bars = plt.bar(methods, accuracies, color=colors[:len(methods)])
    
    plt.xlabel('Ensemble Method', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Comparison of Ensemble Methods', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_learning_curves_ensemble(X, y, models_dict):
    """
    Plot learning curves for different ensemble methods.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    models_dict : dict
        Dictionary of model names and models
    """
    plt.figure(figsize=(12, 6))
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    for name, model in models_dict.items():
        train_scores = []
        
        for train_size in train_sizes:
            n_samples = int(len(X) * train_size)
            X_subset = X[:n_samples]
            y_subset = y[:n_samples]
            
            scores = cross_val_score(model, X_subset, y_subset, cv=3, 
                                    scoring='accuracy')
            train_scores.append(scores.mean())
        
        plt.plot(train_sizes * 100, train_scores, marker='o', 
                label=name, linewidth=2)
    
    plt.xlabel('Training Set Size (%)', fontsize=12)
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.title('Learning Curves - Ensemble Methods', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_classification
    from sklearn.preprocessing import StandardScaler
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare ensemble methods
    print("=" * 50)
    print("Comparing Ensemble Methods")
    print("=" * 50)
    
    results = compare_ensemble_methods(X_train_scaled, X_test_scaled, y_train, y_test)
    
    for method, accuracy in results.items():
        print(f"{method}: {accuracy:.4f}")
    
    # Detailed example: Stacking
    print("\n" + "=" * 50)
    print("Stacking Ensemble Details")
    print("=" * 50)
    
    stacking_clf = create_stacking_ensemble(X_train_scaled, y_train)
    y_pred = stacking_clf.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Visualizations
    plot_ensemble_comparison(results)
    
    # Learning curves
    print("\n" + "=" * 50)
    print("Generating Learning Curves")
    print("=" * 50)
    
    models_dict = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Bagging': BaggingClassifier(DecisionTreeClassifier(random_state=42), 
                                     n_estimators=10, random_state=42),
        'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=42),
                                       n_estimators=50, random_state=42)
    }
    
    plot_learning_curves_ensemble(X_train_scaled, y_train, models_dict)
    
    print("\nEnsemble methods comparison complete!")
