"""
Bank Marketing Campaign Prediction using Random Forest
======================================================

Real-world application of Random Forest on bank marketing data.

Dataset: Bank Marketing (UCI ML Repository)
- 45,211 samples
- 17 features (age, job, marital status, education, etc.)
- Target: Will client subscribe to term deposit? (yes/no)
- Imbalanced: ~11% positive class

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_and_explore_data(filepath='bank-additional-full.csv'):
    """
    Load and explore the bank marketing dataset.
    
    Download from: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    Or Kaggle: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
    """
    print("=" * 70)
    print("LOADING BANK MARKETING DATASET")
    print("=" * 70)
    
    # Load data (semicolon separated)
    df = pd.read_csv(filepath, sep=';')
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    # Target distribution
    print(f"\nTarget Distribution:")
    print(df['y'].value_counts())
    print(f"\nSubscription Rate: {(df['y'] == 'yes').sum() / len(df) * 100:.2f}%")
    
    # Statistical summary
    print(f"\nNumerical Features Summary:")
    print(df.describe())
    
    return df


def visualize_data(df):
    """Visualize dataset characteristics."""
    print("\n" + "=" * 70)
    print("DATA VISUALIZATION")
    print("=" * 70)
    
    # Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df['y'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
    axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Subscribed to Term Deposit')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['No', 'Yes'], rotation=0)
    
    df['y'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.2f%%',
                                labels=['No', 'Yes'], colors=['skyblue', 'salmon'])
    axes[1].set_title('Target Distribution (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Age distribution by target
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    df[df['y'] == 'no']['age'].hist(bins=30, alpha=0.7, label='No', color='skyblue')
    df[df['y'] == 'yes']['age'].hist(bins=30, alpha=0.7, label='Yes', color='salmon')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution by Subscription')
    plt.legend()
    
    # Job distribution
    plt.subplot(1, 2, 2)
    job_counts = df['job'].value_counts().head(10)
    job_counts.plot(kind='barh', color='steelblue')
    plt.xlabel('Count')
    plt.title('Top 10 Job Categories')
    plt.tight_layout()
    plt.show()
    
    # Correlation with target for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 6))
    correlations = df[numerical_cols].corrwith(df['y'].map({'no': 0, 'yes': 1})).sort_values()
    correlations.plot(kind='barh', color='teal')
    plt.xlabel('Correlation with Subscription')
    plt.title('Feature Correlations with Target')
    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    """Preprocess the dataset."""
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    
    # Create a copy
    df_processed = df.copy()
    
    # Encode target variable
    df_processed['y'] = df_processed['y'].map({'no': 0, 'yes': 1})
    
    # Identify categorical and numerical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('y')  # Remove target
    
    print(f"\nCategorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = df_processed.drop('y', axis=1)
    y = df_processed['y']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Positive class: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    
    return X, y, label_encoders


def train_single_decision_tree(X_train, X_test, y_train, y_test):
    """Train a single Decision Tree for comparison."""
    print("\n" + "=" * 70)
    print("SINGLE DECISION TREE (Baseline)")
    print("=" * 70)
    
    dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)[:, 1]
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return dt, y_pred, y_pred_proba


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest classifier."""
    print("\n" + "=" * 70)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 70)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    
    return rf, y_pred, y_pred_proba


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8]
    }
    
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    
    print("\nPerforming Grid Search (this may take a while)...")
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best ROC-AUC Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def plot_feature_importance(model, feature_names, top_n=15):
    """Plot top N most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features - Random Forest', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(y_test, y_pred_dt, y_pred_rf):
    """Plot confusion matrices for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = [
        ('Decision Tree', y_pred_dt),
        ('Random Forest', y_pred_rf)
    ]
    
    for idx, (name, y_pred) in enumerate(models):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[idx].set_title(f'Confusion Matrix - {name}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, y_proba_dt, y_proba_rf):
    """Plot ROC curves for comparison."""
    plt.figure(figsize=(10, 6))
    
    models = [
        ('Decision Tree', y_proba_dt),
        ('Random Forest', y_proba_rf)
    ]
    
    for name, y_proba in models:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Bank Marketing Prediction', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_trees_vs_performance(X_train, X_test, y_train, y_test):
    """Plot performance vs number of trees."""
    n_estimators_range = range(10, 201, 10)
    train_scores = []
    test_scores = []
    
    print("\nAnalyzing performance vs number of trees...")
    for n in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n, max_depth=10, 
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_scores, marker='o', label='Training Score', linewidth=2)
    plt.plot(n_estimators_range, test_scores, marker='s', label='Test Score', linewidth=2)
    plt.xlabel('Number of Trees', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Random Forest Performance vs Number of Trees', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BANK MARKETING PREDICTION WITH RANDOM FOREST")
    print("=" * 70)
    
    print("\n⚠️  IMPORTANT: Download the dataset first!")
    print("1. Go to: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
    print("   OR: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing")
    print("2. Download bank-additional-full.csv")
    print("3. Place it in the same directory as this script\n")
    
    # Uncomment below when you have the dataset
    """
    # Load and explore
    df = load_and_explore_data('bank-additional-full.csv')
    visualize_data(df)
    
    # Preprocess
    X, y, label_encoders = preprocess_data(df)
    
    # Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Positive class in test: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # Train models
    dt, y_pred_dt, y_proba_dt = train_single_decision_tree(
        X_train, X_test, y_train, y_test
    )
    
    rf, y_pred_rf, y_proba_rf = train_random_forest(
        X_train, X_test, y_train, y_test
    )
    
    # Hyperparameter tuning (optional - takes time)
    # best_rf = hyperparameter_tuning(X_train, y_train)
    # y_pred_best = best_rf.predict(X_test)
    # y_proba_best = best_rf.predict_proba(X_test)[:, 1]
    
    # Visualizations
    plot_confusion_matrices(y_test, y_pred_dt, y_pred_rf)
    plot_roc_curves(y_test, y_proba_dt, y_proba_rf)
    plot_feature_importance(rf, X.columns, top_n=15)
    plot_trees_vs_performance(X_train, X_test, y_train, y_test)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    
    results = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_dt),
            accuracy_score(y_test, y_pred_rf)
        ],
        'Precision': [
            precision_score(y_test, y_pred_dt),
            precision_score(y_test, y_pred_rf)
        ],
        'Recall': [
            recall_score(y_test, y_pred_dt),
            recall_score(y_test, y_pred_rf)
        ],
        'F1-Score': [
            f1_score(y_test, y_pred_dt),
            f1_score(y_test, y_pred_rf)
        ],
        'ROC-AUC': [
            roc_auc_score(y_test, y_proba_dt),
            roc_auc_score(y_test, y_proba_rf)
        ]
    })
    
    print("\n", results.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ BANK MARKETING PREDICTION PROJECT COMPLETE!")
    print("=" * 70)
    print("\nKey Learnings:")
    print("1. Random Forest outperforms single Decision Tree")
    print("2. Ensemble methods reduce overfitting")
    print("3. Feature importance reveals key marketing factors")
    print("4. More trees generally improve performance (up to a point)")
    print("5. Class imbalance requires special handling")
    """
