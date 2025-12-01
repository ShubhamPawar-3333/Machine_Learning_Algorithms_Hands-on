"""
Credit Card Fraud Detection using Decision Trees
================================================

Real-world application of Decision Trees on highly imbalanced dataset.

Dataset: Credit Card Fraud Detection (Kaggle)
- 284,807 transactions
- Only 0.172% are fraudulent
- 30 features (anonymized for privacy)

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


def load_and_explore_data(filepath='creditcard.csv'):
    """
    Load and explore the credit card fraud dataset.
    
    Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    print("=" * 60)
    print("LOADING CREDIT CARD FRAUD DETECTION DATASET")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum().sum())
    
    # Class distribution
    print(f"\nClass Distribution:")
    print(df['Class'].value_counts())
    print(f"\nFraud Percentage: {df['Class'].sum() / len(df) * 100:.4f}%")
    
    return df


def visualize_data(df):
    """Visualize dataset characteristics."""
    print("\n" + "=" * 60)
    print("DATA VISUALIZATION")
    print("=" * 60)
    
    # Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    df['Class'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class (0: Normal, 1: Fraud)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['Normal', 'Fraud'], rotation=0)
    
    # Pie chart
    df['Class'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.4f%%',
                                    labels=['Normal', 'Fraud'], colors=['skyblue', 'salmon'])
    axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Transaction amount distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Amount distribution for normal transactions
    axes[0].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, color='skyblue', label='Normal')
    axes[0].set_xlabel('Transaction Amount')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Normal Transactions Amount Distribution')
    axes[0].set_xlim([0, 500])
    
    # Amount distribution for fraudulent transactions
    axes[1].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, color='salmon', label='Fraud')
    axes[1].set_xlabel('Transaction Amount')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Fraudulent Transactions Amount Distribution')
    axes[1].set_xlim([0, 500])
    
    plt.tight_layout()
    plt.show()
    
    # Time distribution
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df[df['Class'] == 0]['Time'], df[df['Class'] == 0]['Amount'], 
                alpha=0.3, s=1, label='Normal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amount')
    plt.title('Normal Transactions Over Time')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(df[df['Class'] == 1]['Time'], df[df['Class'] == 1]['Amount'], 
                alpha=0.7, s=10, color='red', label='Fraud')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amount')
    plt.title('Fraudulent Transactions Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    """Preprocess the dataset."""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Amount and Time features
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def train_baseline_model(X_train, X_test, y_train, y_test):
    """Train baseline Decision Tree without handling imbalance."""
    print("\n" + "=" * 60)
    print("BASELINE MODEL (No Imbalance Handling)")
    print("=" * 60)
    
    # Train model
    dt_baseline = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_baseline.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_baseline.predict(X_test)
    y_pred_proba = dt_baseline.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    return dt_baseline, y_pred, y_pred_proba


def train_with_class_weight(X_train, X_test, y_train, y_test):
    """Train Decision Tree with class weights."""
    print("\n" + "=" * 60)
    print("MODEL WITH CLASS WEIGHTS")
    print("=" * 60)
    
    # Train model with balanced class weights
    dt_weighted = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    dt_weighted.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_weighted.predict(X_test)
    y_pred_proba = dt_weighted.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    return dt_weighted, y_pred, y_pred_proba


def train_with_smote(X_train, X_test, y_train, y_test):
    """Train Decision Tree with SMOTE oversampling."""
    print("\n" + "=" * 60)
    print("MODEL WITH SMOTE (Oversampling)")
    print("=" * 60)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"\nOriginal training set: {X_train.shape}")
    print(f"After SMOTE: {X_train_smote.shape}")
    print(f"Class distribution after SMOTE:")
    print(pd.Series(y_train_smote).value_counts())
    
    # Train model
    dt_smote = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_smote.fit(X_train_smote, y_train_smote)
    
    # Predictions
    y_pred = dt_smote.predict(X_test)
    y_pred_proba = dt_smote.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    return dt_smote, y_pred, y_pred_proba


def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_weighted, y_pred_smote):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = [
        ('Baseline', y_pred_baseline),
        ('Class Weighted', y_pred_weighted),
        ('SMOTE', y_pred_smote)
    ]
    
    for idx, (name, y_pred) in enumerate(models):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        axes[idx].set_title(f'Confusion Matrix - {name}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, y_proba_baseline, y_proba_weighted, y_proba_smote):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 6))
    
    models = [
        ('Baseline', y_proba_baseline),
        ('Class Weighted', y_proba_weighted),
        ('SMOTE', y_proba_smote)
    ]
    
    for name, y_proba in models:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Fraud Detection Models', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15):
    """Plot top N most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CREDIT CARD FRAUD DETECTION WITH DECISION TREES")
    print("=" * 60)
    
    # NOTE: Download dataset from Kaggle first
    # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    print("\n⚠️  IMPORTANT: Download the dataset first!")
    print("1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("2. Download creditcard.csv")
    print("3. Place it in the same directory as this script\n")
    
    # Uncomment below when you have the dataset
    """
    # Load and explore
    df = load_and_explore_data('creditcard.csv')
    visualize_data(df)
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Fraud in test set: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # Train models
    dt_baseline, y_pred_baseline, y_proba_baseline = train_baseline_model(
        X_train, X_test, y_train, y_test
    )
    
    dt_weighted, y_pred_weighted, y_proba_weighted = train_with_class_weight(
        X_train, X_test, y_train, y_test
    )
    
    dt_smote, y_pred_smote, y_proba_smote = train_with_smote(
        X_train, X_test, y_train, y_test
    )
    
    # Visualizations
    plot_confusion_matrices(y_test, y_pred_baseline, y_pred_weighted, y_pred_smote)
    plot_roc_curves(y_test, y_proba_baseline, y_proba_weighted, y_proba_smote)
    plot_feature_importance(dt_smote, X.columns, top_n=15)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    
    results = pd.DataFrame({
        'Model': ['Baseline', 'Class Weighted', 'SMOTE'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_baseline),
            accuracy_score(y_test, y_pred_weighted),
            accuracy_score(y_test, y_pred_smote)
        ],
        'Precision': [
            precision_score(y_test, y_pred_baseline),
            precision_score(y_test, y_pred_weighted),
            precision_score(y_test, y_pred_smote)
        ],
        'Recall': [
            recall_score(y_test, y_pred_baseline),
            recall_score(y_test, y_pred_weighted),
            recall_score(y_test, y_pred_smote)
        ],
        'F1-Score': [
            f1_score(y_test, y_pred_baseline),
            f1_score(y_test, y_pred_weighted),
            f1_score(y_test, y_pred_smote)
        ],
        'ROC-AUC': [
            roc_auc_score(y_test, y_proba_baseline),
            roc_auc_score(y_test, y_proba_weighted),
            roc_auc_score(y_test, y_proba_smote)
        ]
    })
    
    print("\n", results.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ FRAUD DETECTION PROJECT COMPLETE!")
    print("=" * 60)
    print("\nKey Learnings:")
    print("1. Imbalanced data requires special handling")
    print("2. Accuracy alone is misleading - focus on Recall and F1")
    print("3. SMOTE or class weights improve fraud detection")
    print("4. Decision Trees can identify fraud patterns")
    print("5. Feature importance reveals key fraud indicators")
    """
