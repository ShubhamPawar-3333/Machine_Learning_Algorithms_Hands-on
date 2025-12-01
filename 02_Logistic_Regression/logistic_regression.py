"""
Logistic Regression Implementation
===================================

This module provides implementation of Logistic Regression
for binary and multi-class classification.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegressionFromScratch:
    """
    Logistic Regression implementation from scratch using Gradient Descent.
    
    Uses sigmoid function: σ(z) = 1 / (1 + e^(-z))
    where z = X·θ
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        n_iterations : int, default=1000
            Number of iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Parameters:
        -----------
        z : array-like
            Linear combination of inputs
            
        Returns:
        --------
        sigmoid : array-like
            Sigmoid of z
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using Gradient Descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Apply sigmoid
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost (log loss)
            cost = self._compute_cost(y, y_predicted)
            self.cost_history.append(cost)
        
        return self
    
    def _compute_cost(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted probabilities
            
        Returns:
        --------
        cost : float
            Binary cross-entropy loss
        """
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        probabilities : array, shape (n_samples,)
            Predicted probabilities
        """
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        threshold : float, default=0.5
            Decision threshold
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (Log Loss)')
        plt.title('Cost Function over Iterations')
        plt.grid(True)
        plt.show()


def evaluate_classification(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate classification model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for ROC-AUC
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1-Score': f1_score(y_true, y_pred, average='binary')
    }
    
    if y_pred_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_decision_boundary(X, y, model, title='Decision Boundary'):
    """
    Plot decision boundary for 2D data.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Feature data (2 features only)
    y : array-like
        Labels
    model : fitted model
        Model with predict method
    title : str
        Plot title
    """
    # Create mesh
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar()
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn Logistic Regression")
    print("=" * 50)
    
    sklearn_model = LogisticRegression(random_state=42)
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    y_pred_proba_sklearn = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = evaluate_classification(y_test, y_pred_sklearn, y_pred_proba_sklearn)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Method 2: From scratch
    print("\n" + "=" * 50)
    print("Logistic Regression from Scratch")
    print("=" * 50)
    
    custom_model = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    custom_model.fit(X_train_scaled, y_train)
    y_pred_custom = custom_model.predict(X_test_scaled)
    y_pred_proba_custom = custom_model.predict_proba(X_test_scaled)
    
    metrics = evaluate_classification(y_test, y_pred_custom, y_pred_proba_custom)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualizations
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_test, y_pred_sklearn))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_sklearn, labels=['Class 0', 'Class 1'])
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba_sklearn)
    
    # Plot cost history
    custom_model.plot_cost_history()
    
    # Plot decision boundary
    plot_decision_boundary(X_test_scaled, y_test, sklearn_model, 
                          'Logistic Regression Decision Boundary')
