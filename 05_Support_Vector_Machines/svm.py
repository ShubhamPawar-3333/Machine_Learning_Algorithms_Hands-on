"""
Support Vector Machine (SVM) Implementation
==========================================

This module provides implementation and examples of SVM.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


class SVMFromScratch:
    """
    Simple SVM implementation using gradient descent (for educational purposes).
    
    Note: This is a simplified version. For production, use scikit-learn's SVM.
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize SVM.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        lambda_param : float
            Regularization parameter
        n_iters : int
            Number of iterations
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        """
        Fit SVM classifier (binary classification, labels should be -1 and 1).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (-1 or 1)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        X = np.array(X)
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


def tune_svm_hyperparameters(X_train, y_train):
    """
    Tune SVM hyperparameters using Grid Search.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
        
    Returns:
    --------
    best_params : dict
        Best hyperparameters
    best_model : SVC
        Best model
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_estimator_


def plot_svm_decision_boundary(X, y, model, title='SVM Decision Boundary'):
    """
    Plot SVM decision boundary (for 2D data).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Feature data
    y : array-like
        Labels
    model : fitted SVM model
        Trained model
    title : str
        Plot title
    """
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='RdYlBu', 
                edgecolors='black', s=50)
    
    # Plot support vectors
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='k', 
                   label='Support Vectors')
        plt.legend()
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar()
    plt.show()


def compare_svm_kernels(X_train, X_test, y_train, y_test):
    """
    Compare different SVM kernels.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    results : dict
        Accuracy scores for each kernel
    """
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    results = {}
    
    for kernel in kernels:
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[kernel] = accuracy
    
    return results


def plot_kernel_comparison(results):
    """Plot comparison of SVM kernels."""
    plt.figure(figsize=(10, 6))
    kernels = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(kernels, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    plt.xlabel('Kernel Type', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Comparison of SVM Kernels', fontsize=14)
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
    from sklearn.datasets import make_classification, load_iris
    
    # Load dataset
    iris = load_iris()
    # Use only 2 classes for binary classification
    X = iris.data[iris.target != 2]
    y = iris.target[iris.target != 2]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling (CRITICAL for SVM!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn SVM (RBF Kernel)")
    print("=" * 50)
    
    sklearn_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print(f"Number of Support Vectors: {len(sklearn_model.support_vectors_)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sklearn))
    
    # Compare kernels
    print("\n" + "=" * 50)
    print("Comparing SVM Kernels")
    print("=" * 50)
    
    kernel_results = compare_svm_kernels(X_train_scaled, X_test_scaled, y_train, y_test)
    for kernel, accuracy in kernel_results.items():
        print(f"{kernel.capitalize()} Kernel: {accuracy:.4f}")
    
    # Hyperparameter tuning
    print("\n" + "=" * 50)
    print("Hyperparameter Tuning")
    print("=" * 50)
    
    best_params, best_model = tune_svm_hyperparameters(X_train_scaled, y_train)
    print(f"Best Parameters: {best_params}")
    
    y_pred_best = best_model.predict(X_test_scaled)
    print(f"Best Model Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    
    # Visualizations
    plot_kernel_comparison(kernel_results)
    
    # For 2D visualization
    X_2d = X[:, :2]
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y, test_size=0.2, random_state=42
    )
    
    scaler_2d = StandardScaler()
    X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler_2d.transform(X_test_2d)
    
    svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_2d.fit(X_train_2d_scaled, y_train_2d)
    
    plot_svm_decision_boundary(X_test_2d_scaled, y_test_2d, svm_2d, 
                               'SVM Decision Boundary (RBF Kernel)')
