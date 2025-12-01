"""
K-Nearest Neighbors (KNN) Implementation
========================================

This module provides implementation of KNN
for both classification and regression.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class KNNFromScratch:
    """
    K-Nearest Neighbors Classifier implementation from scratch.
    
    Uses Euclidean distance by default.
    """
    
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        Initialize KNN classifier.
        
        Parameters:
        -----------
        k : int, default=5
            Number of neighbors
        distance_metric : str, default='euclidean'
            Distance metric ('euclidean', 'manhattan', 'minkowski')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Store training data (lazy learning).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """
        Predict single sample.
        
        Parameters:
        -----------
        x : array-like
            Single sample
            
        Returns:
        --------
        prediction : int
            Predicted class
        """
        # Calculate distances
        distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority voting
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points.
        
        Parameters:
        -----------
        x1, x2 : array-like
            Points
            
        Returns:
        --------
        distance : float
            Distance between points
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            p = 3  # Minkowski parameter
            return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        probabilities : array
            Class probabilities
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(np.unique(self.y_train))
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            counts = Counter(k_nearest_labels)
            for class_label, count in counts.items():
                probabilities[i, class_label] = count / self.k
        
        return probabilities


def find_optimal_k(X_train, X_test, y_train, y_test, k_range):
    """
    Find optimal K value using validation set.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    k_range : list
        Range of K values to test
        
    Returns:
    --------
    optimal_k : int
        Best K value
    accuracies : list
        Accuracies for each K
    """
    accuracies = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    optimal_k = k_range[np.argmax(accuracies)]
    return optimal_k, accuracies


def plot_k_vs_accuracy(k_range, accuracies):
    """
    Plot K vs Accuracy.
    
    Parameters:
    -----------
    k_range : list
        Range of K values
    accuracies : list
        Corresponding accuracies
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('K vs Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    
    # Mark optimal K
    optimal_k = k_range[np.argmax(accuracies)]
    plt.axvline(x=optimal_k, color='r', linestyle='--', 
                label=f'Optimal K = {optimal_k}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_decision_boundary_knn(X, y, model, title='KNN Decision Boundary'):
    """
    Plot decision boundary for 2D data.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Feature data (2 features only)
    y : array-like
        Labels
    model : fitted model
        KNN model
    title : str
        Plot title
    """
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='viridis', 
                edgecolors='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar()
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
    
    # Feature scaling (IMPORTANT for KNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn KNN")
    print("=" * 50)
    
    sklearn_model = KNeighborsClassifier(n_neighbors=5)
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sklearn, target_names=iris.target_names))
    
    # Method 2: From scratch
    print("\n" + "=" * 50)
    print("KNN from Scratch")
    print("=" * 50)
    
    custom_model = KNNFromScratch(k=5)
    custom_model.fit(X_train_scaled, y_train)
    y_pred_custom = custom_model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
    
    # Find optimal K
    print("\n" + "=" * 50)
    print("Finding Optimal K")
    print("=" * 50)
    
    k_range = list(range(1, 31))
    optimal_k, accuracies = find_optimal_k(X_train_scaled, X_test_scaled, 
                                           y_train, y_test, k_range)
    
    print(f"Optimal K: {optimal_k}")
    print(f"Best Accuracy: {max(accuracies):.4f}")
    
    # Visualizations
    plot_k_vs_accuracy(k_range, accuracies)
    
    # For 2D visualization, use only 2 features
    X_2d = X[:, :2]
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y, test_size=0.2, random_state=42
    )
    
    scaler_2d = StandardScaler()
    X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler_2d.transform(X_test_2d)
    
    knn_2d = KNeighborsClassifier(n_neighbors=5)
    knn_2d.fit(X_train_2d_scaled, y_train_2d)
    
    plot_decision_boundary_knn(X_test_2d_scaled, y_test_2d, knn_2d, 
                               'KNN Decision Boundary (K=5)')
