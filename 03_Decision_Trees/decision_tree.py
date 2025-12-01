"""
Decision Tree Implementation
============================

This module provides implementation of Decision Trees
for both classification and regression.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn import tree
import matplotlib.pyplot as plt


class Node:
    """Node class for Decision Tree."""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Value if leaf node


class DecisionTreeClassifierFromScratch:
    """
    Decision Tree Classifier implementation from scratch.
    
    Uses Gini impurity for splitting criterion.
    """
    
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        Initialize Decision Tree Classifier.
        
        Parameters:
        -----------
        max_depth : int, default=10
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        """
        Build decision tree classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels
        depth : int
            Current depth
            
        Returns:
        --------
        node : Node
            Decision tree node
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        # Create child nodes
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y, n_features):
        """
        Find the best split for a node.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels
        n_features : int
            Number of features
            
        Returns:
        --------
        best_feature : int
            Index of best feature
        best_threshold : float
            Best threshold value
        """
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain for a split.
        
        Parameters:
        -----------
        y : array-like
            Labels
        X_column : array-like
            Feature column
        threshold : float
            Split threshold
            
        Returns:
        --------
        gain : float
            Information gain
        """
        # Parent gini
        parent_gini = self._gini_impurity(y)
        
        # Create children
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        # Calculate weighted gini of children
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        gini_left = self._gini_impurity(y[left_idxs])
        gini_right = self._gini_impurity(y[right_idxs])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        
        # Information gain
        gain = parent_gini - child_gini
        return gain
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity.
        
        Parameters:
        -----------
        y : array-like
            Labels
            
        Returns:
        --------
        gini : float
            Gini impurity
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _most_common_label(self, y):
        """Return the most common label."""
        return np.bincount(y).argmax()
    
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
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse tree to make prediction.
        
        Parameters:
        -----------
        x : array-like
            Single sample
        node : Node
            Current node
            
        Returns:
        --------
        prediction : int
            Predicted class
        """
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def plot_tree_sklearn(model, feature_names=None, class_names=None):
    """
    Visualize decision tree.
    
    Parameters:
    -----------
    model : fitted DecisionTreeClassifier
        Trained model
    feature_names : list, optional
        Feature names
    class_names : list, optional
        Class names
    """
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, feature_names=feature_names, 
                   class_names=class_names, filled=True, rounded=True)
    plt.title('Decision Tree Visualization')
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    model : fitted model
        Model with feature_importances_ attribute
    feature_names : list
        Feature names
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_classification
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn Decision Tree")
    print("=" * 50)
    
    sklearn_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sklearn, target_names=iris.target_names))
    
    # Method 2: From scratch
    print("\n" + "=" * 50)
    print("Decision Tree from Scratch")
    print("=" * 50)
    
    custom_model = DecisionTreeClassifierFromScratch(max_depth=5)
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
    
    # Visualizations
    plot_tree_sklearn(sklearn_model, feature_names=iris.feature_names, 
                     class_names=iris.target_names)
    
    plot_feature_importance(sklearn_model, iris.feature_names)
