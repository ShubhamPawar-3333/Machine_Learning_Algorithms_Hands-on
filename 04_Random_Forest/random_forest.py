"""
Random Forest Implementation
============================

This module provides implementation of Random Forest
for both classification and regression.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class DecisionTreeForForest:
    """Simple Decision Tree for Random Forest."""
    
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
        return self
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_total_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return {'value': leaf_value}
        
        # Random feature selection
        feature_idxs = np.random.choice(n_total_features, self.n_features, replace=False)
        
        # Find best split among random features
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)
        
        if best_feature is None:
            return {'value': self._most_common_label(y)}
        
        # Create child nodes
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left,
            'right': right
        }
    
    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feature_idx in feature_idxs:
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
        parent_gini = self._gini_impurity(y)
        
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        gini_left = self._gini_impurity(y[left_idxs])
        gini_right = self._gini_impurity(y[right_idxs])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        
        return parent_gini - child_gini
    
    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if 'value' in node:
            return node['value']
        
        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])


class RandomForestFromScratch:
    """
    Random Forest Classifier implementation from scratch.
    
    Ensemble of decision trees using bagging and random feature selection.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True, random_state=None):
        """
        Initialize Random Forest.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, default=10
            Maximum depth of each tree
        min_samples_split : int, default=2
            Minimum samples required to split
        max_features : str or int, default='sqrt'
            Number of features to consider for best split
        bootstrap : bool, default=True
            Whether to use bootstrap samples
        random_state : int, optional
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        """
        Build a forest of trees.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.trees = []
        n_samples, n_features = X.shape
        
        # Determine number of features per tree
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features
        
        # Build each tree
        for _ in range(self.n_estimators):
            tree = DecisionTreeForForest(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=max_features
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                idxs = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
            else:
                X_sample, y_sample = X, y
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using majority voting.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(Counter(tree_predictions[:, i]).most_common(1)[0][0])
        
        return np.array(predictions)
    
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
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        n_samples = X.shape[0]
        n_classes = len(np.unique(tree_predictions))
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            counts = Counter(tree_predictions[:, i])
            for class_label, count in counts.items():
                probabilities[i, class_label] = count / self.n_estimators
        
        return probabilities


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from Random Forest.
    
    Parameters:
    -----------
    model : fitted RandomForestClassifier
        Trained model
    feature_names : list
        Feature names
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance - Random Forest')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


def plot_oob_error(n_estimators_range, X, y):
    """
    Plot Out-of-Bag error vs number of estimators.
    
    Parameters:
    -----------
    n_estimators_range : list
        Range of n_estimators to test
    X : array-like
        Features
    y : array-like
        Labels
    """
    oob_errors = []
    
    for n_estimators in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=42)
        rf.fit(X, y)
        oob_errors.append(1 - rf.oob_score_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, oob_errors, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('OOB Error Rate')
    plt.title('Out-of-Bag Error vs Number of Trees')
    plt.grid(True)
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
    print("Scikit-learn Random Forest")
    print("=" * 50)
    
    sklearn_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sklearn, target_names=iris.target_names))
    
    # Method 2: From scratch
    print("\n" + "=" * 50)
    print("Random Forest from Scratch")
    print("=" * 50)
    
    custom_model = RandomForestFromScratch(n_estimators=10, max_depth=5, random_state=42)
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
    
    # Visualizations
    plot_feature_importance(sklearn_model, iris.feature_names)
    
    # Plot OOB error
    plot_oob_error(range(10, 201, 10), X_train, y_train)
