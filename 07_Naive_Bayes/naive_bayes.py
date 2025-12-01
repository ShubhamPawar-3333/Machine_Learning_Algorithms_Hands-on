"""
Naive Bayes Implementation
==========================

This module provides implementation of Naive Bayes classifiers.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


class GaussianNaiveBayesFromScratch:
    """
    Gaussian Naive Bayes implementation from scratch.
    
    Assumes features follow a Gaussian (normal) distribution.
    """
    
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
        
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        
        # Calculate mean, variance, and prior for each class
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
        
        return self
    
    def _calculate_likelihood(self, class_idx, x):
        """
        Calculate likelihood using Gaussian PDF.
        
        Parameters:
        -----------
        class_idx : int
            Class index
        x : array-like
            Sample
            
        Returns:
        --------
        likelihood : float
            Likelihood of sample given class
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        
        # Gaussian PDF
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        # Product of probabilities (assuming independence)
        likelihood = np.prod(numerator / denominator)
        return likelihood
    
    def _calculate_posterior(self, x):
        """
        Calculate posterior probability for each class.
        
        Parameters:
        -----------
        x : array-like
            Sample
            
        Returns:
        --------
        posteriors : dict
            Posterior probabilities for each class
        """
        posteriors = {}
        
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._calculate_likelihood(c, x) + 1e-10))
            posteriors[c] = prior + likelihood
        
        return posteriors
    
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
        predictions = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            predictions.append(max(posteriors, key=posteriors.get))
        
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
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            posteriors = self._calculate_posterior(x)
            
            # Convert log probabilities to probabilities
            max_log_prob = max(posteriors.values())
            exp_probs = {c: np.exp(posteriors[c] - max_log_prob) for c in self.classes}
            total = sum(exp_probs.values())
            
            for j, c in enumerate(self.classes):
                probabilities[i, j] = exp_probs[c] / total
        
        return probabilities


def compare_naive_bayes_types(X_train, X_test, y_train, y_test):
    """
    Compare different types of Naive Bayes classifiers.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
        
    Returns:
    --------
    results : dict
        Accuracy scores for each type
    """
    results = {}
    
    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    results['Gaussian'] = accuracy_score(y_test, y_pred_gnb)
    
    # Multinomial Naive Bayes (requires non-negative features)
    if np.all(X_train >= 0) and np.all(X_test >= 0):
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        y_pred_mnb = mnb.predict(X_test)
        results['Multinomial'] = accuracy_score(y_test, y_pred_mnb)
    
    # Bernoulli Naive Bayes
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred_bnb = bnb.predict(X_test)
    results['Bernoulli'] = accuracy_score(y_test, y_pred_bnb)
    
    return results


def plot_naive_bayes_comparison(results):
    """
    Plot comparison of Naive Bayes types.
    
    Parameters:
    -----------
    results : dict
        Accuracy scores for each type
    """
    plt.figure(figsize=(10, 6))
    types = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(types, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'][:len(types)])
    plt.xlabel('Naive Bayes Type', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Comparison of Naive Bayes Classifiers', fontsize=14)
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn Gaussian Naive Bayes")
    print("=" * 50)
    
    sklearn_model = GaussianNB()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sklearn, target_names=iris.target_names))
    
    # Method 2: From scratch
    print("\n" + "=" * 50)
    print("Gaussian Naive Bayes from Scratch")
    print("=" * 50)
    
    custom_model = GaussianNaiveBayesFromScratch()
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
    
    # Compare different types
    print("\n" + "=" * 50)
    print("Comparing Naive Bayes Types")
    print("=" * 50)
    
    results = compare_naive_bayes_types(X_train, X_test, y_train, y_test)
    for nb_type, accuracy in results.items():
        print(f"{nb_type} Naive Bayes: {accuracy:.4f}")
    
    # Visualization
    plot_naive_bayes_comparison(results)
