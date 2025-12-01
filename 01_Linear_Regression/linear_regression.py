"""
Linear Regression Implementation
=================================

This module provides a clean implementation of Linear Regression
both from scratch and using scikit-learn.

Author: Your Name
Date: 2025-12-01
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


class LinearRegressionFromScratch:
    """
    Linear Regression implementation from scratch using Normal Equation.
    
    The model fits a linear equation: y = X·θ + ε
    where θ are the parameters (coefficients and intercept).
    """
    
    def __init__(self):
        """Initialize the Linear Regression model."""
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """
        Fit the linear regression model using Normal Equation.
        
        Normal Equation: θ = (X^T X)^-1 X^T y
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Add bias term (column of ones)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation: θ = (X^T X)^-1 X^T y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        # Extract intercept and coefficients
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return X.dot(self.coefficients) + self.intercept
    
    def score(self, X, y):
        """
        Calculate R² score.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class LinearRegressionGradientDescent:
    """
    Linear Regression using Gradient Descent optimization.
    
    This implementation uses iterative optimization instead of
    the closed-form Normal Equation.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the model.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        n_iterations : int, default=1000
            Number of iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None
        self.intercept = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Fit the model using Gradient Descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = self.predict(X)
            
            # Calculate gradients
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
            
            # Calculate cost (MSE)
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        X = np.array(X)
        return X.dot(self.coefficients) + self.intercept
    
    def plot_cost_history(self):
        """Plot the cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Cost Function over Iterations')
        plt.grid(True)
        plt.show()


def evaluate_model(y_true, y_pred):
    """
    Evaluate regression model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    metrics = {
        'R² Score': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    return metrics


def plot_predictions(y_true, y_pred, title='Actual vs Predicted'):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plot residuals.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn Linear Regression")
    print("=" * 50)
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred_sklearn)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Method 2: From scratch (Normal Equation)
    print("\n" + "=" * 50)
    print("Linear Regression from Scratch (Normal Equation)")
    print("=" * 50)
    
    custom_model = LinearRegressionFromScratch()
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred_custom)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Method 3: Gradient Descent
    print("\n" + "=" * 50)
    print("Linear Regression with Gradient Descent")
    print("=" * 50)
    
    gd_model = LinearRegressionGradientDescent(learning_rate=0.01, n_iterations=1000)
    gd_model.fit(X_train, y_train)
    y_pred_gd = gd_model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred_gd)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualizations
    plot_predictions(y_test, y_pred_sklearn, 'Scikit-learn: Actual vs Predicted')
    plot_residuals(y_test, y_pred_sklearn)
    gd_model.plot_cost_history()
