"""
Principal Component Analysis (PCA) Implementation
=================================================

This module provides PCA implementation for dimensionality reduction.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PCAFromScratch:
    """
    PCA implementation from scratch using eigenvalue decomposition.
    """
    
    def __init__(self, n_components=2):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int
            Number of principal components
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        """
        Fit PCA model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        X = np.array(X)
        
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components].T
        
        # Calculate explained variance
        self.explained_variance = eigenvalues[:self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance
        
        return self
    
    def transform(self, X):
        """
        Transform data to principal components.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        X = np.array(X)
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : array-like
            Transformed data
            
        Returns:
        --------
        X_original : array
            Data in original space
        """
        return np.dot(X_transformed, self.components) + self.mean


def plot_explained_variance(pca, title='Explained Variance Ratio'):
    """
    Plot explained variance ratio.
    
    Parameters:
    -----------
    pca : fitted PCA model
        PCA model with explained_variance_ratio_ attribute
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual explained variance
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Explained Variance by Component')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_pca_2d(X_pca, y, title='PCA 2D Projection'):
    """
    Plot 2D PCA projection.
    
    Parameters:
    -----------
    X_pca : array-like, shape (n_samples, 2)
        PCA-transformed data
    y : array-like
        Labels
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                         alpha=0.6, edgecolors='black', s=50)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pca_3d(X_pca, y, title='PCA 3D Projection'):
    """
    Plot 3D PCA projection.
    
    Parameters:
    -----------
    X_pca : array-like, shape (n_samples, 3)
        PCA-transformed data
    y : array-like
        Labels
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                        c=y, cmap='viridis', alpha=0.6, s=50)
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    ax.set_title(title)
    
    plt.colorbar(scatter, label='Class', shrink=0.5)
    plt.tight_layout()
    plt.show()


def plot_component_loadings(pca, feature_names, n_components=2):
    """
    Plot PCA component loadings.
    
    Parameters:
    -----------
    pca : fitted PCA model
        PCA model
    feature_names : list
        Feature names
    n_components : int
        Number of components to plot
    """
    fig, axes = plt.subplots(1, n_components, figsize=(14, 5))
    
    if n_components == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        indices = np.argsort(np.abs(loadings))[::-1]
        
        ax.barh(range(len(loadings)), loadings[indices])
        ax.set_yticks(range(len(loadings)))
        ax.set_yticklabels([feature_names[j] for j in indices])
        ax.set_xlabel('Loading Value')
        ax.set_title(f'PC{i+1} Loadings')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


def find_optimal_components(X, variance_threshold=0.95):
    """
    Find optimal number of components for desired variance.
    
    Parameters:
    -----------
    X : array-like
        Data
    variance_threshold : float
        Desired cumulative variance (0-1)
        
    Returns:
    --------
    n_components : int
        Optimal number of components
    """
    pca = PCA()
    pca.fit(X)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    
    return n_components


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_digits
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Standardize features (IMPORTANT for PCA!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn PCA")
    print("=" * 50)
    
    pca_sklearn = PCA(n_components=2)
    X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)
    
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_pca_sklearn.shape}")
    print(f"\nExplained variance ratio: {pca_sklearn.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.sum(pca_sklearn.explained_variance_ratio_):.4f}")
    
    # Method 2: From scratch
    print("\n" + "=" * 50)
    print("PCA from Scratch")
    print("=" * 50)
    
    pca_custom = PCAFromScratch(n_components=2)
    X_pca_custom = pca_custom.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca_custom.explained_variance_ratio}")
    print(f"Cumulative variance: {np.sum(pca_custom.explained_variance_ratio):.4f}")
    
    # Find optimal number of components
    print("\n" + "=" * 50)
    print("Finding Optimal Components")
    print("=" * 50)
    
    optimal_n = find_optimal_components(X_scaled, variance_threshold=0.95)
    print(f"Components needed for 95% variance: {optimal_n}")
    
    # Full PCA for visualization
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Visualizations
    plot_explained_variance(pca_full, 'PCA Explained Variance - Iris Dataset')
    plot_pca_2d(X_pca_sklearn, y, 'Iris Dataset - 2D PCA Projection')
    
    # 3D projection
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    plot_pca_3d(X_pca_3d, y, 'Iris Dataset - 3D PCA Projection')
    
    # Component loadings
    plot_component_loadings(pca_sklearn, iris.feature_names, n_components=2)
    
    # Reconstruction
    print("\n" + "=" * 50)
    print("Reconstruction Error")
    print("=" * 50)
    
    X_reconstructed = pca_sklearn.inverse_transform(X_pca_sklearn)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"Mean Squared Reconstruction Error: {reconstruction_error:.6f}")
    
    # Example with higher dimensional data (MNIST digits)
    print("\n" + "=" * 50)
    print("PCA on MNIST Digits (64 dimensions)")
    print("=" * 50)
    
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target
    
    # Standardize
    X_digits_scaled = StandardScaler().fit_transform(X_digits)
    
    # Apply PCA
    pca_digits = PCA(n_components=0.95)  # Keep 95% variance
    X_digits_pca = pca_digits.fit_transform(X_digits_scaled)
    
    print(f"Original dimensions: {X_digits.shape[1]}")
    print(f"Reduced dimensions: {X_digits_pca.shape[1]}")
    print(f"Variance preserved: {np.sum(pca_digits.explained_variance_ratio_):.4f}")
    
    plot_explained_variance(pca_digits, 'PCA Explained Variance - MNIST Digits')
