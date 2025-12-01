"""
K-Means Clustering Implementation
=================================

This module provides implementation of K-Means clustering algorithm.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class KMeansFromScratch:
    """
    K-Means Clustering implementation from scratch.
    
    Uses Lloyd's algorithm with K-Means++ initialization.
    """
    
    def __init__(self, n_clusters=3, max_iters=300, init='k-means++', random_state=None):
        """
        Initialize K-Means.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters
        max_iters : int, default=300
            Maximum number of iterations
        init : str, default='k-means++'
            Initialization method ('random' or 'k-means++')
        random_state : int, optional
            Random seed
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def fit(self, X):
        """
        Compute K-Means clustering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        if self.random_state:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        
        # Initialize centroids
        if self.init == 'k-means++':
            self.centroids = self._kmeans_plusplus_init(X)
        else:
            self.centroids = self._random_init(X)
        
        # Iterate until convergence
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)
            
            # Calculate new centroids
            new_centroids = self._calculate_centroids(X, labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        self.labels = self._assign_clusters(X)
        self.inertia_ = self._calculate_inertia(X, self.labels)
        
        return self
    
    def _random_init(self, X):
        """Randomly initialize centroids."""
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def _kmeans_plusplus_init(self, X):
        """Initialize centroids using K-Means++ algorithm."""
        centroids = []
        
        # Choose first centroid randomly
        centroids.append(X[np.random.randint(X.shape[0])])
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for idx, prob in enumerate(cumulative_probs):
                if r < prob:
                    centroids.append(X[idx])
                    break
        
        return np.array(centroids)
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid."""
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def _calculate_centroids(self, X, labels):
        """Calculate new centroids as mean of assigned points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[k] = X[np.random.randint(X.shape[0])]
        return centroids
    
    def _calculate_inertia(self, X, labels):
        """Calculate within-cluster sum of squares."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k])**2)
        return inertia
    
    def predict(self, X):
        """Predict cluster labels for samples."""
        X = np.array(X)
        return self._assign_clusters(X)


def elbow_method(X, k_range):
    """
    Use elbow method to find optimal number of clusters.
    
    Parameters:
    -----------
    X : array-like
        Data
    k_range : list
        Range of K values to test
        
    Returns:
    --------
    inertias : list
        Inertia values for each K
    """
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return inertias


def silhouette_method(X, k_range):
    """
    Use silhouette score to find optimal number of clusters.
    
    Parameters:
    -----------
    X : array-like
        Data
    k_range : list
        Range of K values to test
        
    Returns:
    --------
    silhouette_scores : list
        Silhouette scores for each K
    """
    silhouette_scores = []
    
    for k in k_range:
        if k < 2:
            silhouette_scores.append(0)
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    return silhouette_scores


def plot_elbow_curve(k_range, inertias):
    """Plot elbow curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (WCSS)', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.show()


def plot_silhouette_scores(k_range, scores):
    """Plot silhouette scores."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score for Different K', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    
    # Mark optimal K
    if len(scores) > 0:
        optimal_k = k_range[np.argmax(scores)]
        plt.axvline(x=optimal_k, color='r', linestyle='--', 
                    label=f'Optimal K = {optimal_k}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_clusters(X, labels, centroids=None, title='K-Means Clustering'):
    """
    Plot clusters (for 2D data).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Data (2 features)
    labels : array-like
        Cluster labels
    centroids : array-like, optional
        Cluster centroids
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='black', s=50)
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
                   s=200, edgecolors='black', linewidths=2, label='Centroids')
        plt.legend()
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                           cluster_std=0.60, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn K-Means")
    print("=" * 50)
    
    sklearn_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    sklearn_model.fit(X_scaled)
    labels_sklearn = sklearn_model.labels_
    
    print(f"Inertia: {sklearn_model.inertia_:.4f}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels_sklearn):.4f}")
    
    # Method 2: From scratch
    print("\n" + "=" * 50)
    print("K-Means from Scratch")
    print("=" * 50)
    
    custom_model = KMeansFromScratch(n_clusters=4, random_state=42)
    custom_model.fit(X_scaled)
    
    print(f"Inertia: {custom_model.inertia_:.4f}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, custom_model.labels):.4f}")
    
    # Find optimal K
    print("\n" + "=" * 50)
    print("Finding Optimal K")
    print("=" * 50)
    
    k_range = range(2, 11)
    inertias = elbow_method(X_scaled, k_range)
    silhouette_scores = silhouette_method(X_scaled, k_range)
    
    # Visualizations
    plot_elbow_curve(k_range, inertias)
    plot_silhouette_scores(k_range, silhouette_scores)
    plot_clusters(X_scaled, labels_sklearn, sklearn_model.cluster_centers_, 
                 'K-Means Clustering (K=4)')
