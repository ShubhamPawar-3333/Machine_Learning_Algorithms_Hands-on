# K-Means Clustering 🎨

## Overview
K-Means is an **unsupervised learning** algorithm used for clustering data into K groups. It partitions data into clusters where each data point belongs to the cluster with the nearest mean (centroid).

## How It Works

### Algorithm Steps
1. **Initialize**: Randomly select K centroids
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned points
4. **Repeat**: Steps 2-3 until convergence

### Convergence Criteria
- Centroids don't change significantly
- Assignments don't change
- Maximum iterations reached

## Mathematical Foundation

### Objective Function
Minimize within-cluster sum of squares (WCSS):
```
J = Σᵏᵢ₌₁ Σₓ∈Cᵢ ||x - μᵢ||²
```
Where:
- K = number of clusters
- Cᵢ = cluster i
- μᵢ = centroid of cluster i
- x = data point

### Distance Metric
Typically uses **Euclidean distance**:
```
d(x, μ) = √(Σ(xᵢ - μᵢ)²)
```

## Key Concepts

### Choosing K (Number of Clusters)

#### 1. Elbow Method
- Plot WCSS vs K
- Look for "elbow" point where decrease slows
- Choose K at the elbow

#### 2. Silhouette Score
- Measures how similar a point is to its cluster vs other clusters
- Range: [-1, 1], higher is better
- Formula: `s = (b - a) / max(a, b)`
  - a = average distance within cluster
  - b = average distance to nearest cluster

#### 3. Gap Statistic
- Compares WCSS with expected WCSS under null distribution
- Choose K where gap is largest

#### 4. Domain Knowledge
- Use business/domain understanding

## Advantages
✅ Simple and easy to understand  
✅ Fast and efficient (O(n×K×i×d))  
✅ Scales well to large datasets  
✅ Guaranteed to converge  
✅ Works well with spherical clusters  

## Disadvantages
❌ Must specify K beforehand  
❌ Sensitive to initial centroid placement  
❌ Assumes spherical clusters of similar size  
❌ Sensitive to outliers  
❌ Doesn't work well with non-convex shapes  
❌ Different runs may give different results  

## Initialization Methods

### 1. Random Initialization
- Randomly select K points as centroids
- **Problem**: May converge to local optimum

### 2. K-Means++ (Recommended)
- Smart initialization to spread out initial centroids
- Better convergence and results
- **Default in scikit-learn**

### 3. Multiple Runs
- Run algorithm multiple times with different initializations
- Choose best result (lowest WCSS)

## Hyperparameters

### Important Parameters
- **n_clusters (K)** - Number of clusters
- **init** - Initialization method ('k-means++' or 'random')
- **n_init** - Number of times to run with different initializations
- **max_iter** - Maximum iterations
- **tol** - Convergence tolerance
- **random_state** - For reproducibility

## Use Cases
- 👥 **Customer Segmentation** - Group customers by behavior
- 🛍️ **Market Basket Analysis** - Product grouping
- 🖼️ **Image Compression** - Reduce colors in images
- 📝 **Document Clustering** - Group similar documents
- 🧬 **Gene Sequence Analysis** - Bioinformatics
- 🌍 **Geographic Clustering** - Location-based grouping
- 🎵 **Music Categorization** - Group similar songs
- 📊 **Anomaly Detection** - Identify outliers

## Evaluation Metrics

### Internal Metrics (No ground truth needed)
- **Inertia (WCSS)** - Within-cluster sum of squares
- **Silhouette Score** - Cluster cohesion and separation
- **Davies-Bouldin Index** - Average similarity between clusters
- **Calinski-Harabasz Index** - Ratio of between/within cluster dispersion

### External Metrics (With ground truth)
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**
- **Fowlkes-Mallows Score**

## Real-World Datasets
1. **Customer Data** - RFM analysis, segmentation
2. **Iris Dataset** - Classic clustering example
3. **Mall Customers** - Customer segmentation
4. **Wine Dataset** - Wine variety clustering
5. **Credit Card Data** - User behavior clustering
6. **Image Datasets** - Color quantization

## Handling Outliers

### Problems
- Outliers can significantly affect centroids
- May create their own clusters

### Solutions
1. **Remove outliers** before clustering
2. **Use DBSCAN** instead (density-based clustering)
3. **Use K-Medoids** (more robust to outliers)

## Feature Scaling
⚠️ **IMPORTANT**: K-Means is sensitive to feature scaling!

Always scale features before clustering:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Variants of K-Means

### 1. Mini-Batch K-Means
- Uses mini-batches for faster computation
- Good for very large datasets
- Slightly less accurate but much faster

### 2. K-Medoids (PAM)
- Uses actual data points as centroids
- More robust to outliers

### 3. Fuzzy C-Means
- Soft clustering (points can belong to multiple clusters)
- Provides membership probabilities

## Comparison with Other Clustering Algorithms

| Aspect | K-Means | DBSCAN | Hierarchical |
|--------|---------|--------|--------------|
| Speed | Fast | Medium | Slow |
| Shape | Spherical | Any | Any |
| Outliers | Sensitive | Robust | Sensitive |
| K Required | Yes | No | No |
| Scalability | Excellent | Good | Poor |

## Files in This Folder
- `kmeans_tutorial.ipynb` - Interactive Jupyter notebook
- `kmeans.py` - Python implementation from scratch
- `datasets/` - Sample datasets for practice

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [StatQuest: K-Means](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [K-Means++: The Advantages of Careful Seeding](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
- [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

## Next Steps
1. Open `kmeans_tutorial.ipynb` in Jupyter
2. Practice finding optimal K using elbow method
3. Visualize clusters in 2D/3D
4. Compare K-Means with K-Means++
5. Try customer segmentation project
6. Experiment with image compression

---
**Happy Learning! 🚀**
