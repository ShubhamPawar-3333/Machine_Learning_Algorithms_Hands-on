# Principal Component Analysis (PCA) 📊

## Overview
Principal Component Analysis is an **unsupervised learning** technique used for **dimensionality reduction**. It transforms high-dimensional data into a lower-dimensional space while preserving as much variance (information) as possible.

## What is Dimensionality Reduction?

### Problem
- High-dimensional data (many features)
- Computational complexity
- Curse of dimensionality
- Visualization difficulty
- Redundant/correlated features

### Solution
- Reduce number of features
- Keep most important information
- Improve model performance
- Enable visualization

## How PCA Works

### Algorithm Steps
1. **Standardize** the data (mean=0, std=1)
2. **Compute covariance matrix**
3. **Calculate eigenvectors and eigenvalues**
4. **Sort eigenvectors** by eigenvalues (descending)
5. **Select top K eigenvectors** (principal components)
6. **Transform data** to new K-dimensional space

### Mathematical Foundation

#### Covariance Matrix
```
Cov(X) = (1/n) × XᵀX
```

#### Eigenvalue Decomposition
```
Cov(X) × v = λ × v
```
Where:
- v = eigenvector (principal component direction)
- λ = eigenvalue (variance explained)

#### Transformation
```
Z = X × W
```
Where:
- X = original data (n × d)
- W = selected eigenvectors (d × k)
- Z = transformed data (n × k)

## Key Concepts

### Principal Components
- **PC1**: Direction of maximum variance
- **PC2**: Direction of second-most variance (orthogonal to PC1)
- **PC3**: Third-most variance (orthogonal to PC1 & PC2)
- And so on...

### Variance Explained
- Each PC explains a portion of total variance
- **Cumulative variance**: Sum of variance explained by selected PCs
- Typically aim for 95% cumulative variance

### Choosing Number of Components

#### 1. Explained Variance Ratio
- Plot cumulative variance vs number of components
- Choose K where cumulative variance ≥ 95%

#### 2. Scree Plot
- Plot eigenvalues vs component number
- Look for "elbow" point

#### 3. Kaiser Criterion
- Keep components with eigenvalue > 1

#### 4. Domain Knowledge
- Based on application requirements

## Advantages
✅ Reduces dimensionality  
✅ Removes correlated features  
✅ Improves model performance  
✅ Speeds up training  
✅ Enables visualization (2D/3D)  
✅ Reduces noise  
✅ Prevents overfitting  

## Disadvantages
❌ Loss of interpretability (PCs are combinations)  
❌ Assumes linear relationships  
❌ Sensitive to feature scaling  
❌ May lose some information  
❌ Computationally expensive for very large datasets  

## Use Cases
- 📊 **Data Visualization** - Reduce to 2D/3D for plotting
- 🖼️ **Image Compression** - Reduce image dimensions
- 🧬 **Gene Expression Analysis** - Bioinformatics
- 📝 **Text Mining** - Latent Semantic Analysis
- 🎵 **Audio Processing** - Feature extraction
- 📈 **Financial Analysis** - Portfolio optimization
- 🔍 **Anomaly Detection** - Outlier identification
- 🎯 **Feature Engineering** - Create new features

## Evaluation

### Metrics
- **Explained Variance Ratio** - Variance preserved
- **Reconstruction Error** - Information lost
- **Downstream Task Performance** - Model accuracy after PCA

## Real-World Datasets
1. **Iris Dataset** - Visualize 4D data in 2D
2. **MNIST Digits** - Reduce 784 dimensions
3. **Wine Dataset** - Feature reduction
4. **Breast Cancer** - 30 features to fewer
5. **Gene Expression** - Thousands of genes
6. **Face Recognition** - Eigenfaces

## Feature Scaling
⚠️ **CRITICAL**: PCA is highly sensitive to feature scaling!

Always standardize before PCA:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## PCA vs Other Techniques

### Linear Dimensionality Reduction
- **PCA** - Unsupervised, maximizes variance
- **LDA** - Supervised, maximizes class separation
- **ICA** - Independent Component Analysis

### Non-Linear Dimensionality Reduction
- **t-SNE** - Great for visualization, slow
- **UMAP** - Faster than t-SNE, preserves global structure
- **Kernel PCA** - Non-linear version of PCA
- **Autoencoders** - Neural network-based

## Comparison Table

| Aspect | PCA | t-SNE | UMAP |
|--------|-----|-------|------|
| Speed | Fast | Slow | Fast |
| Scalability | Excellent | Poor | Good |
| Interpretability | Medium | Low | Low |
| Global Structure | Preserves | Loses | Preserves |
| Local Structure | Loses | Preserves | Preserves |
| Use Case | General | Visualization | Visualization |

## Inverse Transform
PCA can reconstruct original data (with some loss):
```python
X_reconstructed = pca.inverse_transform(X_pca)
```

### Applications
- **Image Compression**: Compress and decompress
- **Denoising**: Remove noise by dropping low-variance components
- **Anomaly Detection**: High reconstruction error = anomaly

## Kernel PCA
- Non-linear extension of PCA
- Uses kernel trick (like SVM)
- Can capture non-linear patterns

### Kernels
- RBF (Gaussian)
- Polynomial
- Sigmoid

## Incremental PCA
- For datasets too large for memory
- Processes data in mini-batches
- Useful for online learning

## Sparse PCA
- Produces sparse components
- Easier to interpret
- Uses L1 regularization

## Best Practices

### 1. Preprocessing
- **Always standardize** features
- Handle missing values
- Remove outliers (optional)

### 2. Choosing Components
- Plot explained variance
- Use cross-validation
- Consider downstream task

### 3. Interpretation
- Examine component loadings
- Visualize in 2D/3D
- Check reconstruction error

### 4. Validation
- Test on held-out data
- Monitor model performance
- Compare with/without PCA

## Common Mistakes

### ❌ Don't Do This
1. Forget to standardize
2. Apply PCA before train/test split
3. Use PCA on categorical data
4. Assume PCA always helps

### ✅ Do This
1. Standardize first
2. Fit PCA on training data only
3. Encode categorical variables first
4. Validate performance

## Visualization Examples

### 2D Projection
```python
# Reduce to 2 components
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Plot
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
```

### 3D Projection
```python
# Reduce to 3 components
pca = PCA(n_components=3)
X_3d = pca.fit_transform(X_scaled)

# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y)
```

## Files in This Folder
- `pca_tutorial.ipynb` - Interactive Jupyter notebook
- `pca.py` - Implementation from scratch
- `visualization_examples.ipynb` - 2D/3D visualizations
- `image_compression.ipynb` - PCA for images
- `datasets/` - Sample datasets

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [PCA Explained Visually](https://setosa.io/ev/principal-component-analysis/)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)
- [Eigenfaces Paper](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)

## Next Steps
1. Open `pca_tutorial.ipynb` in Jupyter
2. Understand eigenvectors and eigenvalues
3. Practice with Iris dataset visualization
4. Try image compression with PCA
5. Compare PCA with t-SNE and UMAP
6. Implement PCA from scratch
7. Use PCA as preprocessing for other algorithms

---
**Happy Learning! 🚀**
