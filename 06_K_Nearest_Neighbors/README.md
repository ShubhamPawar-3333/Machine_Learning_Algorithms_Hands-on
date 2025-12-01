# K-Nearest Neighbors (KNN) 👥

## Overview
K-Nearest Neighbors is a simple, intuitive supervised learning algorithm used for both **classification** and **regression**. It's a **lazy learning** algorithm that makes predictions based on the K closest training examples in the feature space.

## How It Works

### Algorithm Steps
1. **Choose K** - Number of neighbors to consider
2. **Calculate Distance** - Find distance to all training samples
3. **Find K Nearest** - Select K closest neighbors
4. **Vote/Average**:
   - **Classification**: Majority vote among K neighbors
   - **Regression**: Average of K neighbors' values

### Distance Metrics

#### Euclidean Distance (Most Common)
```
d = √(Σ(xᵢ - yᵢ)²)
```

#### Manhattan Distance
```
d = Σ|xᵢ - yᵢ|
```

#### Minkowski Distance (Generalized)
```
d = (Σ|xᵢ - yᵢ|^p)^(1/p)
```

#### Hamming Distance
For categorical variables

## Key Concepts

### Choosing K
- **Small K (e.g., K=1)**: 
  - More sensitive to noise
  - Complex decision boundary
  - Prone to overfitting
  
- **Large K (e.g., K=100)**:
  - Smoother decision boundary
  - Less sensitive to noise
  - May underfit

- **Rule of Thumb**: K = √n (where n is number of samples)
- **Best Practice**: Use cross-validation to find optimal K
- **K should be odd** for binary classification (avoid ties)

### Lazy Learning
- **No training phase** - Just stores data
- **Prediction is slow** - Computes distances at prediction time
- **Memory intensive** - Stores all training data

## Advantages
✅ Simple and easy to understand  
✅ No training phase (fast training)  
✅ Naturally handles multi-class problems  
✅ No assumptions about data distribution  
✅ Can be used for both classification and regression  
✅ Adapts easily as new data is added  

## Disadvantages
❌ Slow prediction (especially with large datasets)  
❌ Memory intensive (stores all training data)  
❌ Sensitive to feature scaling  
❌ Curse of dimensionality (poor performance in high dimensions)  
❌ Sensitive to irrelevant features  
❌ Doesn't work well with imbalanced datasets  

## Hyperparameters

### Important Parameters
- **n_neighbors (K)** - Number of neighbors
- **weights** - 'uniform' or 'distance' (closer neighbors have more influence)
- **metric** - Distance metric (euclidean, manhattan, minkowski)
- **p** - Power parameter for Minkowski distance
- **algorithm** - Method to compute neighbors (auto, ball_tree, kd_tree, brute)

## Use Cases
- 🎬 **Recommendation Systems** - Movie/product recommendations
- 🏥 **Medical Diagnosis** - Disease classification
- 💳 **Credit Scoring** - Loan approval
- 🖼️ **Image Recognition** - Handwriting recognition
- 📝 **Text Classification** - Document categorization
- 🌾 **Pattern Recognition** - Various applications
- 📊 **Anomaly Detection** - Outlier detection

## Feature Scaling
⚠️ **CRITICAL**: KNN is highly sensitive to feature scaling!

Always scale features before training:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

### Regression
- R² Score
- MSE, RMSE, MAE

## Real-World Datasets
1. **Iris Dataset** - Classic classification problem
2. **MNIST Digits** - Handwritten digit recognition
3. **Wine Quality** - Multi-class classification
4. **Breast Cancer** - Binary classification
5. **Customer Segmentation** - Clustering similar customers
6. **House Prices** - Regression problem

## Optimization Techniques

### 1. Feature Selection
Remove irrelevant features to improve performance

### 2. Dimensionality Reduction
Use PCA to reduce dimensions before applying KNN

### 3. Data Structures
- **KD-Tree**: Fast for low dimensions (<20)
- **Ball Tree**: Better for high dimensions
- **Brute Force**: Simple but slow

### 4. Weighted KNN
Give more weight to closer neighbors

## Curse of Dimensionality
As dimensions increase:
- All points become equidistant
- KNN performance degrades
- **Solution**: Use dimensionality reduction (PCA)

## Comparison with Other Algorithms

| Aspect | KNN | Decision Tree | SVM |
|--------|-----|---------------|-----|
| Training Speed | Instant | Fast | Slow |
| Prediction Speed | Slow | Fast | Medium |
| Memory Usage | High | Low | Medium |
| Interpretability | Medium | High | Low |
| Feature Scaling | Required | Not Required | Required |

## Files in This Folder
- `knn_tutorial.ipynb` - Interactive Jupyter notebook
- `knn.py` - Python implementation from scratch
- `datasets/` - Sample datasets for practice

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [StatQuest: KNN](https://www.youtube.com/watch?v=HVXime0nQeI)
- [KNN for Beginners](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

## Next Steps
1. Open `knn_tutorial.ipynb` in Jupyter
2. Experiment with different K values
3. Compare different distance metrics
4. Practice feature scaling
5. Understand the curse of dimensionality
6. Try weighted KNN

---
**Happy Learning! 🚀**
