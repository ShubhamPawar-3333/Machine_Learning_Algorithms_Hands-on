# Support Vector Machines (SVM) 🎯

## Overview
Support Vector Machines are powerful supervised learning algorithms used for **classification** and **regression**. SVMs find the optimal hyperplane that maximizes the margin between different classes.

## How It Works

### Key Concept: Maximum Margin Classifier
```
Class A  •  •  •  |  Hyperplane  |  •  •  •  Class B
         ←margin→ | (decision)  | ←margin→
```

The SVM finds the hyperplane that:
1. Separates the classes
2. Maximizes the margin (distance) to nearest data points
3. These nearest points are called **Support Vectors**

## Types of SVM

### 1. Linear SVM
- Used when data is linearly separable
- Finds a straight line (2D) or hyperplane (higher dimensions)

### 2. Non-Linear SVM (Kernel SVM)
- Used when data is not linearly separable
- Uses kernel trick to transform data into higher dimensions

### 3. SVM for Regression (SVR)
- Predicts continuous values
- Finds a hyperplane that fits the data within a margin

## Kernel Functions

### Linear Kernel
```
K(x, y) = x · y
```
Best for linearly separable data

### Polynomial Kernel
```
K(x, y) = (x · y + c)^d
```
Good for polynomial relationships

### Radial Basis Function (RBF/Gaussian)
```
K(x, y) = exp(-γ||x - y||²)
```
Most popular, works well for non-linear data

### Sigmoid Kernel
```
K(x, y) = tanh(α(x · y) + c)
```
Similar to neural networks

## Key Concepts

### Support Vectors
- Data points closest to the hyperplane
- Critical for defining the decision boundary
- Removing other points doesn't change the model

### Margin
- Distance between hyperplane and nearest data points
- **Hard Margin**: No misclassification allowed
- **Soft Margin**: Allows some misclassification (C parameter)

### C Parameter (Regularization)
- **Small C**: Larger margin, more misclassification (underfitting)
- **Large C**: Smaller margin, less misclassification (overfitting)

### Gamma Parameter (for RBF kernel)
- **Small γ**: Far-reaching influence, smoother decision boundary
- **Large γ**: Close influence, complex decision boundary

## Advantages
✅ Effective in high-dimensional spaces  
✅ Works well with clear margin of separation  
✅ Memory efficient (uses support vectors only)  
✅ Versatile (different kernels for different data)  
✅ Robust to overfitting (especially in high dimensions)  

## Disadvantages
❌ Slow training on large datasets  
❌ Sensitive to feature scaling  
❌ Difficult to interpret  
❌ Choosing the right kernel is challenging  
❌ Not suitable for large datasets (>100k samples)  
❌ Doesn't provide probability estimates directly  

## Hyperparameters

### Important Parameters
- **C** - Regularization parameter
- **kernel** - Type of kernel (linear, rbf, poly, sigmoid)
- **gamma** - Kernel coefficient (for rbf, poly, sigmoid)
- **degree** - Degree of polynomial (for poly kernel)
- **class_weight** - Handle imbalanced datasets

## Use Cases
- 🖼️ **Image Classification** - Handwriting recognition, face detection
- 📝 **Text Classification** - Spam detection, sentiment analysis
- 🧬 **Bioinformatics** - Protein classification, cancer detection
- 💳 **Credit Scoring** - Loan approval
- 🎯 **Pattern Recognition** - Character recognition
- 🏥 **Medical Diagnosis** - Disease classification

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

### Regression (SVR)
- R² Score
- MSE, RMSE, MAE

## Real-World Datasets
1. **Iris Dataset** - Multi-class classification
2. **Breast Cancer** - Binary classification
3. **MNIST Digits** - Image classification
4. **Spam Detection** - Text classification
5. **Credit Card Fraud** - Imbalanced classification
6. **Wine Quality** - Multi-class classification

## Feature Scaling
⚠️ **CRITICAL**: SVM is sensitive to feature scaling!

Always scale features before training:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Comparison with Other Algorithms

| Aspect | SVM | Logistic Regression | Decision Tree |
|--------|-----|---------------------|---------------|
| Non-linear | Yes (kernels) | No | Yes |
| Interpretability | Low | High | High |
| Speed | Slow | Fast | Fast |
| Large datasets | Poor | Good | Good |
| High dimensions | Excellent | Good | Poor |

## Files in This Folder
- `svm_tutorial.ipynb` - Interactive Jupyter notebook
- `svm.py` - Python implementation
- `datasets/` - Sample datasets for practice

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [StatQuest: SVM](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)
- [SVM Tutorial](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

## Next Steps
1. Open `svm_tutorial.ipynb` in Jupyter
2. Understand different kernels and their use cases
3. Practice hyperparameter tuning (C and gamma)
4. Compare linear vs non-linear SVM
5. Always remember to scale your features!

---
**Happy Learning! 🚀**
