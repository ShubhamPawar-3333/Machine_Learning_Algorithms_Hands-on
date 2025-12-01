# Decision Trees 🌳

## Overview
Decision Trees are versatile supervised learning algorithms used for both **classification** and **regression** tasks. They work by recursively splitting the data based on feature values to create a tree-like model of decisions.

## Types
1. **Classification Trees** - Predict categorical outcomes
2. **Regression Trees** - Predict continuous values
3. **CART** - Classification and Regression Trees

## How It Works

### Tree Structure
```
Root Node (entire dataset)
    ├── Internal Node (decision based on feature)
    │   ├── Leaf Node (final prediction)
    │   └── Leaf Node (final prediction)
    └── Internal Node (decision based on feature)
        ├── Leaf Node (final prediction)
        └── Leaf Node (final prediction)
```

### Splitting Criteria

#### For Classification
- **Gini Impurity**: `Gini = 1 - Σ(pᵢ)²`
- **Entropy**: `Entropy = -Σ(pᵢ·log₂(pᵢ))`
- **Information Gain**: `IG = Entropy(parent) - Weighted_Avg(Entropy(children))`

#### For Regression
- **Mean Squared Error (MSE)**: Minimize variance in leaf nodes
- **Mean Absolute Error (MAE)**: Minimize absolute differences

## Key Concepts

### Advantages
✅ Easy to understand and interpret  
✅ Requires little data preprocessing  
✅ Handles both numerical and categorical data  
✅ Can capture non-linear relationships  
✅ Feature scaling not required  
✅ Provides feature importance  

### Disadvantages
❌ Prone to overfitting  
❌ Unstable (small changes in data can change tree)  
❌ Biased towards features with more levels  
❌ Can create overly complex trees  
❌ Not suitable for extrapolation  

## Hyperparameters
- **max_depth** - Maximum depth of tree
- **min_samples_split** - Minimum samples to split a node
- **min_samples_leaf** - Minimum samples in a leaf node
- **max_features** - Number of features to consider for split
- **criterion** - Splitting criterion (gini, entropy, mse)

## Use Cases
- 🏥 **Medical Diagnosis** - Disease classification
- 💳 **Credit Scoring** - Loan approval decisions
- 🎯 **Customer Segmentation** - Marketing strategies
- 🌾 **Crop Recommendation** - Agricultural decisions
- 🏠 **House Price Prediction** - Real estate valuation
- 📧 **Email Filtering** - Spam detection

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC

### Regression
- R² Score
- MSE, RMSE, MAE

## Real-World Datasets
1. **Iris Dataset** - Flower species classification
2. **Titanic Dataset** - Survival prediction
3. **Wine Quality** - Quality classification
4. **Diabetes Dataset** - Disease prediction
5. **House Prices** - Price prediction (regression)
6. **Credit Risk** - Default prediction

## Preventing Overfitting
1. **Pruning** - Remove unnecessary branches
2. **Set max_depth** - Limit tree depth
3. **min_samples_split** - Require minimum samples for splits
4. **min_samples_leaf** - Require minimum samples in leaves
5. **Use ensemble methods** - Random Forest, Gradient Boosting

## Files in This Folder
- `decision_trees_tutorial.ipynb` - Interactive Jupyter notebook
- `decision_tree.py` - Python implementation
- `datasets/` - Sample datasets for practice

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [Visualizing Decision Trees](https://explained.ai/decision-tree-viz/)

## Next Steps
1. Open `decision_trees_tutorial.ipynb` in Jupyter
2. Learn about splitting criteria and tree construction
3. Visualize decision trees
4. Practice pruning and hyperparameter tuning

---
**Happy Learning! 🚀**
