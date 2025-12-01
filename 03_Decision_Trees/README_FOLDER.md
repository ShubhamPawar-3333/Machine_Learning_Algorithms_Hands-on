# Decision Trees - Complete Guide

This folder contains comprehensive resources for learning Decision Trees.

## 📁 Files in This Folder

### 1. **README.md** (This file)
Overview of Decision Trees algorithm, theory, and resources.

### 2. **decision_tree.py**
General-purpose Decision Tree implementation with:
- From-scratch implementation using Gini impurity
- Scikit-learn examples
- Visualization functions
- Feature importance analysis
- Reusable for any dataset

**Use this for**: Learning the algorithm, quick experiments, general projects

### 3. **decision_trees_tutorial.ipynb**
Interactive Jupyter notebook covering:
- Basic Decision Tree concepts
- Iris dataset examples
- Tree visualization
- Hyperparameter tuning
- Overfitting prevention
- Regression examples
- Implementation from scratch

**Use this for**: Step-by-step learning, interactive exploration

### 4. **fraud_detection_project.py**
**Real-world production-level project** - Credit Card Fraud Detection
- 284,807 real transactions
- Highly imbalanced dataset (0.172% fraud)
- Three approaches to handle imbalance:
  - Baseline model
  - Class weights
  - SMOTE oversampling
- Comprehensive evaluation metrics
- ROC curves and confusion matrices
- Feature importance analysis

**Use this for**: Real-world application, understanding imbalanced data

---

## 🚀 Quick Start

### Option 1: Learn the Basics
```bash
jupyter notebook decision_trees_tutorial.ipynb
```

### Option 2: Run General Implementation
```python
python decision_tree.py
```

### Option 3: Real-World Project
1. Download dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in this folder
3. Uncomment the main code in `fraud_detection_project.py`
4. Run: `python fraud_detection_project.py`

---

## 📊 When to Use Decision Trees

**Best For:**
- ✅ Interpretable models (need to explain decisions)
- ✅ Mixed feature types (categorical + numerical)
- ✅ Non-linear relationships
- ✅ Feature importance analysis
- ✅ Quick prototyping

**Not Ideal For:**
- ❌ Very high-dimensional data (use Random Forest instead)
- ❌ When you need the absolute best accuracy (ensemble methods better)
- ❌ Extrapolation beyond training data range

---

## 🎯 Learning Path

1. **Start Here**: `decision_trees_tutorial.ipynb` - Learn basics with Iris dataset
2. **Practice**: Modify `decision_tree.py` for your own datasets
3. **Real-World**: Run `fraud_detection_project.py` to see production application
4. **Next**: Move to Random Forest (ensemble of Decision Trees)

---

## 💡 Key Concepts

### Splitting Criteria
- **Gini Impurity**: Measures node impurity (0 = pure, 0.5 = max impurity)
- **Entropy**: Information gain-based splitting

### Important Hyperparameters
- `max_depth`: Controls tree depth (prevents overfitting)
- `min_samples_split`: Minimum samples to split a node
- `min_samples_leaf`: Minimum samples in leaf nodes
- `class_weight`: Handle imbalanced data

### Advantages
1. Easy to understand and visualize
2. No feature scaling needed
3. Handles non-linear relationships
4. Works with mixed data types
5. Provides feature importance

### Disadvantages
1. Prone to overfitting
2. Unstable (small data changes = different tree)
3. Biased toward dominant classes
4. Can create overly complex trees

---

## 📚 Additional Resources

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

---

**Happy Learning! 🌳**
