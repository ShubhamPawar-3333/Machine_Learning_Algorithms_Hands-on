# Logistic Regression 🎯

## Overview
Logistic Regression is a supervised learning algorithm used for **classification** problems. Despite its name, it's used for classification, not regression. It predicts the probability of an instance belonging to a particular class.

## Types
1. **Binary Logistic Regression** - Two classes (0 or 1)
2. **Multinomial Logistic Regression** - Multiple classes (>2)
3. **Ordinal Logistic Regression** - Ordered classes

## Mathematical Foundation

### Sigmoid Function
```
σ(z) = 1 / (1 + e⁻ᶻ)
```
Where: `z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

The sigmoid function maps any real value to a probability between 0 and 1.

### Decision Boundary
```
If σ(z) ≥ 0.5 → Class 1
If σ(z) < 0.5 → Class 0
```

### Cost Function (Log Loss)
```
J(θ) = -(1/m) Σ[y·log(h(x)) + (1-y)·log(1-h(x))]
```

## Key Concepts

### Assumptions
1. **Binary/Multinomial outcome** - Dependent variable is categorical
2. **Independence** - Observations are independent
3. **No Multicollinearity** - Features are not highly correlated
4. **Linearity** - Linear relationship between log-odds and features
5. **Large Sample Size** - Works better with more data

## Use Cases
- 📧 **Email Spam Detection** - Spam or Not Spam
- 🏥 **Disease Diagnosis** - Disease present or absent
- 💳 **Credit Card Fraud Detection** - Fraudulent or legitimate
- 🎓 **Student Admission** - Admit or reject
- 📱 **Customer Churn** - Will customer leave or stay?
- 🎬 **Sentiment Analysis** - Positive or negative review

## Advantages
✅ Simple and efficient  
✅ Provides probability scores  
✅ Works well for linearly separable classes  
✅ Less prone to overfitting (with regularization)  
✅ Easy to interpret  

## Disadvantages
❌ Assumes linear decision boundary  
❌ Cannot solve non-linear problems  
❌ Sensitive to outliers  
❌ Requires feature scaling  

## Evaluation Metrics
- **Accuracy** - Overall correctness
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve
- **Confusion Matrix** - Detailed classification results

## Real-World Datasets
1. **Titanic Survival** - Kaggle Titanic Dataset
2. **Heart Disease** - UCI Heart Disease Dataset
3. **Credit Card Fraud** - Highly imbalanced dataset
4. **Diabetes Prediction** - Pima Indians Diabetes
5. **Customer Churn** - Telecom churn dataset
6. **Loan Default** - Credit risk assessment

## Files in This Folder
- `logistic_regression_tutorial.ipynb` - Interactive Jupyter notebook
- `logistic_regression.py` - Python implementation from scratch
- `datasets/` - Sample datasets for practice

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)

## Next Steps
1. Open `logistic_regression_tutorial.ipynb` in Jupyter
2. Learn about sigmoid function and decision boundaries
3. Practice with binary and multi-class classification
4. Implement from scratch and compare with scikit-learn

---
**Happy Learning! 🚀**
