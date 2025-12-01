# Gradient Boosting 🚀

## Overview
Gradient Boosting is a powerful **ensemble learning** technique that builds models sequentially, where each new model corrects errors made by previous models. It's one of the most effective ML algorithms for structured/tabular data.

## How It Works

### Sequential Learning
```
Model₁ → Residuals₁ → Model₂ → Residuals₂ → Model₃ → ... → Final Model
```

### Algorithm Steps
1. **Start with simple model** (often just the mean)
2. **Calculate residuals** (errors) from predictions
3. **Train new model** to predict these residuals
4. **Add to ensemble** with a learning rate
5. **Repeat** until stopping criterion

### Mathematical Foundation
```
F(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)
```
Where:
- F(x) = final prediction
- F₀(x) = initial model
- hᵢ(x) = weak learner i
- η = learning rate
- m = number of iterations

## Key Concepts

### Gradient Descent in Function Space
- Traditional gradient descent: Update parameters
- Gradient boosting: Update predictions
- Fits new models to negative gradient of loss function

### Weak Learners
- Typically shallow decision trees (depth 3-8)
- Each tree is "weak" but ensemble is "strong"
- Trees are built sequentially, not in parallel

### Learning Rate (Shrinkage)
- Controls contribution of each tree
- **Small learning rate** (0.01-0.1): Better generalization, needs more trees
- **Large learning rate** (0.3-1.0): Faster training, may overfit

## Popular Implementations

### 1. Gradient Boosting (Scikit-learn)
- Original implementation
- Good baseline
- Slower than modern alternatives

### 2. XGBoost (Extreme Gradient Boosting)
- **Most popular** for competitions
- Fast and efficient
- Regularization built-in
- Handles missing values
- Parallel processing

### 3. LightGBM (Microsoft)
- **Fastest** training speed
- Lower memory usage
- Leaf-wise tree growth
- Great for large datasets

### 4. CatBoost (Yandex)
- Handles **categorical features** automatically
- Robust to overfitting
- Ordered boosting
- Good default parameters

## Advantages
✅ Extremely high accuracy  
✅ Handles various data types  
✅ Automatic feature interactions  
✅ Provides feature importance  
✅ Robust to outliers  
✅ Handles missing values (XGBoost, LightGBM)  
✅ Less feature engineering needed  

## Disadvantages
❌ Prone to overfitting (needs careful tuning)  
❌ Slower training than Random Forest  
❌ Sensitive to hyperparameters  
❌ Sequential training (can't parallelize fully)  
❌ Requires more memory  
❌ Less interpretable  

## Hyperparameters

### Tree Parameters
- **max_depth** - Maximum tree depth (3-10 typical)
- **min_child_weight** - Minimum sum of weights in leaf
- **min_samples_split** - Minimum samples to split

### Boosting Parameters
- **n_estimators** - Number of trees (100-1000+)
- **learning_rate** - Shrinkage (0.01-0.3)
- **subsample** - Fraction of samples per tree (0.5-1.0)
- **colsample_bytree** - Fraction of features per tree

### Regularization
- **reg_alpha** - L1 regularization (Lasso)
- **reg_lambda** - L2 regularization (Ridge)
- **gamma** - Minimum loss reduction for split

## Use Cases
- 💳 **Credit Scoring** - Loan default prediction
- 🎯 **Customer Churn** - Retention prediction
- 📈 **Sales Forecasting** - Revenue prediction
- 💰 **Fraud Detection** - Transaction fraud
- 🏥 **Disease Prediction** - Medical diagnosis
- 🏆 **Kaggle Competitions** - Winning solution
- 📊 **Ranking Problems** - Search engines
- 🎮 **Click-Through Rate** - Ad optimization

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Log Loss

### Regression
- R² Score
- MSE, RMSE, MAE
- MAPE (Mean Absolute Percentage Error)

## Real-World Datasets
1. **House Prices** - Kaggle competition
2. **Titanic** - Classification
3. **Credit Card Fraud** - Imbalanced data
4. **Customer Churn** - Business analytics
5. **Black Friday Sales** - Retail forecasting
6. **Loan Prediction** - Financial risk

## Preventing Overfitting

### 1. Reduce Model Complexity
- Decrease max_depth
- Increase min_child_weight
- Increase min_samples_split

### 2. Add Randomness
- Use subsample < 1.0
- Use colsample_bytree < 1.0

### 3. Regularization
- Increase reg_alpha or reg_lambda
- Increase gamma

### 4. Early Stopping
- Monitor validation error
- Stop when no improvement

### 5. Lower Learning Rate
- Use smaller learning rate
- Increase n_estimators

## XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed | Fast | **Fastest** | Medium |
| Memory | Medium | **Low** | High |
| Accuracy | High | High | **Highest** |
| Categorical | Manual | Manual | **Automatic** |
| Default Params | Good | Good | **Best** |
| Overfitting | Medium | Higher | **Lower** |

## Feature Importance

### Types
1. **Gain** - Average gain when feature is used
2. **Cover** - Average coverage of splits
3. **Frequency** - Number of times feature is used

### Interpretation
- Identifies most important features
- Helps with feature selection
- Provides model insights

## Installation

### XGBoost
```bash
pip install xgboost
```

### LightGBM
```bash
pip install lightgbm
```

### CatBoost
```bash
pip install catboost
```

## Best Practices

### 1. Start Simple
- Begin with default parameters
- Use early stopping
- Monitor train/validation error

### 2. Hyperparameter Tuning
- Use cross-validation
- Grid search or random search
- Bayesian optimization for advanced tuning

### 3. Feature Engineering
- Create interaction features
- Handle missing values appropriately
- Encode categorical variables (except CatBoost)

### 4. Validation Strategy
- Use stratified k-fold for classification
- Time-based split for time series
- Monitor multiple metrics

## Comparison with Other Algorithms

| Aspect | Gradient Boosting | Random Forest | Neural Networks |
|--------|-------------------|---------------|-----------------|
| Tabular Data | **Excellent** | Excellent | Good |
| Training Speed | Medium | Fast | Slow |
| Prediction Speed | Fast | Fast | Fast |
| Interpretability | Medium | Medium | Low |
| Hyperparameter Sensitivity | High | Low | Very High |

## Files in This Folder
- `gradient_boosting_tutorial.ipynb` - Interactive Jupyter notebook
- `xgboost_examples.py` - XGBoost implementation
- `lightgbm_examples.py` - LightGBM implementation
- `catboost_examples.py` - CatBoost implementation
- `hyperparameter_tuning.ipynb` - Tuning guide
- `datasets/` - Sample datasets

## Resources
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [StatQuest: Gradient Boosting](https://www.youtube.com/watch?v=3CC4N4z3GJc)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [Kaggle: XGBoost Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)

## Next Steps
1. Open `gradient_boosting_tutorial.ipynb` in Jupyter
2. Compare XGBoost, LightGBM, and CatBoost
3. Practice hyperparameter tuning
4. Analyze feature importance
5. Work on Kaggle competition
6. Learn early stopping and cross-validation

---
**Happy Learning! 🚀**
