# Linear Regression 📈

## Overview
Linear Regression is a supervised learning algorithm used for predicting continuous values. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

## Types
1. **Simple Linear Regression** - One independent variable
2. **Multiple Linear Regression** - Multiple independent variables

## Mathematical Foundation

### Simple Linear Regression
```
y = β₀ + β₁x + ε
```
Where:
- `y` = dependent variable (target)
- `x` = independent variable (feature)
- `β₀` = intercept
- `β₁` = slope (coefficient)
- `ε` = error term

### Multiple Linear Regression
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

## Key Concepts

### Cost Function (Mean Squared Error)
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

### Assumptions
1. **Linearity** - Linear relationship between features and target
2. **Independence** - Observations are independent
3. **Homoscedasticity** - Constant variance of residuals
4. **Normality** - Residuals are normally distributed
5. **No Multicollinearity** - Features are not highly correlated

## Use Cases
- 📊 **Sales Forecasting** - Predict future sales based on historical data
- 🏠 **House Price Prediction** - Estimate property values
- 📈 **Stock Price Prediction** - Financial forecasting
- 🌡️ **Temperature Prediction** - Weather forecasting
- 💰 **Salary Prediction** - Based on experience, education, etc.

## Advantages
✅ Simple and easy to implement  
✅ Fast training and prediction  
✅ Interpretable coefficients  
✅ Works well with linearly separable data  
✅ Low computational cost  

## Disadvantages
❌ Assumes linear relationship  
❌ Sensitive to outliers  
❌ Prone to overfitting with many features  
❌ Cannot capture complex patterns  

## Evaluation Metrics
- **R² Score** - Coefficient of determination (0 to 1, higher is better)
- **Mean Squared Error (MSE)** - Average squared difference
- **Root Mean Squared Error (RMSE)** - Square root of MSE
- **Mean Absolute Error (MAE)** - Average absolute difference

## Real-World Datasets
1. **House Prices** - Kaggle House Prices Dataset
2. **Insurance Costs** - Medical Cost Personal Dataset
3. **Car Prices** - Vehicle pricing dataset
4. **Energy Consumption** - Power consumption prediction
5. **Student Performance** - Grade prediction

## Files in This Folder
- `linear_regression_tutorial.ipynb` - Interactive Jupyter notebook
- `linear_regression.py` - Python implementation from scratch
- `datasets/` - Sample datasets for practice

## Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)

## Next Steps
1. Open `linear_regression_tutorial.ipynb` in Jupyter
2. Follow along with the examples
3. Try implementing from scratch in `linear_regression.py`
4. Practice with real-world datasets

---
**Happy Learning! 🚀**
