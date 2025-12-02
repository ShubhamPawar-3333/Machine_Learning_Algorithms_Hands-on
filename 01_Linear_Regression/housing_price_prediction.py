"""
California Housing Price Prediction using Linear Regression
===========================================================

Real-world application of Linear Regression on California housing data.

Dataset: California Housing (sklearn built-in)
- 20,640 samples
- 8 features (median income, house age, rooms, etc.)
- Target: Median house value
- Real-world regression problem

Author: Machine Learning Hands-on
Date: 2025-12-02
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_and_explore_data():
    """Load and explore the California housing dataset."""
    print("=" * 70)
    print("LOADING CALIFORNIA HOUSING DATASET")
    print("=" * 70)
    
    # Load data
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFeature Names:")
    for i, name in enumerate(housing.feature_names):
        print(f"  {i+1}. {name}: {housing.feature_names[i]}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum().sum())
    
    print(f"\nStatistical Summary:")
    print(df.describe())
    
    print(f"\nTarget Variable (MedHouseVal):")
    print(f"  Mean: ${df['MedHouseVal'].mean():.2f} (in $100k)")
    print(f"  Median: ${df['MedHouseVal'].median():.2f}")
    print(f"  Min: ${df['MedHouseVal'].min():.2f}")
    print(f"  Max: ${df['MedHouseVal'].max():.2f}")
    
    return df, housing


def visualize_data(df):
    """Visualize dataset characteristics."""
    print("\n" + "=" * 70)
    print("DATA VISUALIZATION")
    print("=" * 70)
    
    # Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['MedHouseVal'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Median House Value ($100k)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of House Prices', fontweight='bold')
    axes[0].axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: ${df["MedHouseVal"].mean():.2f}')
    axes[0].legend()
    
    axes[1].boxplot(df['MedHouseVal'])
    axes[1].set_ylabel('Median House Value ($100k)')
    axes[1].set_title('Box Plot of House Prices', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Feature correlations with target
    plt.figure(figsize=(10, 6))
    correlations = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values()
    correlations.plot(kind='barh', color='teal')
    plt.xlabel('Correlation with House Price')
    plt.title('Feature Correlations with Target', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # Scatter plots for top features
    top_features = df.corr()['MedHouseVal'].abs().sort_values(ascending=False)[1:5].index
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features):
        axes[idx].scatter(df[feature], df['MedHouseVal'], alpha=0.3, s=10)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Median House Value')
        axes[idx].set_title(f'{feature} vs House Price')
        
        # Add trend line
        z = np.polyfit(df[feature], df['MedHouseVal'], 1)
        p = np.poly1d(z)
        axes[idx].plot(df[feature], p(df[feature]), "r--", linewidth=2, alpha=0.8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    """Preprocess the dataset."""
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    
    # Separate features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def train_baseline_model(X_train, X_test, y_train, y_test):
    """Train baseline Linear Regression model."""
    print("\n" + "=" * 70)
    print("BASELINE LINEAR REGRESSION")
    print("=" * 70)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = lr.predict(X_train_scaled)
    y_pred_test = lr.predict(X_test_scaled)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\nTraining R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"\nTraining RMSE: ${train_rmse:.4f} (in $100k)")
    print(f"Test RMSE: ${test_rmse:.4f} (in $100k)")
    print(f"Test MAE: ${test_mae:.4f} (in $100k)")
    
    # Coefficients
    print(f"\nModel Coefficients:")
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': lr.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df.to_string(index=False))
    print(f"\nIntercept: {lr.intercept_:.4f}")
    
    return lr, scaler, y_pred_test


def train_ridge_regression(X_train, X_test, y_train, y_test, scaler):
    """Train Ridge Regression (L2 regularization)."""
    print("\n" + "=" * 70)
    print("RIDGE REGRESSION (L2 Regularization)")
    print("=" * 70)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different alpha values
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    best_alpha = None
    best_score = -np.inf
    
    print("\nTesting different alpha values:")
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        score = ridge.score(X_test_scaled, y_test)
        print(f"  Alpha={alpha:6.3f} -> R² = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    print(f"\nBest Alpha: {best_alpha}")
    
    # Train with best alpha
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    
    print(f"\nRidge Regression Results:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):.4f}")
    
    return ridge, y_pred


def train_lasso_regression(X_train, X_test, y_train, y_test, scaler):
    """Train Lasso Regression (L1 regularization)."""
    print("\n" + "=" * 70)
    print("LASSO REGRESSION (L1 Regularization)")
    print("=" * 70)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different alpha values
    alphas = [0.001, 0.01, 0.1, 1, 10]
    best_alpha = None
    best_score = -np.inf
    
    print("\nTesting different alpha values:")
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        score = lasso.score(X_test_scaled, y_test)
        n_features_used = np.sum(lasso.coef_ != 0)
        print(f"  Alpha={alpha:6.3f} -> R² = {score:.4f}, Features used: {n_features_used}")
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    print(f"\nBest Alpha: {best_alpha}")
    
    # Train with best alpha
    lasso = Lasso(alpha=best_alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    
    print(f"\nLasso Regression Results:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):.4f}")
    
    # Feature selection
    print(f"\nFeature Selection (non-zero coefficients):")
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': lasso.coef_
    })
    selected_features = coef_df[coef_df['Coefficient'] != 0].sort_values('Coefficient', key=abs, ascending=False)
    print(selected_features.to_string(index=False))
    
    return lasso, y_pred


def train_elasticnet(X_train, X_test, y_train, y_test, scaler):
    """Train ElasticNet (L1 + L2 regularization)."""
    print("\n" + "=" * 70)
    print("ELASTICNET (L1 + L2 Regularization)")
    print("=" * 70)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ElasticNet
    elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    elasticnet.fit(X_train_scaled, y_train)
    y_pred = elasticnet.predict(X_test_scaled)
    
    print(f"\nElasticNet Results:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):.4f}")
    
    return elasticnet, y_pred


def plot_predictions(y_test, y_pred_lr, y_pred_ridge, y_pred_lasso, y_pred_elastic):
    """Plot actual vs predicted values for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    models = [
        ('Linear Regression', y_pred_lr),
        ('Ridge Regression', y_pred_ridge),
        ('Lasso Regression', y_pred_lasso),
        ('ElasticNet', y_pred_elastic)
    ]
    
    for idx, (name, y_pred) in enumerate(models):
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[idx].plot([y_test.min(), y_test.max()], 
                      [y_test.min(), y_test.max()], 
                      'r--', lw=2, label='Perfect Prediction')
        axes[idx].set_xlabel('Actual Price ($100k)')
        axes[idx].set_ylabel('Predicted Price ($100k)')
        axes[idx].set_title(f'{name}\nR² = {r2_score(y_test, y_pred):.4f}', 
                           fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals(y_test, y_pred_lr, y_pred_ridge):
    """Plot residuals for model diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals_lr = y_test - y_pred_lr
    residuals_ridge = y_test - y_pred_ridge
    
    # Residual plot
    axes[0].scatter(y_pred_lr, residuals_lr, alpha=0.5, s=20, label='Linear Regression')
    axes[0].scatter(y_pred_ridge, residuals_ridge, alpha=0.5, s=20, label='Ridge Regression')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals_lr, bins=50, alpha=0.6, label='Linear Regression', color='blue')
    axes[1].hist(residuals_ridge, bins=50, alpha=0.6, label='Ridge Regression', color='orange')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CALIFORNIA HOUSING PRICE PREDICTION")
    print("=" * 70)
    
    # Load and explore
    df, housing = load_and_explore_data()
    visualize_data(df)
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    lr, scaler, y_pred_lr = train_baseline_model(X_train, X_test, y_train, y_test)
    ridge, y_pred_ridge = train_ridge_regression(X_train, X_test, y_train, y_test, scaler)
    lasso, y_pred_lasso = train_lasso_regression(X_train, X_test, y_train, y_test, scaler)
    elasticnet, y_pred_elastic = train_elasticnet(X_train, X_test, y_train, y_test, scaler)
    
    # Visualizations
    plot_predictions(y_test, y_pred_lr, y_pred_ridge, y_pred_lasso, y_pred_elastic)
    plot_residuals(y_test, y_pred_lr, y_pred_ridge)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    
    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet'],
        'R² Score': [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_ridge),
            r2_score(y_test, y_pred_lasso),
            r2_score(y_test, y_pred_elastic)
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
            np.sqrt(mean_squared_error(y_test, y_pred_elastic))
        ],
        'MAE': [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_ridge),
            mean_absolute_error(y_test, y_pred_lasso),
            mean_absolute_error(y_test, y_pred_elastic)
        ]
    })
    
    print("\n", results.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ HOUSING PRICE PREDICTION PROJECT COMPLETE!")
    print("=" * 70)
    print("\nKey Learnings:")
    print("1. Linear Regression provides baseline performance")
    print("2. Ridge helps with multicollinearity")
    print("3. Lasso performs feature selection")
    print("4. ElasticNet combines benefits of Ridge and Lasso")
    print("5. Regularization prevents overfitting")
