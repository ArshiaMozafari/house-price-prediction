# Regression-house-price-prediction
#### 1st file: Linear and Polynomiyal Regression on Califormia housing dataset, using Regularization as a method to reduce variance and prevent overfitting.
##### Packages: numpy, pandas, matplotlib, scikit-learn.

#### 2nd file: Multiple regression models: RandomForest, XGBoost, Catboost, ANN.
##### Packages: all of those above + tensorflow.

# 🏡 California Housing Price Prediction

This project explores the **California Housing dataset** from [Scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) to build and evaluate regression models for predicting median house values.

## Overview
Predict California housing prices using multiple regression models. The dataset contains 8 features and ~20,000 samples. Goal: maximize test R² while avoiding overfitting.

## Models Tested

### 1. Linear Regression
- Baseline model.
- Train R² ≈ 0.61, Test R² ≈ 0.58
- Underfits due to limited nonlinear modeling.
-With Polynomial features(degree = 3) + lasso regularization it reached test R² = 0.68
### 2. Random Forest
- Ensemble of decision trees.
- Parameters: n_estimators=500, max_depth=20
- Cross-validation and Test R² ≈ 0.81
- Good performance; slower on CPU with large trees.

### 3. XGBoost
- Gradient boosting trees.
- Parameters: n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3
- Cross-validation R² ≈ 0.85, Test R² ≈ 0.86
- Best tree-based performance; needs more trees due to weaker regularization.

### 4. CatBoost
- Gradient boosting optimized for categorical/numerical data.
- Parameters: iterations=1000, depth=8, learning_rate=0.05, l2_leaf_reg=3
- Cross-validation and Test R² ≈ 0.85
- Similar to XGBoost; requires more iterations.

### 5. Artificial Neural Network (ANN)
- Fully connected network: 128 → 64 → 32 → 1 (ReLU, linear output)
- Optimizer: Adam, learning_rate=0.0005
- Regularization: L2 (λ=0.0005 on first layer)
- EarlyStopping: patience=15
- Train R² ≈ 0.81, Test R² ≈ 0.79

## Key Insights
1. Tree-based models outperform ANN on this dataset.
2. ANN performance sensitive to layers, neurons, L2, learning rate, and scaling.
3. Maximum achievable test R² is ~0.85 with current features.

