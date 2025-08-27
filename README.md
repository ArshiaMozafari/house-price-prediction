# Regression-Regularization-house-price-prediction-
Linear and Polynomiyal Regression on Califormia housing dataset and using Regularization as a method to decrease variance and prevent overtting.

# 🏡 California Housing Price Prediction

This project explores the **California Housing dataset** from [Scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) to build and evaluate regression models for predicting median house values.

---

## 📊 Dataset
- Source: California housing prices (1990 census data).
- Shape: ~20,000 records × 8 features.
- Target: `MedHouseVal` (Median House Value, in $100,000s).
- Features include:
  - Median Income  
  - House Age  
  - Average Rooms  
  - Average Bedrooms  
  - Population  
  - Average Occupancy  
  - Latitude  
  - Longitude  

---

## ⚙️ Methods
1. **Data Preprocessing**
   - Train/test split (80/20).
   - Standardization (`StandardScaler`).
   - Polynomial feature expansion (degree=2).
   - Residual analysis (histogram, density plots).

2. **Models Tried**
   - Linear Regression  
   - Lasso Regression (with cross-validation for alpha)  
   - Ridge Regression  

3. **Evaluation Metrics**
   - R² (coefficient of determination)  
   - MSE (mean squared error)  
   - MAE (mean absolute error)  

---

## 📈 Results
- **Linear Regression (degree=1):** R² ≈ 0.57 (test)
- **Polynomial Regression (degree=2):** R² ≈ 0.64 (test)
- **Polynomial Regression (degree=3):** Overfitting (R² < 0 on test)
- **Lasso (α = 0.001, degree=2):** R² ≈ 0.66 (test), sparse coefficients
- **Ridge (degree=2):** R² ≈ 0.64 (test)

🔎 **Best model:** Polynomial Regression (degree=2, with Lasso regularization) → R² ≈ **0.66** on test data.  

---

## 📉 Error Analysis
- Residuals mostly lie between **-2 and 2** (≈ ±$200k).  
- Residual distribution is approximately normal, centered at 0.  
- Model tends to struggle with extreme house values.  

---

