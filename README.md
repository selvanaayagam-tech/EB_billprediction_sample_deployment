# Household Electricity Bill Predictor

## 1. Project Overview
A Linear Regression model that predicts a household's daily electricity consumption from appliance usage hours, and scales it up to estimate a full monthly bill using the actual LT Commercial tariff slab structure. Built as a "what-if" calculator, adjust your appliance hours and instantly see how your estimated bill changes.

## 2. Problem Statement
Predict a household's daily electricity consumption (Daily_kWh) based on how many hours each appliance runs per day, then convert that into a monthly bill estimate using a fixed slab-based tariff formula (not predicted by the model, calculated separately since it's a known government rate structure).

## 3. Dataset
A synthetically generated dataset simulating 9 years (2017-2025) of daily electricity usage for a single household, ~3,269 rows, covering 13 household appliances (AC, fridge, heater/geyser, fan, lights, TV, washing machine, and others). The dataset includes realistic data-quality issues (missing values, inconsistent entries, duplicate records) intentionally added for preprocessing practice, along with seasonal usage patterns (higher AC/fan usage in summer, higher heater usage in winter).

## 4. Tools & Libraries Used
- Python
- pandas, NumPy
- scikit-learn (LinearRegression, StandardScaler, train_test_split)
- Matplotlib
- joblib (model persistence)
- Streamlit (deployment)

## 5. Methodology / Approach
1. Cleaned the raw CSV: coerced all columns to numeric, dropped nulls, removed invalid negative values.
2. Removed outliers in Daily_kWh using the IQR method.
3. Split data into train/test sets, scaled features using StandardScaler.
4. Trained a Linear Regression model to predict Daily_kWh from appliance usage hours.
5. Evaluated using MAE, MSE, RMSE, and R-Squared, with an Actual vs Predicted plot to visually confirm fit.
6. Saved the trained model, scaler, and feature order using joblib for deployment.
7. Built a Streamlit app that takes user appliance-hour inputs, predicts daily consumption, scales it to a full month, and applies the LT Commercial tariff formula to output an estimated bill.

## 6. Results
The model achieved a high R-Squared score, closely recovering the true underlying relationship between appliance usage and daily consumption. (Note: this dataset was synthetically generated using a deterministic formula with no added noise, which is why the fit is exceptionally close; real-world smart meter data would be expected to show more variance.)

## 7. Live Demo
Try the deployed app here: yet to deploy

## 8. Key Learnings / Future Work
- Learned why feature scaling doesn't affect OLS Linear Regression's predictions, but does affect coefficient interpretability.
- Learned to separate deterministic business logic (tariff calculation) from the ML prediction step, rather than trying to have a model learn a fixed formula.
- Learned the importance of relative file paths over hardcoded absolute paths when preparing code for deployment.
- Future work: extend this to a SARIMA time-series model on the same dataset to forecast future monthly consumption using the seasonal pattern already present in the data.
