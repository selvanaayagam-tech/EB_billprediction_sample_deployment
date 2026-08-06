# Household Electricity Bill Predictor

## ⚠️ About This Project
This is a **practice/study project** built specifically to learn end-to-end model deployment, training a model, saving it, hosting it on Hugging Face, and deploying a live app via Streamlit Community Cloud. It intentionally uses a single Linear Regression model for simplicity while learning the deployment pipeline.

In a complete, production-style project, multiple models would be trained and compared (Linear Regression, Random Forest, Gradient Boosting, etc.), with the best-performing one (highest R², lowest error) selected for deployment. A future version of this project will do exactly that, multiple models compared inside the notebook, with the best one selected and used in the final `main.py`. For now, this repo focuses on getting the deployment pipeline right.

## 📂 Which File Is Which
- **`app.py`** - the actual deployed app code (Streamlit). This is what runs live on Streamlit Community Cloud. It loads the trained model files directly from Hugging Face and serves the interactive bill calculator.
- **`main.py`** - the training pipeline script. Run this to clean the data, train the model, evaluate it, and save the model/scaler/feature files. Not part of the live app, this is the "how the model was built" file.
- **`*.ipynb`** - the full exploratory notebook: EDA, data cleaning steps, outlier detection, visualizations, and the training pipeline in its original working form. Check this file for the complete step-by-step walkthrough of the approach.
- **Model files (`.pkl`)** — NOT stored in this repo. The trained model, scaler, and feature list are hosted on Hugging Face: **https://huggingface.co/SelvaMech/electricity-bill-regression/tree/main**. `app.py` downloads them automatically at runtime.

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
- huggingface_hub (model file hosting/retrieval)

## 5. Methodology / Approach
1. Cleaned the raw CSV: coerced all columns to numeric, dropped nulls, removed invalid negative values.
2. Removed outliers in Daily_kWh using the IQR method.
3. Split data into train/test sets, scaled features using StandardScaler.
4. Trained a Linear Regression model to predict Daily_kWh from appliance usage hours.
5. Evaluated using MAE, MSE, RMSE, and R-Squared, with an Actual vs Predicted plot to visually confirm fit.
6. Saved the trained model, scaler, and feature order using joblib, uploaded to a Hugging Face Model repo.
7. Built a Streamlit app that downloads the model files from Hugging Face, takes user appliance-hour inputs, predicts daily consumption, scales it to a full month, and applies the LT Commercial tariff formula to output an estimated bill.

## 6. Results
The model achieved a very high R-Squared score, closely recovering the true underlying relationship between appliance usage and daily consumption. (Note: this dataset was synthetically generated using a deterministic formula with no added noise, which is why the fit is exceptionally close; real-world smart meter data would be expected to show more variance.)

## 7. Live Demo
- **Try the app:** https://ebbillpredictionsampledeployment-bpne9za7xb7p2kkp7eejza.streamlit.app
- **Model files:** https://huggingface.co/SelvaMech/electricity-bill-regression

## 8. Key Learnings / Future Work
- Learned the full deployment pipeline: train → save with joblib → host on Hugging Face → serve via Streamlit Community Cloud.
- Learned to separate deterministic business logic (tariff calculation) from the ML prediction step, rather than trying to have a model learn a fixed formula.
- Learned the importance of relative/hosted file paths over hardcoded local paths when preparing code for deployment.
- **Future work:** expand this into a full model comparison project, train multiple models (Random Forest, Gradient Boosting, etc.) inside the notebook, evaluate each, and select the best-performing one for the final deployed `main.py`. Also planning a SARIMA time-series model on the same dataset to forecast future monthly consumption using its seasonal pattern.

## 9. Limitations
- Trained on synthetic, single-household data, not real smart-meter readings.
- Assumes a fixed electricity tariff rate across all years; real tariffs are revised periodically.
- Does not account for appliance efficiency (star ratings) or seasonal usage patterns as separate model inputs.
- Should not be used as a substitute for an actual utility bill; it is an educational estimation tool.
