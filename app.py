import streamlit as st
import joblib
import calendar
from huggingface_hub import hf_hub_download

REPO_ID = "SelvaMech/electricity-bill-regression"

model_path = hf_hub_download(repo_id=REPO_ID, filename="linear_regression_model.pkl")
scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl")
features_path = hf_hub_download(repo_id=REPO_ID, filename="feature_columns.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_columns = joblib.load(features_path)

st.title("Household Electricity Bill Predictor")
st.write("Enter your typical daily appliance usage hours to estimate your monthly bill.")

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Year", min_value=2017, max_value=2035, value=2026, step=1)
with col2:
    month = st.selectbox("Month", list(range(1, 13)), index=0)

days = calendar.monthrange(year, month)[1]
st.caption(f"{days} days in this month")

st.subheader("Appliance Usage (hours per day)")

user_values = []
for feature in feature_columns:
    label = feature.replace('_Hours', '').replace('_', ' ')
    value = st.number_input(f"{label} Hours", min_value=0.0, max_value=24.0, value=0.0, step=0.5)
    user_values.append(value)

if st.button("Predict Bill"):
    user_values_scaled = scaler.transform([user_values])
    daily_kwh = model.predict(user_values_scaled)[0]
    monthly_units = daily_kwh * days

    if monthly_units <= 100:
        bill = monthly_units * 5.5 + 120
    elif monthly_units <= 250:
        bill = (100 * 5.5) + (monthly_units - 100) * 6.5 + 120
    else:
        bill = (100 * 5.5) + (150 * 6.5) + (monthly_units - 250) * 7.2 + 120

    st.subheader("Prediction Result")
    st.metric("Predicted Daily kWh", f"{daily_kwh:.2f}")
    st.metric("Predicted Monthly Units", f"{monthly_units:.2f}")
    st.metric("Estimated Bill", f"Rs {bill:.2f}")
