import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import calendar

print("Libraries imported successfully")

EB_data = pd.read_csv(r"C:\Users\hp\Downloads\electricity_daily_9years_household.csv")
EB_data.drop(columns=['Household_ID', 'Date'], inplace=True)

EB_data = EB_data.apply(pd.to_numeric, errors="coerce")

print("Nulls before dropna:\n", EB_data.isnull().sum())
EB_data.dropna(inplace=True)
print("Shape after dropna:", EB_data.shape)

hour_cols = EB_data.filter(like='_Hours').columns
EB_data = EB_data[(EB_data[hour_cols] >= 0).all(axis=1)]
print("Shape after removing negative hours:", EB_data.shape)

Q1 = EB_data['Daily_kWh'].quantile(0.25)
Q3 = EB_data['Daily_kWh'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Valid range: {lower_bound:.2f} to {upper_bound:.2f}")
print(f"Rows before: {len(EB_data)}")

EB_datan = EB_data[(EB_data['Daily_kWh'] >= lower_bound) & (EB_data['Daily_kWh'] <= upper_bound)]
print(f"Rows after: {len(EB_datan)}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(EB_data['Daily_kWh'], bins=30)
plt.title("Before")
plt.subplot(1, 2, 2)
plt.hist(EB_datan['Daily_kWh'], bins=30)
plt.title("After")
plt.tight_layout()
plt.show()

x = EB_datan.drop(columns=['Daily_kWh', 'Monthly_Units', 'Price_Rs'])
y = EB_datan['Daily_kWh']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train and Test data split successfully")
print("Train size is", x_train.shape)
print("Test size is", x_test.shape)

scaler = StandardScaler()
x_trainscaled = scaler.fit_transform(x_train)
x_testscaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_trainscaled, y_train)
print("Model trained successfully")

y_pred = model.predict(x_testscaled)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(MAE, 2))
print("Mean Squared Error (MSE):", round(MSE, 2))
print("Root Mean Squared Error (RMSE):", round(RMSE, 2))
print("R-Squared (R2):", round(R2 * 100, 2), "%")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.xlabel("Actual Daily_kWh")
plt.ylabel("Predicted Daily_kWh")
plt.title("Actual vs Predicted")
plt.show()

joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
feature_columns = list(x_train.columns)
joblib.dump(feature_columns, 'feature_columns.pkl')
print("Model, scaler, and feature columns saved successfully")
print("Feature order:", feature_columns)

year = int(input("Enter Year: "))
month = int(input("Enter Month (1-12): "))
days = calendar.monthrange(year, month)[1]

user_values = []
for feature in feature_columns:
    value = float(input(f"Enter {feature}: "))
    user_values.append(value)

user_values_scaled = scaler.transform([user_values])
daily_kwh = model.predict(user_values_scaled)[0]
monthly_units = daily_kwh * days

if monthly_units <= 100:
    bill = monthly_units * 5.5 + 120
elif monthly_units <= 250:
    bill = (100 * 5.5) + (monthly_units - 100) * 6.5 + 120
else:
    bill = (100 * 5.5) + (150 * 6.5) + (monthly_units - 250) * 7.2 + 120

print("\n----- Prediction Result -----")
print(f"Month/Year      : {month}/{year} ({days} days)")
print(f"Daily kWh       : {daily_kwh:.2f}")
print(f"Monthly Units   : {monthly_units:.2f}")
print(f"Estimated Bill  : Rs {bill:.2f}")