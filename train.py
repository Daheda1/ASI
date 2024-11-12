import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('pizza_sales.csv')
data = data.dropna()

data = data.drop('pizza_ingredients', axis=1)
data = data.drop('pizza_name', axis=1)

print(data.info())


#input:
#Day of week
#Time of day

data['order_hour'] = pd.to_datetime(data['order_time'], format='%H:%M:%S').dt.hour
print(data['order_hour'].value_counts())

# Group by 'order_date' and 'order_hour' to calculate the total quantity ordered for each hour of each day
daily_hourly_quantity = data.groupby(['order_date', 'order_hour'])['quantity'].sum().reset_index()

# Rename columns for clarity
daily_hourly_quantity.columns = ['order_date', 'order_hour' ,'total_quantity']

daily_hourly_quantity['order_weekday'] = pd.to_datetime(daily_hourly_quantity['order_date'], dayfirst=True, errors='coerce').dt.day_of_week
daily_hourly_quantity['order_month'] = pd.to_datetime(daily_hourly_quantity['order_date'], dayfirst=True, errors='coerce').dt.month

daily_hourly_quantity['order_month'] = daily_hourly_quantity['order_month'].fillna(0).astype('int')
daily_hourly_quantity['order_weekday'] = daily_hourly_quantity['order_weekday'].fillna(0).astype('int')
daily_hourly_quantity['total_quantity'] = daily_hourly_quantity['total_quantity'].fillna(0).astype('int')

daily_hourly_quantity= daily_hourly_quantity.drop('order_date', axis=1)


print(daily_hourly_quantity['order_weekday'].value_counts())
print(daily_hourly_quantity['order_month'].value_counts())

# Display the result
print(daily_hourly_quantity)

X = daily_hourly_quantity.drop('total_quantity', axis=1)
y = daily_hourly_quantity['total_quantity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Model Results:")
print("Mean Absolute Error:", mae_xgb)
print("Mean Squared Error:", mse_xgb)
print("R² Score:", r2_xgb)

joblib.dump(xgb_model, 'xgb_regressor_model.joblib')

