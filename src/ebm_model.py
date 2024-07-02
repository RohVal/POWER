from preprocessing import preprocess_data
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_model, split_data

from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
from interpret import preserve


df = preprocess_data(path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")
df = df.drop(columns=['Date and time'])

wind_directions = ['E/NE','E/SE', 'N',
 'N/NE',
 'N/NW',
 'NE',
 'NW',
 'S',
 'S/SE',
 'S/SW',
 'SE',
 'SW',
 'W',
 'W/NW',
 'W/SW']

def determine_wind_direction(row):
  for direction in wind_directions:
      if row[direction] == 1:
          return direction
  return 'E'  # Default to 'E' if all are zero

df3 = df

# 
df3['WindDirection'] = df3.apply(determine_wind_direction, axis=1)

# Drop old wind dir cols 
df3 = df3.drop(columns=wind_directions)


X_train, X_test, y_train, y_test = split_data(df=df3)

ebm = ExplainableBoostingRegressor()
ebm.fit(X_train, y_train)

# Evaluate the R² score on the training set
r2_train = ebm.score(X_train, y_train)
# print(f'R² score on the training set: {r2_train:.4f}')

# Evaluate the R² score on the test set
r2_test = ebm.score(X_test, y_test)
# print(f'R² score on the test set: {r2_test:.4f}')

train_pred = ebm.predict(X_train)
test_pred = ebm.predict(X_test)

mse_train = mean_squared_error(y_train, train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, train_pred)

mse_test = mean_squared_error(y_test, test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, test_pred)

print("Training Data Metrics:")
print(f"MSE: {mse_train}")
print(f"RMSE: {rmse_train}")
print(f"MAE: {mae_train}")
print(f"R²: {r2_train}")

print("Test Data Metrics:")
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")

# preserve(ebm.explain_global())
# preserve(show(ebm.explain_local(X_test[:5], y_test[:5]), 0))