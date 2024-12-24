# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv('Cement_Data.csv')
X = data[['Blains_Measurement', 'Soundness', 'LOI', 'Insoluble_Residue', 'SO3']]
y = data[['Initial_Setting_Time', 'Final_Setting_Time', '1_Day_Strength', '3_Day_Strength', '7_Day_Strength', '28_Day_Strength']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN Regression model
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train the model on the training data
knn_regressor.fit(X_train_scaled, y_train)

# Predict the output variables on the testing data
y_pred = knn_regressor.predict(X_test_scaled)

# Evaluate the model using additional metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# MAPE calculation
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(y_test, y_pred)

# Percentage error calculation
percentage_error = np.mean((y_test - y_pred) / y_test) * 100

# Max error calculation
max_error = np.max(np.abs(y_test - y_pred))

# AIC and BIC calculations
n = len(X_train)
p = len(X_train.columns)
aic = n * np.log(mse) + 2 * p
bic = n * np.log(mse) + p * np.log(n)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Percentage Error:", percentage_error)
print("Max Error:", max_error)
print("AIC:", aic)
print("BIC:", bic)

# Create a graph to visualize actual vs predicted values
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, ax in enumerate(axs.flatten()):
    ax.scatter(y_test.iloc[:, i], y_pred[:, i], color='blue')
    ax.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(y.columns[i])

plt.tight_layout()
plt.show()