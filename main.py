
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Energy data - ver01.xlsx'  # Update with the correct path
data = pd.read_excel(file_path)
print(data.columns)
print(data.head())
columns_to_plot = ['Heating consumption [kWh/m2/ year]', 'DHW consumption [kWh/m2/ year]', 'Cooling consumption [kWh/m2/ year]']
categories = data['Urban Area']

for column in columns_to_plot:
    data[column] = pd.to_numeric(data[column], errors='coerce')

data.plot(x='Urban Area', y=columns_to_plot, kind='bar', figsize=(10, 6), width=0.8)

plt.title('Energy Consumption Comparison (Heating, DHW, Cooling)')
plt.xlabel('Buildings (A to E)')
plt.ylabel('Consumption (kWh/m²/year)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Convert the necessary columns to numeric for calculation
data['Total occupancy'] = pd.to_numeric(data['Total occupancy'], errors='coerce')
data['Total consumption [kWh/m2/ year]'] = pd.to_numeric(data['Total consumption [kWh/m2/ year]'], errors='coerce')

# Scatter plot to compare Total Occupancy vs Total Consumption
plt.figure(figsize=(10, 6))
plt.scatter(data['Total occupancy'], data['Total consumption [kWh/m2/ year]'], color='blue')
plt.title('Energy Consumption vs Total Occupancy')
plt.xlabel('Total Occupancy')
plt.ylabel('Total Consumption [kWh/m²/year]')
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert necessary columns to numeric for calculation
data['Total occupancy'] = pd.to_numeric(data['Total occupancy'], errors='coerce')
data['Total consumption [kWh/m2/ year]'] = pd.to_numeric(data['Total consumption [kWh/m2/ year]'], errors='coerce')

# Filter out missing or non-numeric data
data = data.dropna(subset=['Urban Area', 'Total occupancy', 'Total consumption [kWh/m2/ year]'])

# Create a figure with two subplots, one for Total Occupancy and one for Total Consumption
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Total Occupancy on the left y-axis (primary axis)
ax1.bar(data['Urban Area'], data['Total occupancy'], color='blue', label='Total Occupancy', alpha=0.6)
ax1.set_xlabel('Urban Area')
ax1.set_ylabel('Total Occupancy', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Comparison of Total Occupancy and Total Consumption by Urban Area')

# Create a second y-axis (secondary axis) to plot Total Consumption
ax2 = ax1.twinx()
ax2.plot(data['Urban Area'], data['Total consumption [kWh/m2/ year]'], color='green', marker='o', label='Total Consumption (kWh/m²/year)', linestyle='--')
ax2.set_ylabel('Total Consumption [kWh/m²/year]', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add legends for both axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Select the relevant features (Total occupancy, etc.) and target (Energy consumption)
# features = ['Total occupancy', 'Roof arra [m2]', 'Total consumption [kWh/m2/ year]']  # Add more features if necessary
# data = data.dropna(subset=features)
#
# print(data.columns)
#
# # Convert the selected columns to numeric
# for feature in features:
#     data[feature] = pd.to_numeric(data[feature], errors='coerce')
#
# X = data[['Total occupancy', 'Roof arra [m2]']]  # Use relevant columns as inputs
# y = data['Total consumption [kWh/m2/ year]']
#
# # Normalize the features
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
#
# # Reshape the data for LSTM (samples, timesteps, features)
# X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
#
# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(50, return_sequences=False, input_shape=(1, X_train.shape[2])))
# model.add(Dense(1))
#
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper right')
# plt.show()
#
# # Predict energy consumption on test data
# y_pred = model.predict(X_test)
#
# # Inverse transform to get the actual values
# y_pred_rescaled = scaler.inverse_transform(y_pred)
# y_test_rescaled = scaler.inverse_transform(y_test)
#
# # Plot the results
# plt.figure(figsize=(10,6))
# plt.plot(y_test_rescaled, label='True Energy Consumption')
# plt.plot(y_pred_rescaled, label='Predicted Energy Consumption')
# plt.title('True vs Predicted Energy Consumption')
# plt.xlabel('Test Samples')
# plt.ylabel('Energy Consumption (kWh/m²/year)')
# plt.legend()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
file_path = 'Energy data - ver01.xlsx'  # Change path accordingly
data = pd.read_excel(file_path)

# Display the first few rows to understand the structure
print(data.head())

# Select features and target (Total consumption [kWh/m2/ year])
features = ['Roof arra [m2]', 'Total occupancy', 'Heating consumption [kWh/m2/ year]',
            'DHW consumption [kWh/m2/ year]', 'Cooling consumption [kWh/m2/ year]']
target = 'Total consumption [kWh/m2/ year]'

# Convert the selected columns to numeric to avoid any errors
for column in features + [target]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Drop rows with missing values
data = data.dropna(subset=features + [target])

# Separate features (X) and target (y)
X = data[features].values
y = data[target].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Reshape the data for LSTM: [samples, time steps, features]
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()

# First LSTM layer with Dropout regularization
model.add(LSTM(units=100, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Dense layer to output a single value
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to get actual values
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_inverse = scaler.inverse_transform(predictions)

# Evaluation Metrics
mse = mean_squared_error(y_test_inverse, predictions_inverse)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inverse, predictions_inverse)
r2 = r2_score(y_test_inverse, predictions_inverse)

# Print evaluation results
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-Squared (R²): {r2}')

# Plot the true vs predicted energy consumption
plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse, label='True Energy Consumption')
plt.plot(predictions_inverse, label='Predicted Energy Consumption')
plt.title('True vs Predicted Energy Consumption')
plt.xlabel('Test Samples')
plt.ylabel('Energy Consumption (kWh/m²/year)')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data

file_path = 'Energy data - ver01.xlsx'  # Change path accordingly
data = pd.read_excel(file_path)
# Features and target selection
features = ['Floors', 'Roof arra [m2]', 'Total occupancy', 'Heating consumption [kWh/m2/ year]',
            'DHW consumption [kWh/m2/ year]', 'Cooling consumption [kWh/m2/ year]']
target = 'Total consumption [kWh/m2/ year]'

# Data Preprocessing
data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')
data = data.dropna(subset=features + [target])

# Split the data into training and test sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data for LSTM model (only for LSTM, others don't need normalization)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementing other models

# 1. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# 2. Random Forest
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# 3. Gradient Boosting
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gradient_boosting_model.fit(X_train, y_train)
y_pred_gb = gradient_boosting_model.predict(X_test)

# Plot the true values vs predicted values for each model
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Energy Consumption', color='blue')

plt.plot(y_pred_linear, label='Linear Regression Predictions', color='red', linestyle='dashed')
plt.plot(y_pred_rf, label='Random Forest Predictions', color='green', linestyle='dashed')
plt.plot(y_pred_gb, label='Gradient Boosting Predictions', color='orange', linestyle='dashed')

# Titles and labels
plt.title('True vs Predicted Energy Consumption (Different Models)')
plt.xlabel('Test Samples')
plt.ylabel('Energy Consumption (kWh/m²/year)')
plt.legend()
plt.tight_layout()

# Show plot
plt.show()

# Calculate and print the evaluation metrics
models = {
    'Linear Regression': y_pred_linear,
    'Random Forest': y_pred_rf,
    'Gradient Boosting': y_pred_gb
}

for model_name, y_pred in models.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# Load the data
file_path = 'Energy data - ver01.xlsx'  # Update with the correct path
data = pd.read_excel(file_path)

# Select the relevant features and target for energy prediction
features = ['Roof arra [m2]', 'Total occupancy']  # Modify this with the correct features
target = 'Total consumption [kWh/m2/ year]'

# Handle missing values in features and target
data = data.dropna(subset=features + [target])

# Convert target and feature columns to numeric values if necessary
for column in features + [target]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Ensure there are no NaN values in target variable (y) after converting to numeric
data = data.dropna(subset=[target])

# Split the data into features (X) and target (y)
X = data[features].values
y = data[target].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# IMPORTANT: Scale the target values as well (for LSTM)
target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))

# Reshaping input data for LSTM (LSTM expects 3D input)
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(1, X_train_scaled.shape[1]), return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=300, batch_size=8, verbose=0)

# Make predictions with LSTM model
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled)

# Evaluation Metrics for LSTM
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)

# Print the results for LSTM
print(f'LSTM - MSE: {mse_lstm:.4f}, R²: {r2_lstm:.4f}')

# Additional Models (Linear Regression, Random Forest, Gradient Boosting)
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

# Evaluate and Compare Models
models = {'LSTM': y_pred_lstm, 'Linear Regression': y_pred_lr, 'Random Forest': y_pred_rf, 'Gradient Boosting': y_pred_gb}

for name, y_pred in models.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")

# Plot the comparison of predictions with True Energy Consumption
plt.figure(figsize=(10, 6))

# Plot True Energy Consumption
plt.plot(y_test, label='True Energy Consumption', color='blue', linestyle='--')

# Plot predictions from all models
plt.plot(y_pred_lstm, label='LSTM Predictions', color='orange')
plt.plot(y_pred_lr, label='Linear Regression Predictions', color='green')
plt.plot(y_pred_rf, label='Random Forest Predictions', color='red')
plt.plot(y_pred_gb, label='Gradient Boosting Predictions', color='purple')

# Add title and labels
plt.title('Comparison of Energy Consumption Predictions')
plt.xlabel('Test Samples')
plt.ylabel('Energy Consumption (kWh/m²/year)')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Ensure that y_test and the predicted arrays have the same length
y_test = y_test[:len(y_pred_lstm)]

# Plot the comparison of predictions with True Energy Consumption
plt.figure(figsize=(10, 6))

# Plot True Energy Consumption
plt.plot(range(len(y_test)), y_test, label='True Energy Consumption', color='blue', linestyle='--')

# Plot predictions from all models
plt.plot(range(len(y_pred_lstm)), y_pred_lstm, label='LSTM Predictions', color='orange')
plt.plot(range(len(y_pred_lr)), y_pred_lr, label='Linear Regression Predictions', color='green')
plt.plot(range(len(y_pred_rf)), y_pred_rf, label='Random Forest Predictions', color='red')
plt.plot(range(len(y_pred_gb)), y_pred_gb, label='Gradient Boosting Predictions', color='purple')

# Add title and labels
plt.title('Comparison of Energy Consumption Predictions')
plt.xlabel('Test Samples')
plt.ylabel('Energy Consumption (kWh/m²/year)')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Ensure that y_test and the predicted arrays have the same length
min_len = min(len(y_test), len(y_pred_lstm), len(y_pred_lr), len(y_pred_rf), len(y_pred_gb))

# Truncate all to the same length
y_test = y_test[:min_len]
y_pred_lstm = y_pred_lstm[:min_len]
y_pred_lr = y_pred_lr[:min_len]
y_pred_rf = y_pred_rf[:min_len]
y_pred_gb = y_pred_gb[:min_len]

# Plot the comparison of predictions with True Energy Consumption
plt.figure(figsize=(10, 6))

# Plot True Energy Consumption
plt.plot(range(len(y_test)), y_test, label='True Energy Consumption', color='blue', linestyle='--', marker='o')
plt.plot(range(len(y_pred_lstm)), y_pred_lstm, label='LSTM Predictions', color='orange', linestyle='-', marker='x')
plt.plot(range(len(y_pred_lr)), y_pred_lr, label='Linear Regression Predictions', color='green', linestyle='-', marker='v')
plt.plot(range(len(y_pred_rf)), y_pred_rf, label='Random Forest Predictions', color='red', linestyle='-', marker='s')
plt.plot(range(len(y_pred_gb)), y_pred_gb, label='Gradient Boosting Predictions', color='purple', linestyle='-', marker='d')

# Add title and labels
plt.title('Comparison of Energy Consumption Predictions')
plt.xlabel('Test Samples')
plt.ylabel('Energy Consumption (kWh/m²/year)')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
