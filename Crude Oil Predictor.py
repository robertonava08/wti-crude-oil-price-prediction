import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# Fetch historical data
oil_WTI = yf.Ticker("CL=F")
hist_data = oil_WTI.history(period="2y")

# Print the historical data to the terminal
print("August 31, 2023 -- August 31, 2024 WTI Crude Oil Prices")
pd.set_option('display.max_rows', None)
print(hist_data)

# Visualize the data (used matplotlib library)
plt.figure(figsize=(10, 5))
plt.plot(hist_data.index, hist_data['Open'], label='Opening Price', color='red')
plt.plot(hist_data.index, hist_data['Close'], label='Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('WTI Crude Oil Prices (August 31, 2023 -- August 31, 2024)')
plt.legend()
plt.grid(True)
plt.show()

# Prepare the data (uses sklearn)
scaler_X = StandardScaler()
scaler_y = StandardScaler()  

# Scale the features 
scaled_data = scaler_X.fit_transform(hist_data[['Open', 'High', 'Low', 'Volume']])
scaled_dataFrame = pd.DataFrame(scaled_data, columns=['Open', 'High', 'Low', 'Volume'], index=hist_data.index)

# Fit scaler_y on the target variable (Close)
scaled_target = scaler_y.fit_transform(hist_data[['Close']])
scaled_dataFrame['Close'] = scaled_target

# Prepare features and target
X = scaled_dataFrame[['Open', 'High', 'Low', 'Volume']]
y = scaled_dataFrame['Close']

# Split data into training and testing sets (splits into 30% and 70% automatically)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#print("Training set shape:", X_train.shape) use this to check for rows and columns of training set
#print("Testing set shape:", X_test.shape) use this to check for rows and columns of training set 

# start and train the Ridge regression model
ridge_model = Ridge(alpha=2.3)
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred_scaled = ridge_model.predict(X_test)

# Inverse transform the predictions and actual values to the original scale (VERY IMPORTANT)
y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Ensure correct index for predictions
predicted_dates = hist_data.index[-len(y_test):] 

# Combine predicted values with actual data
combined_df = pd.DataFrame({
    'Date': predicted_dates,
    'Predicted_Close': y_pred_original,
    'Actual_Close': y_test_original
})

# Plot the predictions alongside the closing costs 
plt.figure(figsize=(10, 5))
plt.plot(combined_df['Date'], combined_df['Predicted_Close'], label='Predicted Close', color='red')
plt.plot(combined_df['Date'], combined_df['Actual_Close'], label='Actual Close', color='blue')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('WTI Crude Oil Price Predictions vs Actual WTI Crude Oil Prices')
plt.legend()
plt.grid(True)
plt.show()

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_original, y_pred_original)
print("\nMean Squared Error:", mse)

# Calculate R-squared
r2 = ridge_model.score(X_test, y_test)
r2_percent = r2 * 100
print("\nR-squared (percentage): {:.2f}%".format(r2_percent))

# Calculate MAPE
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
print("\nMean Absolute Percentage Error:", mape,"%")

prediction_accuracy = 100 - mape

print(f"Prediction Accuracy: {prediction_accuracy:.2f}%")