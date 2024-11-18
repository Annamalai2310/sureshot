import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to load data from Excel file and scale it
def load_and_prepare_data(file_path, n_steps=8):
    data = pd.read_excel(file_path, sheet_name='Sheet1', header=0, index_col=0)
    data = data.T
    data.index = pd.to_datetime(data.index, format='%Y-%m')

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Prepare sequences
    X, y = create_sequences(data_scaled, n_steps)
    return data, X, y, scaler

# Create sequences for LSTM model input
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i, :])
        y.append(data[i, :])  # Predict the next quarter's values
    return np.array(X), np.array(y)

# Function to create and compile LSTM model
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(3)  # Output 3 values: Sales, Profit, and Average Price
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to predict using the trained model
def predict_sales_profit_price(model, data_scaled, scaler, n_steps):
    X_input = np.array([data_scaled[-n_steps:]])  # Last n_steps as input
    predicted_scaled = model.predict(X_input)
    predicted = scaler.inverse_transform(predicted_scaled)[0]
    return predicted

# Optional function to plot historical and predicted values
def plot_predictions(data, predicted):
    plt.figure(figsize=(12, 8))

    # Sales plot
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['Sales(in crores)'], marker='o', label='Sales', color='b')
    plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
    plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[0], color='r', label='Predicted Sales')
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)

    # Profit plot
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['Profit(in crores)'], marker='o', label='Profit', color='g')
    plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
    plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[1], color='r', label='Predicted Profit')
    plt.title('Profit Over Time')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)

    # Average price plot
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['Avg. Price'], marker='o', label='Average Price', color='purple')
    plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
    plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[2], color='r', label='Predicted Average Price')
    plt.title('Average Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt
