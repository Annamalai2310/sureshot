# from flask import Flask, render_template, request
# # Ensure the 'static' directory exists before saving the plot
# import os
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from tensorflow.keras.losses import MeanSquaredError

# app = Flask(__name__)

# # Create the 'static' folder if it doesn't exist
# static_folder = os.path.join(os.getcwd(), 'static')
# if not os.path.exists(static_folder):
#     os.makedirs(static_folder)

# # Load your trained LSTM model
# model = tf.keras.models.load_model('lstm_sales_prediction.h5', custom_objects={'mse': MeanSquaredError()})

# # Company and file path dictionary
# company_files = {
#     "Airtel": "../Airtel.xlsx",
#     "TVS": "../TVS.xlsx",
#     # Add other companies and their file paths here
# }

# @app.route('/')
# def home():
#     return render_template('index.html', companies=company_files.keys())

# @app.route('/predict', methods=['POST'])
# def predict():
#     company_name = request.form['company']
#     excel_file = company_files.get(company_name)

#     # Load data from selected Excel file
#     data = pd.read_excel(excel_file, sheet_name='Sheet1', header=0, index_col=0)
#     data = data.T
#     data.index = pd.to_datetime(data.index, format='%Y-%m')

#     # Scale data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)

#     # Create sequences
#     n_steps = 8
#     X_input = np.array([data_scaled[-n_steps:]])  # Last 4 quarters as input for prediction

#     # Predict using the LSTM model
#     predicted_scaled = model.predict(X_input)
#     predicted = scaler.inverse_transform(predicted_scaled)[0]

#     # Generate plot
#     plt.figure(figsize=(12, 8))

#     # Plot sales
#     plt.subplot(3, 1, 1)
#     plt.plot(data.index, data['Sales(in crores)'], marker='o', label='Sales', color='b')
#     plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
#     plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[0], color='r', label='Predicted Sales')
#     plt.plot([data.index[-1], data.index[-1] + pd.DateOffset(months=3)], [data['Sales(in crores)'].iloc[-1], predicted[0]], color='b', linestyle='-', linewidth=2)  # Connect with a line
#     plt.title('Sales Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Sales')
#     plt.legend()

#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
#     plt.xticks(rotation=45)

#     # Plot profit
#     plt.subplot(3, 1, 2)
#     plt.plot(data.index, data['Profit(in crores)'], marker='o', label='Profit', color='g')
#     plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
#     plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[1], marker='o', color='r', label='Predicted Profit')
#     plt.plot([data.index[-1], data.index[-1] + pd.DateOffset(months=3)], [data['Profit(in crores)'].iloc[-1], predicted[1]], color='green', linestyle='-', linewidth=2)  # Connect with a line
#     plt.title('Profit Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Profit')
#     plt.legend()

#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
#     plt.xticks(rotation=45)

#     # Plot average price
#     plt.subplot(3, 1, 3)
#     plt.plot(data.index, data['Avg. Price'], marker='o', label='Average Price', color='purple')
#     plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
#     plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[2], marker='o', color='r', label='Predicted Average Price')
#     plt.plot([data.index[-1], data.index[-1] + pd.DateOffset(months=3)], [data['Avg. Price'].iloc[-1], predicted[2]], color='purple', linestyle='-', linewidth=2)  # Connect with a line
#     plt.title('Average Price Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Average Price')
#     plt.legend()

#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
#     plt.xticks(rotation=45)
    
#     # Save the plot
#     graph_path = f"static/{company_name}_prediction.png"
#     plt.tight_layout()
#     plt.savefig(graph_path)

#     # Return prediction results and graph path
#     return render_template('result.html', prediction={
#         'Sales': predicted[0],
#         'Profit': predicted[1],
#         'Average_Price': predicted[2]
#     }, graph_path=graph_path)

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, render_template, request
# import os
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from tensorflow.keras.losses import MeanSquaredError

# app = Flask(__name__)

# # Create the 'static' folder if it doesn't exist
# static_folder = os.path.join(os.getcwd(), 'static')
# if not os.path.exists(static_folder):
#     os.makedirs(static_folder)

# # Load your trained LSTM model
# model = tf.keras.models.load_model('lstm_sales_prediction.h5', custom_objects={'mse': MeanSquaredError()})

# # Company and file path dictionary
# company_files = {
#     "Airtel": "../Airtel.xlsx",
#     "TVS": "../TVS.xlsx",
#     # Add other companies and their file paths here
# }

# @app.route('/')
# def home():
#     return render_template('index.html', companies=company_files.keys())

# @app.route('/predict', methods=['POST'])
# def predict():
#     company_name = request.form['company']
#     excel_file = company_files.get(company_name)

#     # Load data from selected Excel file
#     data = pd.read_excel(excel_file, sheet_name='Sheet1', header=0, index_col=0)
#     data = data.T
#     data.index = pd.to_datetime(data.index, format='%Y-%m')

#     # Scale data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)

#     # Create sequences
#     n_steps = 8
#     X_input = np.array([data_scaled[-n_steps:]])  # Last 4 quarters as input for prediction

#     # Predict using the LSTM model
#     predicted_scaled = model.predict(X_input)
#     predicted = scaler.inverse_transform(predicted_scaled)[0]

#     # Calculate change direction for arrows
#     arrow_directions = {
#         'Sales': 'up' if predicted[0] > data['Sales(in crores)'].iloc[-1] else 'down',
#         'Profit': 'up' if predicted[1] > data['Profit(in crores)'].iloc[-1] else 'down',
#         'Average_Price': 'up' if predicted[2] > data['Avg. Price'].iloc[-1] else 'down'
#     }

#     # Generate plot
#     plt.figure(figsize=(12, 8))

#     # Plot sales
#     plt.subplot(3, 1, 1)
#     plt.plot(data.index, data['Sales(in crores)'], marker='o', label='Sales', color='b')
#     plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
#     plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[0], color='r', label='Predicted Sales')
#     plt.plot([data.index[-1], data.index[-1] + pd.DateOffset(months=3)], [data['Sales(in crores)'].iloc[-1], predicted[0]], color='b', linestyle='-', linewidth=2)
#     plt.title('Sales Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Sales')
#     plt.legend()
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
#     plt.xticks(rotation=45)

#     # Plot profit
#     plt.subplot(3, 1, 2)
#     plt.plot(data.index, data['Profit(in crores)'], marker='o', label='Profit', color='g')
#     plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
#     plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[1], marker='o', color='r', label='Predicted Profit')
#     plt.plot([data.index[-1], data.index[-1] + pd.DateOffset(months=3)], [data['Profit(in crores)'].iloc[-1], predicted[1]], color='green', linestyle='-', linewidth=2)
#     plt.title('Profit Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Profit')
#     plt.legend()
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
#     plt.xticks(rotation=45)

#     # Plot average price
#     plt.subplot(3, 1, 3)
#     plt.plot(data.index, data['Avg. Price'], marker='o', label='Average Price', color='purple')
#     plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Prediction Point')
#     plt.scatter(data.index[-1] + pd.DateOffset(months=3), predicted[2], marker='o', color='r', label='Predicted Average Price')
#     plt.plot([data.index[-1], data.index[-1] + pd.DateOffset(months=3)], [data['Avg. Price'].iloc[-1], predicted[2]], color='purple', linestyle='-', linewidth=2)
#     plt.title('Average Price Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Average Price')
#     plt.legend()
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    # plt.xticks(rotation=45)

#     # Save the plot
#     graph_path = f"static/{company_name}_prediction.png"
#     plt.tight_layout()
#     plt.savefig(graph_path)

#     # Return prediction results, graph path, and arrow directions
#     return render_template('result.html', company_name=company_name, prediction={
#         'Sales': predicted[0],
#         'Profit': predicted[1],
#         'Average_Price': predicted[2]
#     }, graph_path=graph_path, arrow_directions=arrow_directions)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' 
 # Secret key for session management

# Create the 'static' folder if it doesn't exist
static_folder = os.path.join(os.getcwd(), 'static')
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

# Load your trained LSTM model
model = tf.keras.models.load_model('lstm_sales_prediction.h5', custom_objects={'mse': MeanSquaredError()})

# Company and file path dictionary
company_files = {
    "Aarti Industries Limited": "Excel_Data/Aarti_industries.xlsx",
    "Abbott India Limited": "Excel_Data/Abbott_India.xlsx",
    "Astral Poly Technik Limited": "Excel_Data/Astral.xlsx",
    "AU Small Finance Bank": "Excel_Data/AU_small_finance.xlsx",
    "Bata India Limited": "Excel_Data/Bata.xlsx",
    "Berger Paints (i) Limited": "Excel_Data/Berger.xlsx",
    "Chambal Fertilizers & Chemicals Limited": "Excel_Data/Chambal_Fertilizers.xlsx",
    "Coforge (Niit Tech)":"Excel_Data/Coforge.xlsx",
    "Dalmia Bharat Limited": "Excel_Data/Dalmia.xlsx",
    "Dr. Lal Path Labs Ltd.": "Excel_Data/Lal_path.xlsx",
    "Hcl Technologies Limited": "Excel_Data/HCL_Tech.xlsx",
    "Infosys Limited":"Excel_Data/Infosys.xlsx",
    "IPCA Laboratories Limited ": "Excel_Data/IPCA_.xlsx",
    "Jubilant Foodworks Limited": "Excel_Data/Jubiliant.xlsx",
    "L&t Technology Services Limited": "Excel_Data/LTTS.xlsx",
    "LTI Mindtree Ltd": "Excel_Data/Mindtree.xlsx",
    "Lupin Limited": "Excel_Data/Lupin.xlsx",
    "Manappuram Finance Limited": "Excel_Data/Manappuram.xlsx",
    "Max Financial Services Limited": "Excel_Data/Max_Financial.xlsx",
    "Metropolis Healthcare Limited": "Excel_Data/Metropolis.xlsx",
    "PVR Inox Ltd": "Excel_Data/PVR.xlsx",
    "Sun Pharmaceuticals Industries Limited": "Excel_Data/Sun_pharma.xlsx",
    "The Ramco Cements Limited": "Excel_Data/Ramco_Cements.xlsx",
    "Trent Limited": "Excel_Data/Trent.xlsx",
    "Upl Limited": "Excel_Data/UPL.xlsx",
}

# Default username and password
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "password123"  # Change this to a more secure password

@app.route('/')
def home():
    return render_template('index.html', companies=company_files.keys())

# Modify the predict() function to ensure that NaN values are handled properly
@app.route('/predict', methods=['POST'])
def predict():
    company_name = request.form['company']
    session['company_name'] = company_name
    excel_file = company_files.get(company_name)

    # Load data from selected Excel file
    data = pd.read_excel(excel_file, sheet_name='Sheet1', header=0, index_col=0)
    data = data.drop(columns=['2024-12-01'])
    data = data.T

    # Ensure the date index is in Mar-22 format
    data.index = pd.to_datetime(data.index, format='%b-%y')
    print(data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    n_steps = 8
    X_input = np.array([data_scaled[-n_steps:]])  # Last 4 quarters as input for prediction

    predicted_scaled = model.predict(X_input)
    predicted_values = scaler.inverse_transform(predicted_scaled)[0]

    # Extend the data with predicted values for plotting
    future_dates = pd.date_range(data.index[-2], periods=1, freq='3MS')[1:]  # Predicted future dates
    prediction_data = pd.DataFrame([predicted_values], index=future_dates, columns=['Sales(in crores)', 'Profit(in crores)', 'Avg. Price'])
    data = pd.concat([data, prediction_data])  # Using pd.concat instead of append 

    # Plot the data and predicted values
    background_color = '#fdffe2'
    plt.figure(figsize=(12, 8), facecolor=background_color)

    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['Sales(in crores)'], marker='o', label='Sales', color='b')
    plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Recent Value')
    plt.scatter(data.index[-3], None, color='r', label='Predicted Sales')  # Masked
    plt.title('Sales Chart')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.5)

     # Plot profit (masked prediction)
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['Profit(in crores)'], marker='o', label='Profit', color='g')
    plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Recent Value')
    plt.scatter(data.index[-1], None, marker='o', color='r', label='Predicted Profit')  # Masked
    plt.title('Profit Chart')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.5)

    # Plot average price (masked prediction)
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['Avg. Price'], marker='o', label='Average Price', color='purple')
    plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Recent Value')
    plt.scatter(data.index[-1], None, marker='o', color='r', label='Predicted Average Price')  # Masked
    plt.title('Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)

    plt.subplots_adjust(hspace=0.5)

    # Save the plot
    graph_path = f"static/{company_name}_prediction.png"
    plt.tight_layout()
    plt.savefig(graph_path)

    # Return the result with the graph
    return render_template('prediction.html', company_name=company_name, graph_path=graph_path)


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('unmask_predictions'))
        else:
            return "Invalid credentials. Please try again."
    return render_template('login.html')

@app.route('/unmask_predictions')
def unmask_predictions():
    if not session.get('logged_in'):
        return redirect(url_for('predict'))

    company_name = session.get('company_name')  # Use the company from session or user input
    excel_file = company_files.get(company_name)

    # Load data from selected Excel file
    data = pd.read_excel(excel_file, sheet_name='Sheet1', header=0, index_col=0)
    data = data.T

    # Ensure the date index is in Mar-22 format
    data.index = pd.to_datetime(data.index, format='%b-%y')
    print(data)

    recent_sales = data['Sales(in crores)'].iloc[-1]
    last_sales = data['Sales(in crores)'].iloc[-2]

    # Extract past profit and average price values
    historical_profit = data['Profit(in crores)'].iloc[:-1]  # Exclude the most recent value
    historical_price = data['Avg. Price'].iloc[:-1]  # Exclude the most recent value

    print(historical_price)
    print(historical_profit)


    # Calculate the average change in profit and price over the last n periods (let's say last 5 periods)
    profit_change = historical_profit.mean()  # Percentage change in profit
    price_change = historical_price.mean()  # Percentage change in price

    last_two_profit = data['Profit(in crores)'].iloc[-3:]  # Last two profit values
    last_two_price = data['Avg. Price'].iloc[-3:]  # Last two price values

    # Mean difference of the last two values for profit and price
    last_profit_change = (last_two_profit.iloc[-2] + last_two_profit.iloc[-3]) / 2
    last_price_change = (last_two_price.iloc[-2] + last_two_price.iloc[-3]) / 2

    print(historical_profit[-1], historical_price[-1],profit_change, price_change, last_price_change, last_profit_change)
    # Predict if the profit will go up or down based on the trend
    if recent_sales > last_sales:
        profit_direction = 'up'
        price_direction = 'up'  # Profit is trending upwards
    else:
        profit_direction = 'down'
        price_direction = 'down'  # Profit is trending downwards

    # Set the prediction dictionary with the processed Avg. Price and Profit
    prediction = {
        "Sales": recent_sales,
    }

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    n_steps = 8
    X_input = np.array([data_scaled[-n_steps:]])  # Last 4 quarters as input for prediction

    predicted_scaled = model.predict(X_input)
    predicted_values = scaler.inverse_transform(predicted_scaled)[0]

    # Extend the data with predicted values for plotting
    future_dates = pd.date_range(data.index[-1], periods=1, freq='3MS')[1:]  # Predicted future dates
    prediction_data = pd.DataFrame([predicted_values], index=future_dates, columns=['Sales(in crores)', 'Profit(in crores)', 'Avg. Price'])
    data = pd.concat([data, prediction_data])  # Using pd.concat instead of append 

    # Plot the data and predicted values
    background_color = '#fdffe2'
    plt.figure(figsize=(12, 8), facecolor=background_color)

    # Plot Sales
    plt.subplot(3, 1, 1)
    plt.plot(data.index[:-1], data['Sales(in crores)'][:-1], marker='o', label='Sales', color='b')

    # Plot the second-to-last to last segment in red
    plt.plot(data.index[-2:], data['Sales(in crores)'].iloc[-2:], color='r', marker='o', label='Predicted Sales')
    plt.axvline(x=data.index[-2], color='gray', linestyle='--', label='Recent Value')
    point_x = data.index[-2]
    point_y = data['Sales(in crores)'].iloc[-2]  # Value for the last data point
    plt.scatter(point_x, point_y, color='b', s=50, zorder=2)
    last_point_x = data.index[-1]
    last_point_y = data['Sales(in crores)'].iloc[-1]  # Value for the last data point
    plt.scatter(last_point_x, last_point_y, color='r', s=50, zorder=2)  # Masked
    plt.title('Sales Chart')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)

    # Plot Profit
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['Profit(in crores)'], marker='o', label='Profit', color='g')
    plt.axvline(x=data.index[-2], color='gray', linestyle='--', label='Recent Value')  # Masked
    plt.title('Profit Chart')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)

    # Plot Average Price
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['Avg. Price'], marker='o', label='Average Price', color='purple')
    plt.axvline(x=data.index[-2], color='gray', linestyle='--', label='Recent Value') # Masked
    plt.title('Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    plt.xticks(rotation=45)

    # Save the plot
    graph_path = f"static/{company_name}_unmasked_prediction.png"
    plt.tight_layout()
    plt.savefig(graph_path)

    # Return the result with the graph
    return render_template('result.html',  graph_path=graph_path, company_name=company_name, prediction=prediction, arrow_directions={'Profit': profit_direction, 'Avg_Price': price_direction})


@app.route('/logout')
def logout():
    session['logged_in'] = False
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
