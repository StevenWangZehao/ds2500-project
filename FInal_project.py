"""
    DS 2500
    Spring 2024
    Final Project
"""

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def fetch_stock_data(stock_ticker, start_date):
    ''' Given a string for stock ticker and a start date(YYYY-MM-DD)
        Return a DataFrame containing historical stock data using 
        Yahoo Finance API
    '''
    df = yf.Ticker(stock_ticker)
    df_history = df.history(start = start_date)
    
    # Reset index to make 'Date' a column
    df_history.reset_index(inplace = True)
    return df_history

def plot_stock_data(data, stock_ticker):
    ''' Given a data (DataFrame contaning 'Date' and 'Close' columns) and 
        a string for stock ticker
        Plot the stock data
    '''
    plt.figure(figsize = (10, 6))
    plt.plot(data['Date'], data['Close'], marker = '',
             color = 'blue', linewidth = 2)
    plt.title(f'{stock_ticker} Stock Data Over One Year')
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.show()
    
def scale_data(data):
    ''' Given a unscaled data
        Scale the given data using MinMaxScaler to a range of (0, 1) 
        Return the scaled data and the scaler object
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_train_data(scaled_data, training_data_len):
    ''' Given a scaled data and an int for training data length
        Create training data from the scaled data
        Return the input features (x_train) and the target variable (y_train)
    '''
    train_data = scaled_data[0 : training_data_len, :]
    
    #split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train

def train_knn_model(x_train, y_train):
    ''' Given the input features (x_train) and the target variable (y_train)
        Train the model using x_train and y_train
        Return the trained KNN model
    '''
    #convert the x_train and y_train to numpy arrays
    knn_model = KNeighborsRegressor(n_neighbors = 5)
    knn_model.fit(x_train, y_train)
    return knn_model

def make_predictions(knn_model, scaled_data, training_data_len, scaler):
    ''' Given a knn model, scaled data, an int for training data length,
        and a scaler
        Make predictions using the knn model
        Return the predictions and the unscaled actual values
    '''
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = scaled_data[training_data_len - 60:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])
    x_test = np.array(x_test)
    
    # Fit the model with training data
    predictions = knn_model.predict(x_test)
    
    # Make predictions on the test data
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_unscaled = scaler.inverse_transform(
        np.array(y_test[60:]).reshape(-1, 1))
    return predictions, y_test_unscaled

def calculate_rmse(y_test, predictions):
    ''' Given unscaled actual values and predictions
        Calculate the Root Mean Squared Error (RMSE) between the 
        actual and predicted values
        Return the RMSE
    '''
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse

def visualize_data(train, valid, stock_ticker, predictions):
    ''' Given the training data, vaildation data, a string for 
        stock ticker, and predictions
        Visualize the training and validation data along with the predictions
    '''
    plt.figure(figsize = (16, 8))
    plt.title(f'Model for {stock_ticker}', fontsize = 18)
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price ($)', fontsize = 18)
    plt.plot(train['Date'], train['Close'], label = 'Train')
    plt.plot(valid['Date'], valid['Close'], label = 'Actual', alpha = 0.7)
    plt.plot(valid['Date'], predictions, label = 'Prediction', alpha = 0.7)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.show()

def main():
    # Prompt user inputs
    stock_ticker = input(
        "Enter the stock ticker symbol (e.g., 'AAPL' for Apple Inc.): ")
    start_date = input("Enter the start date for the analysis (YYYY-MM-DD): ")

    # Fetch data
    df_history = fetch_stock_data(stock_ticker, start_date)

    # Plot data
    plot_stock_data(df_history, stock_ticker)

    # Prepare data for modeling
    data = df_history['Close']
    
    # Reshape to 2D array for scaling
    dataset = data.values.reshape(-1, 1)
    
    # Get the number of rows to train the model on, with 80% of the data
    training_data_len = math.ceil(len(dataset) * 0.8)
    scaled_data, scaler = scale_data(dataset)
    x_train, y_train = create_train_data(scaled_data, training_data_len)

    # Train KNN model
    knn_model = train_knn_model(x_train, y_train)

    # Make predictions
    predictions, y_test_unscaled = make_predictions(
        knn_model, scaled_data, training_data_len, scaler)

    # Calculate RMSE
    rmse = calculate_rmse(y_test_unscaled, predictions)
    print(f'The RMSE of the KNN predictions is: {rmse}')

    # Visualize data
    train = df_history[:training_data_len].copy()
    valid = df_history[training_data_len:].copy()
    visualize_data(train, valid, stock_ticker, predictions)

if __name__ == "__main__":
    main()
