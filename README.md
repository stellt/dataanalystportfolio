# dataanalystportfolio

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import time

def get_historical_data(ticker, start_date, end_date, interval):
    return yf.download(ticker, start=start_date, end=end_date, interval=interval)

def calculate_moving_averages(data):
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    return data

def prepare_data(data):
    X = data.index.to_series().apply(lambda x: x.hour * 60 + x.minute).values.reshape(-1, 1)
    y = data['Close'].values
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def forecast(model, forecast_window, end_date):
    forecast_dates = pd.date_range(start=end_date, periods=forecast_window, freq='30min')
    forecast_X = forecast_dates.to_series().apply(lambda x: x.hour * 60 + x.minute).values.reshape(-1, 1)
    forecast_y = model.predict(forecast_X)
    return forecast_dates, forecast_y

def plot_forecast(data, forecast_dates, forecast_y):
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Actual')
    plt.plot(forecast_dates, forecast_y, label='Forecast')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    plt.legend(loc='upper left')
    plt.show()

def main():
    ticker = '^GSPC'
    start_date = dt.datetime.now() - dt.timedelta(days=60)
    end_date = dt.datetime.now()
    interval = '30m'
    forecast_window = 30  # 30 minutes

    while True:
        data = get_historical_data(ticker, start_date, end_date, interval)
        if data.empty:
            print("No data available. Exiting.")
            return
        data = calculate_moving_averages(data)
        X, y = prepare_data(data)
        model = train_model(X, y)
        forecast_dates, forecast_y = forecast(model, forecast_window, end_date)
        plot_forecast(data, forecast_dates, forecast_y)
        print(f'Forecast for next {forecast_window} minutes:')
        print(forecast_y)
        time.sleep(1800)  # 1800 seconds = 30 minutes
        end_date = dt.datetime.now()

if __name__ == '__main__':
    main()
