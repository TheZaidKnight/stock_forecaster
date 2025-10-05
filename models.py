import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_model_metrics(y_true, y_pred):
    """Calculates RMSE, MAE, and MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# --- Traditional Model: ARIMA ---
def get_arima_forecast(historical_data: pd.DataFrame, forecast_steps: int):
    """
    Trains an ARIMA model and returns forecasts.
    """
    # Using 'Close' prices for the time series
    series = historical_data['Close'].values
    
    # Fit ARIMA model (p,d,q) parameters are placeholders, can be optimized
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    
    # Make prediction
    forecast = model_fit.forecast(steps=forecast_steps)
    return forecast

# --- Neural Model: LSTM ---
def create_dataset(dataset, time_step=1):
    """Create sequences for LSTM model."""
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def get_lstm_forecast(historical_data: pd.DataFrame, forecast_steps: int):
    """
    Trains an LSTM model and returns forecasts.
    """
    close_prices = historical_data['Close'].values.reshape(-1,1)
    
    # 1. Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    # 2. Create sequences
    time_step = 10 # Use last 10 days to predict the next
    X, y = create_dataset(scaled_prices, time_step)
    
    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # 3. Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 4. Train the model
    model.fit(X, y, batch_size=1, epochs=5) # epochs can be increased for better accuracy
    
    # 5. Make predictions
    last_sequence = scaled_prices[-time_step:].reshape(1, time_step, 1)
    forecast_scaled = []
    
    current_input = last_sequence
    for _ in range(forecast_steps):
        pred = model.predict(current_input)[0]
        forecast_scaled.append(pred)
        # Reshape pred to be (1, 1, 1) and append to current_input, then trim
        new_input = np.append(current_input[:,1:,:], [[pred]], axis=1)
        current_input = new_input

    # 6. Inverse transform the predictions
    forecast = scaler.inverse_transform(forecast_scaled)
    return forecast.flatten()