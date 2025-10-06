import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def check_stationarity(timeseries):
    """Check if the time series is stationary using Augmented Dickey-Fuller test."""
    result = adfuller(timeseries.dropna())
    return result[1] < 0.05  # p-value < 0.05 means stationary

def find_optimal_arima_params(series, max_p=5, max_d=2, max_q=5):
    """Find optimal ARIMA parameters using AIC."""
    best_aic = float('inf')
    best_params = (1, 1, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                except:
                    continue
    return best_params

def get_model_metrics(y_true, y_pred):
    """Calculates RMSE, MAE, and MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# --- Traditional Model: ARIMA ---
def get_arima_forecast(historical_data: pd.DataFrame, forecast_steps: int):
    """
    Trains an ARIMA model with optimized parameters and returns forecasts.
    """
    try:
        # Using 'Close' prices for the time series
        series = historical_data['Close'].dropna()
        
        if len(series) < 10:
            print("Warning: Insufficient data for ARIMA model")
            return np.full(forecast_steps, series.iloc[-1])
        
        # Check stationarity and difference if needed
        if not check_stationarity(series):
            print("Series is not stationary, applying differencing...")
            series_diff = series.diff().dropna()
        else:
            series_diff = series
        
        # Find optimal parameters
        optimal_params = find_optimal_arima_params(series_diff)
        print(f"Optimal ARIMA parameters: {optimal_params}")
        
        # Fit ARIMA model with optimal parameters
        model = ARIMA(series_diff, order=optimal_params)
        model_fit = model.fit()
        
        # Make prediction
        forecast_diff = model_fit.forecast(steps=forecast_steps)
        
        # Convert back to original scale if we differenced
        if not check_stationarity(series):
            # Integrate the forecast back to original scale
            last_value = series.iloc[-1]
            forecast = []
            for i, diff_val in enumerate(forecast_diff):
                if i == 0:
                    forecast.append(last_value + diff_val)
                else:
                    forecast.append(forecast[i-1] + diff_val)
            forecast = np.array(forecast)
        else:
            forecast = forecast_diff
            
        return forecast
        
    except Exception as e:
        print(f"Error in ARIMA forecasting: {e}")
        # Fallback to simple moving average
        last_price = historical_data['Close'].iloc[-1]
        return np.full(forecast_steps, last_price)

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
    Trains an LSTM model with improved architecture and returns forecasts.
    """
    try:
        close_prices = historical_data['Close'].values.reshape(-1,1)
        
        if len(close_prices) < 20:
            print("Warning: Insufficient data for LSTM model")
            return np.full(forecast_steps, close_prices[-1][0])
        
        # 1. Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_prices = scaler.fit_transform(close_prices)
        
        # 2. Create sequences
        time_step = min(15, len(scaled_prices) // 3)  # Adaptive time step
        X, y = create_dataset(scaled_prices, time_step)
        
        if len(X) < 5:
            print("Warning: Insufficient sequences for LSTM training")
            return np.full(forecast_steps, close_prices[-1][0])
        
        # Reshape input to be [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # 3. Build improved LSTM Model
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        # 4. Train the model with early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        # Use smaller batch size for better convergence
        batch_size = min(16, len(X))
        epochs = min(50, max(10, len(X) // 2))  # Adaptive epochs
        
        model.fit(X, y, 
                 batch_size=batch_size, 
                 epochs=epochs, 
                 callbacks=[early_stopping],
                 verbose=0)
        
        # 5. Make predictions
        last_sequence = scaled_prices[-time_step:].reshape(1, time_step, 1)
        forecast_scaled = []
        
        current_input = last_sequence
        for _ in range(forecast_steps):
            pred = model.predict(current_input, verbose=0)[0]
            forecast_scaled.append(pred)
            # Reshape pred to be (1, 1, 1) and append to current_input, then trim
            new_input = np.append(current_input[:,1:,:], [[pred]], axis=1)
            current_input = new_input

        # 6. Inverse transform the predictions
        forecast = scaler.inverse_transform(forecast_scaled)
        return forecast.flatten()
        
    except Exception as e:
        print(f"Error in LSTM forecasting: {e}")
        # Fallback to simple moving average
        last_price = historical_data['Close'].iloc[-1]
        return np.full(forecast_steps, last_price)


# --- Traditional Model: Moving Average Baseline ---
def get_moving_average_forecast(historical_data: pd.DataFrame, forecast_steps: int, window: int = 7):
    """
    Simple moving average baseline: use the last `window` closes average as constant forecast.
    """
    try:
        closes = historical_data['Close'].dropna().values
        if len(closes) == 0:
            return np.zeros(forecast_steps)
        window = min(window, len(closes))
        avg = np.mean(closes[-window:])
        return np.full(forecast_steps, avg)
    except Exception as _e:
        last_price = historical_data['Close'].iloc[-1]
        return np.full(forecast_steps, last_price)


# --- Neural Model: GRU ---
def get_gru_forecast(historical_data: pd.DataFrame, forecast_steps: int):
    """
    Trains a GRU model and returns forecasts similar to LSTM flow.
    """
    try:
        close_prices = historical_data['Close'].values.reshape(-1,1)
        if len(close_prices) < 20:
            return np.full(forecast_steps, close_prices[-1][0])

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_prices = scaler.fit_transform(close_prices)

        time_step = min(15, len(scaled_prices) // 3)
        X, y = create_dataset(scaled_prices, time_step)
        if len(X) < 5:
            return np.full(forecast_steps, close_prices[-1][0])

        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential()
        model.add(GRU(100, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        batch_size = min(16, len(X))
        epochs = min(50, max(10, len(X) // 2))
        model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping], verbose=0)

        last_sequence = scaled_prices[-time_step:].reshape(1, time_step, 1)
        forecast_scaled = []
        current_input = last_sequence
        for _ in range(forecast_steps):
            pred = model.predict(current_input, verbose=0)[0]
            forecast_scaled.append(pred)
            new_input = np.append(current_input[:,1:,:], [[pred]], axis=1)
            current_input = new_input

        forecast = scaler.inverse_transform(forecast_scaled)
        return forecast.flatten()
    except Exception as _e:
        last_price = historical_data['Close'].iloc[-1]
        return np.full(forecast_steps, last_price)


# --- Ensemble ---
def get_ensemble_forecast(validation_components, future_components):
    """
    Create an ensemble using inverse-RMSE weights.
    - validation_components: list of tuples (name, val_preds_array, metrics_dict)
    - future_components: list of arrays for future forecasts in the same order as validation_components,
      or None to build ensemble for validation horizon.
    Returns a numpy array for the ensemble forecast.
    """
    # Compute weights from RMSE
    rmses = []
    for _name, _val_preds, metrics in validation_components:
        rmse = metrics.get('RMSE', None)
        if rmse is None or rmse <= 0:
            rmse = 1.0
        rmses.append(rmse)

    inv = np.array([1.0 / r if r > 0 else 1.0 for r in rmses], dtype=float)
    weights = inv / np.sum(inv)

    # Choose source arrays (future or validation)
    if future_components is None:
        arrays = [vc[1] for vc in validation_components]
    else:
        arrays = future_components

    # Align lengths and compute weighted sum
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    stacked = np.vstack(arrays)  # shape: (models, steps)
    ensemble = np.average(stacked, axis=0, weights=weights)
    return ensemble