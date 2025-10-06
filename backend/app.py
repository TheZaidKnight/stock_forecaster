from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from data_pipeline import process_and_store_data
from models import get_arima_forecast, get_lstm_forecast, get_model_metrics
from visualisation import create_forecast_chart
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        ticker = request.form.get('ticker', '').upper().strip()
        days_history = int(request.form.get('days_history', 90))
        forecast_horizon = int(request.form.get('forecast_horizon', 7))
        horizon_unit = request.form.get('horizon_unit', 'days')
        
        # Input validation
        if not ticker:
            flash('Please enter a valid stock ticker symbol.', 'error')
            return redirect(url_for('index'))
        
        if days_history < 30 or days_history > 365:
            flash('Days of historical data must be between 30 and 365.', 'error')
            return redirect(url_for('index'))
            
        if horizon_unit == 'days':
            if forecast_horizon < 1 or forecast_horizon > 30:
                flash('Forecast horizon (days) must be between 1 and 30.', 'error')
                return redirect(url_for('index'))
        elif horizon_unit == 'hours':
            if forecast_horizon < 1 or forecast_horizon > 72:
                flash('Forecast horizon (hours) must be between 1 and 72.', 'error')
                return redirect(url_for('index'))
        else:
            flash('Invalid horizon unit.', 'error')
            return redirect(url_for('index'))

        # Step 1: Get data
        print(f"Processing forecast for {ticker}...")
        historical_data = process_and_store_data(ticker, days_history, horizon_unit=horizon_unit)
        
        if historical_data is None or historical_data.empty:
            flash(f'Could not fetch data for {ticker}. Please check the ticker symbol and try again.', 'error')
            return redirect(url_for('index'))

        # Ensure we have enough data for forecasting
        if len(historical_data) < 30:
            flash(f'Insufficient historical data for {ticker}. Please increase the days of historical data.', 'error')
            return redirect(url_for('index'))

        # Step 2: Run models
        # Prepare artifacts directory
        from datetime import datetime
        ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        base_dir = f"artifacts/{ts}-{ticker}-{horizon_unit}{forecast_horizon}"

        print("Running ARIMA model...")
        arima_preds = get_arima_forecast(historical_data, forecast_horizon, export_dir=f"{base_dir}/arima")

        print("Running LSTM model...")
        lstm_preds = get_lstm_forecast(historical_data, forecast_horizon, export_dir=f"{base_dir}/lstm")

        # Moving Average baseline (traditional)
        from models import get_moving_average_forecast, get_gru_forecast, get_ensemble_forecast
        print("Running Moving Average baseline...")
        ma_preds = get_moving_average_forecast(historical_data, forecast_horizon)

        print("Running GRU model...")
        gru_preds = get_gru_forecast(historical_data, forecast_horizon, export_dir=f"{base_dir}/gru")
        
        # Step 3: Calculate model performance metrics
        # Use the last portion of historical data for validation
        validation_size = min(10, len(historical_data) // 4)
        validation_data = historical_data['Close'].tail(validation_size).values
        
        # Get predictions for validation period
        arima_val_preds = get_arima_forecast(historical_data.iloc[:-validation_size], validation_size)
        lstm_val_preds = get_lstm_forecast(historical_data.iloc[:-validation_size], validation_size)
        ma_val_preds = get_moving_average_forecast(historical_data.iloc[:-validation_size], validation_size)
        gru_val_preds = get_gru_forecast(historical_data.iloc[:-validation_size], validation_size)
        
        # Calculate metrics
        arima_metrics = get_model_metrics(validation_data, arima_val_preds)
        lstm_metrics = get_model_metrics(validation_data, lstm_val_preds)
        ma_metrics = get_model_metrics(validation_data, ma_val_preds)
        gru_metrics = get_model_metrics(validation_data, gru_val_preds)

        # Ensemble forecast weighted by inverse RMSE
        ensemble_preds = get_ensemble_forecast([
            ('ARIMA', arima_val_preds, arima_metrics),
            ('LSTM', lstm_val_preds, lstm_metrics),
            ('MA', ma_val_preds, ma_metrics),
            ('GRU', gru_val_preds, gru_metrics)
        ], [arima_preds, lstm_preds, ma_preds, gru_preds])
        ensemble_val = get_ensemble_forecast([
            ('ARIMA', arima_val_preds, arima_metrics),
            ('LSTM', lstm_val_preds, lstm_metrics),
            ('MA', ma_val_preds, ma_metrics),
            ('GRU', gru_val_preds, gru_metrics)
        ], None)
        ensemble_metrics = get_model_metrics(validation_data, ensemble_val)
        
        forecasts = {
            'ARIMA': arima_preds,
            'LSTM': lstm_preds,
            'MA': ma_preds,
            'GRU': gru_preds,
            'Ensemble': ensemble_preds
        }
        
        # Step 4: Create visualization
        print("Creating visualization...")
        chart_html = create_forecast_chart(historical_data, forecasts, ticker, horizon_unit=horizon_unit)

        # Prepare data for the results page
        def _last(arr):
            arr_np = np.asarray(arr).reshape(-1)
            return float(arr_np[-1])

        predictions = {
            "ARIMA": f"${_last(arima_preds):.2f}",
            "LSTM": f"${_last(lstm_preds):.2f}",
            "MA": f"${_last(ma_preds):.2f}",
            "GRU": f"${_last(gru_preds):.2f}",
            "Ensemble": f"${_last(ensemble_preds):.2f}"
        }
        
        # Calculate percentage change from current price
        current_price = historical_data['Close'].iloc[-1]
        arima_change = ((_last(arima_preds) - current_price) / current_price) * 100
        lstm_change = ((_last(lstm_preds) - current_price) / current_price) * 100
        ma_change = ((_last(ma_preds) - current_price) / current_price) * 100
        gru_change = ((_last(gru_preds) - current_price) / current_price) * 100
        ensemble_change = ((_last(ensemble_preds) - current_price) / current_price) * 100
        
        prediction_changes = {
            "ARIMA": f"{arima_change:+.2f}%",
            "LSTM": f"{lstm_change:+.2f}%",
            "MA": f"{ma_change:+.2f}%",
            "GRU": f"{gru_change:+.2f}%",
            "Ensemble": f"{ensemble_change:+.2f}%"
        }

        # TODO: persist predictions to DB
        try:
            from data_pipeline import get_db_connection
            db = get_db_connection()
            if db is not None:
                predictions_col = db['predictions']
                # ensure metrics are JSON-serializable floats
                def _float_metrics(d):
                    return {k: float(v) for k, v in d.items()}

                predictions_col.insert_one({
                    'ticker': ticker,
                    'horizon': forecast_horizon,
                    'unit': horizon_unit,
                    'generated_at': pd.Timestamp.utcnow().to_pydatetime(),
                    'current_price': float(current_price),
                    'models': {
                        'ARIMA': {'last': _last(arima_preds), 'metrics': _float_metrics(arima_metrics)},
                        'LSTM': {'last': _last(lstm_preds), 'metrics': _float_metrics(lstm_metrics)},
                        'MA': {'last': _last(ma_preds), 'metrics': _float_metrics(ma_metrics)},
                        'GRU': {'last': _last(gru_preds), 'metrics': _float_metrics(gru_metrics)},
                        'Ensemble': {'last': _last(ensemble_preds), 'metrics': _float_metrics(ensemble_metrics)},
                    }
                })
        except Exception as _e:
            print(f"Warning: Failed to persist predictions: {_e}")

        return render_template('results.html', 
                               ticker=ticker, 
                               chart_html=chart_html,
                               predictions=predictions,
                               prediction_changes=prediction_changes,
                               current_price=f"${current_price:.2f}",
                               arima_metrics=arima_metrics,
                               lstm_metrics=lstm_metrics,
                               ma_metrics=ma_metrics,
                               gru_metrics=gru_metrics,
                               ensemble_metrics=ensemble_metrics,
                               days_history=days_history,
                               forecast_horizon=forecast_horizon,
                               horizon_unit=horizon_unit)
                               
    except ValueError as e:
        flash(f'Invalid input: {str(e)}', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error in forecast: {str(e)}")
        print(traceback.format_exc())
        flash(f'An error occurred while processing your request: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)