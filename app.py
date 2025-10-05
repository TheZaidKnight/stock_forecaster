from flask import Flask, render_template, request
from data_pipeline import process_and_store_data
from models import get_arima_forecast, get_lstm_forecast
from visualization import create_forecast_chart

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    ticker = request.form.get('ticker').upper()
    days_history = int(request.form.get('days_history'))
    forecast_horizon = int(request.form.get('forecast_horizon'))

    # Step 1: Get data
    # In a real app, you might fetch from DB, but for simplicity we re-process.
    historical_data = process_and_store_data(ticker, days_history)
    
    if historical_data is None:
        return "Error: Could not fetch data for the given ticker."

    # Step 2: Run models
    arima_preds = get_arima_forecast(historical_data, forecast_horizon)
    lstm_preds = get_lstm_forecast(historical_data, forecast_horizon)
    
    forecasts = {
        'ARIMA': arima_preds,
        'LSTM': lstm_preds
    }
    
    # Step 3: Create visualization
    chart_html = create_forecast_chart(historical_data, forecasts, ticker)

    # Prepare data for the results page
    predictions = {
        "ARIMA": f"${arima_preds[-1]:.2f}",
        "LSTM": f"${lstm_preds[-1]:.2f}"
    }

    return render_template('results.html', 
                           ticker=ticker, 
                           chart_html=chart_html,
                           predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)