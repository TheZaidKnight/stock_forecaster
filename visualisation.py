import plotly.graph_objects as go
import pandas as pd

def create_forecast_chart(historical_df, forecasts, ticker):
    """
    Creates a Plotly candlestick chart with forecast overlays.
    
    Args:
        historical_df (pd.DataFrame): DataFrame with OHLC data.
        forecasts (dict): A dictionary where keys are model names 
                          and values are forecast arrays.
                          e.g., {'ARIMA': [val1, val2], 'LSTM': [val3, val4]}
        ticker (str): The stock ticker symbol.
    """
    fig = go.Figure()

    # 1. Add historical data (candlestick)
    fig.add_trace(go.Candlestick(x=historical_df['Date'],
                                 open=historical_df['Open'],
                                 high=historical_df['High'],
                                 low=historical_df['Low'],
                                 close=historical_df['Close'],
                                 name='Historical Price'))

    # 2. Add forecast data (lines)
    last_date = historical_df['Date'].iloc[-1]
    forecast_dates = pd.to_datetime(pd.date_range(start=last_date, periods=len(next(iter(forecasts.values()))) + 1).tolist()[1:])


    for model_name, forecast_values in forecasts.items():
        fig.add_trace(go.Scatter(x=forecast_dates, 
                                 y=forecast_values, 
                                 mode='lines',
                                 name=f'{model_name} Forecast'))

    fig.update_layout(
        title=f'Stock Price Forecast for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        legend_title='Legend'
    )
    
    # Convert figure to HTML for embedding in Flask
    chart_html = fig.to_html(full_html=False)
    return chart_html