import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_forecast_chart(historical_df, forecasts, ticker, *, horizon_unit: str = 'days'):
    """
    Creates an enhanced Plotly candlestick chart with forecast overlays and technical indicators.
    
    Args:
        historical_df (pd.DataFrame): DataFrame with OHLC data.
        forecasts (dict): A dictionary where keys are model names 
                          and values are forecast arrays.
                          e.g., {'ARIMA': [val1, val2], 'LSTM': [val3, val4]}
        ticker (str): The stock ticker symbol.
    """
    # Create subplots with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Price Forecast', 'Volume'),
        row_width=[0.7, 0.3]
    )

    # 1. Add historical data (candlestick)
    fig.add_trace(go.Candlestick(
        x=historical_df.index,
        open=historical_df['Open'],
        high=historical_df['High'],
        low=historical_df['Low'],
        close=historical_df['Close'],
        name='Historical Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # 2. Add moving average if available
    if '7_day_MA' in historical_df.columns:
        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df['7_day_MA'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='orange', width=2, dash='dash')
        ), row=1, col=1)

    # 3. Add forecast data (lines)
    if forecasts:
        forecast_length = len(next(iter(forecasts.values())))
        last_date = historical_df.index[-1]

        # Create forecast dates depending on horizon unit
        if horizon_unit == 'hours':
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=forecast_length, freq='H')
        else:
            forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_length)
        
        # Color scheme for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model_name, forecast_values) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=forecast_dates, 
                y=forecast_values, 
                mode='lines+markers',
                name=f'{model_name} Forecast',
                line=dict(color=color, width=3),
                marker=dict(size=6)
            ), row=1, col=1)

    # 4. Add volume chart
    if 'Volume' in historical_df.columns:
        fig.add_trace(go.Bar(
            x=historical_df.index,
            y=historical_df['Volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ), row=2, col=1)

    # 5. Update layout with enhanced styling
    fig.update_layout(
        title=dict(
            text=f'📈 {ticker} Stock Price Forecast & Analysis',
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white',
        font=dict(family="Arial", size=12),
        margin=dict(l=60, r=60, t=110, b=90),
        height=600,
        showlegend=True
    )

    # 6. Add annotations for current price
    current_price = historical_df['Close'].iloc[-1]
    fig.add_annotation(
        x=historical_df.index[-1],
        y=current_price,
        text=f"Current: ${current_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#636363",
        borderwidth=1
    )

    # 7. Add forecast confidence intervals (if multiple models)
    if len(forecasts) > 1:
        forecast_values_list = list(forecasts.values())
        forecast_mean = np.mean(forecast_values_list, axis=0)
        forecast_std = np.std(forecast_values_list, axis=0)
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_mean + 2*forecast_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_mean - 2*forecast_std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            name='Confidence Interval',
            hoverinfo='skip'
        ), row=1, col=1)

    # 8. Update axes
    # Apply range breaks to hide non-trading periods for better alignment
    if horizon_unit == 'hours':
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray',
            rangebreaks=[
                dict(bounds=[16, 9], pattern="hour"),  # hide non-market hours roughly
                dict(bounds=["sat", "mon"])            # hide weekends
            ]
        )
    else:
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray',
            rangebreaks=[dict(bounds=["sat", "mon"])]
        )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$.2f')
    
    # Convert figure to HTML for embedding in Flask
    chart_html = fig.to_html(
        full_html=False,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        }
    )
    return chart_html