import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualisation import create_forecast_chart

class TestVisualization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample historical data
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            'Volume': [1000000] * 10,
            '7_day_MA': [104, 104.5, 105, 105.5, 106, 106.5, 107, 107.5, 108, 108.5]
        }, index=pd.date_range('2023-01-01', periods=10))
        
        self.forecasts = {
            'ARIMA': np.array([114, 115, 116]),
            'LSTM': np.array([114.5, 115.5, 116.5])
        }
        self.ticker = 'AAPL'
        
    def test_create_forecast_chart_basic(self):
        """Test basic chart creation."""
        chart_html = create_forecast_chart(self.sample_data, self.forecasts, self.ticker)
        
        self.assertIsInstance(chart_html, str)
        self.assertIn('plotly', chart_html.lower())
        self.assertIn(self.ticker, chart_html)
        
    def test_create_forecast_chart_with_volume(self):
        """Test chart creation with volume data."""
        chart_html = create_forecast_chart(self.sample_data, self.forecasts, self.ticker)
        
        self.assertIsInstance(chart_html, str)
        self.assertIn('Volume', chart_html)
        
    def test_create_forecast_chart_with_moving_average(self):
        """Test chart creation with moving average."""
        chart_html = create_forecast_chart(self.sample_data, self.forecasts, self.ticker)
        
        self.assertIsInstance(chart_html, str)
        self.assertIn('MA', chart_html)
        
    def test_create_forecast_chart_empty_forecasts(self):
        """Test chart creation with empty forecasts."""
        empty_forecasts = {}
        chart_html = create_forecast_chart(self.sample_data, empty_forecasts, self.ticker)
        
        self.assertIsInstance(chart_html, str)
        self.assertIn('plotly', chart_html.lower())
        
    def test_create_forecast_chart_single_model(self):
        """Test chart creation with single model forecast."""
        single_forecast = {'ARIMA': np.array([114, 115, 116])}
        chart_html = create_forecast_chart(self.sample_data, single_forecast, self.ticker)
        
        self.assertIsInstance(chart_html, str)
        self.assertIn('ARIMA', chart_html)
        
    def test_create_forecast_chart_multiple_models(self):
        """Test chart creation with multiple model forecasts."""
        multiple_forecasts = {
            'ARIMA': np.array([114, 115, 116]),
            'LSTM': np.array([114.5, 115.5, 116.5]),
            'SVM': np.array([113.5, 114.5, 115.5])
        }
        chart_html = create_forecast_chart(self.sample_data, multiple_forecasts, self.ticker)
        
        self.assertIsInstance(chart_html, str)
        self.assertIn('Confidence Interval', chart_html)

if __name__ == '__main__':
    unittest.main()
