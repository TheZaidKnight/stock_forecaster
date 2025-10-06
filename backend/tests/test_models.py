import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_arima_forecast, get_lstm_forecast, get_model_metrics, mean_absolute_percentage_error

class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample historical data
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            'Volume': [1000000] * 10
        }, index=pd.date_range('2023-01-01', periods=10))
        
        self.forecast_steps = 3
        
    def test_mean_absolute_percentage_error(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 110, 120])
        y_pred = np.array([105, 115, 125])
        
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        self.assertIsInstance(mape, float)
        self.assertGreater(mape, 0)
        
    def test_get_model_metrics(self):
        """Test model metrics calculation."""
        y_true = np.array([100, 110, 120])
        y_pred = np.array([105, 115, 125])
        
        metrics = get_model_metrics(y_true, y_pred)
        
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertIsInstance(metrics['RMSE'], float)
        self.assertIsInstance(metrics['MAE'], float)
        self.assertIsInstance(metrics['MAPE'], float)
        
    @patch('models.ARIMA')
    def test_get_arima_forecast_success(self, mock_arima):
        """Test successful ARIMA forecasting."""
        # Mock ARIMA model
        mock_model = MagicMock()
        mock_fitted_model = MagicMock()
        mock_fitted_model.forecast.return_value = np.array([114, 115, 116])
        mock_model.fit.return_value = mock_fitted_model
        mock_arima.return_value = mock_model
        
        result = get_arima_forecast(self.sample_data, self.forecast_steps)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.forecast_steps)
        
    def test_get_arima_forecast_insufficient_data(self):
        """Test ARIMA forecasting with insufficient data."""
        small_data = self.sample_data.head(2)
        
        result = get_arima_forecast(small_data, self.forecast_steps)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.forecast_steps)
        
    @patch('models.Sequential')
    def test_get_lstm_forecast_success(self, mock_sequential):
        """Test successful LSTM forecasting."""
        # Mock LSTM model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.9]])
        mock_model.fit.return_value = None
        mock_sequential.return_value = mock_model
        
        result = get_lstm_forecast(self.sample_data, self.forecast_steps)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.forecast_steps)
        
    def test_get_lstm_forecast_insufficient_data(self):
        """Test LSTM forecasting with insufficient data."""
        small_data = self.sample_data.head(5)
        
        result = get_lstm_forecast(small_data, self.forecast_steps)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.forecast_steps)
        
    def test_get_lstm_forecast_exception_handling(self):
        """Test LSTM forecasting exception handling."""
        # Create data that will cause an exception
        problematic_data = pd.DataFrame({
            'Close': [100]  # Only one data point
        })
        
        result = get_lstm_forecast(problematic_data, self.forecast_steps)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.forecast_steps)

if __name__ == '__main__':
    unittest.main()
