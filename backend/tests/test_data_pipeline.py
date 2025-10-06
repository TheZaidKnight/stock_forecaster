import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import get_historical_data, get_financial_news, process_and_store_data, get_db_connection

class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_ticker = "AAPL"
        self.test_days = 30
        
    @patch('data_pipeline.yf.Ticker')
    def test_get_historical_data_success(self, mock_ticker):
        """Test successful historical data retrieval."""
        # Mock the yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = get_historical_data(self.test_ticker, 3)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn('Close', result.columns)
        
    @patch('data_pipeline.yf.Ticker')
    def test_get_historical_data_empty_response(self, mock_ticker):
        """Test handling of empty response from yfinance."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        result = get_historical_data(self.test_ticker, 30)
        
        self.assertIsNone(result)
        
    @patch('data_pipeline.yf.Ticker')
    def test_get_historical_data_exception(self, mock_ticker):
        """Test handling of exceptions in historical data retrieval."""
        mock_ticker.side_effect = Exception("API Error")
        
        result = get_historical_data(self.test_ticker, 30)
        
        self.assertIsNone(result)
    
    @patch('data_pipeline.requests.get')
    def test_get_news_from_api_success(self, mock_get):
        """Test successful news retrieval from API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Test News Article",
                    "publishedAt": "2023-01-01T00:00:00Z",
                    "url": "http://example.com",
                    "source": {"name": "Test Source"}
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        from data_pipeline import get_news_from_api
        result = get_news_from_api(self.test_ticker)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
    @patch('data_pipeline.requests.get')
    def test_get_news_from_api_failure(self, mock_get):
        """Test handling of API failure."""
        mock_get.side_effect = Exception("API Error")
        
        from data_pipeline import get_news_from_api
        result = get_news_from_api(self.test_ticker)
        
        self.assertEqual(result, [])
    
    @patch('data_pipeline.get_db_connection')
    @patch('data_pipeline.get_historical_data')
    @patch('data_pipeline.get_financial_news')
    def test_process_and_store_data_success(self, mock_news, mock_hist, mock_db):
        """Test successful data processing and storage."""
        # Mock historical data
        mock_hist_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_hist.return_value = mock_hist_data
        
        # Mock news data
        mock_news.return_value = {'2023-01-01': ['Test news']}
        
        # Mock database
        mock_db_instance = MagicMock()
        mock_collection = MagicMock()
        mock_db_instance.__getitem__.return_value = mock_collection
        mock_db.return_value = mock_db_instance
        
        result = process_and_store_data(self.test_ticker, 30)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('7_day_MA', result.columns)
        self.assertIn('Daily_Return', result.columns)
        
    @patch('data_pipeline.get_db_connection')
    @patch('data_pipeline.get_historical_data')
    def test_process_and_store_data_no_db(self, mock_hist, mock_db):
        """Test data processing when database is not available."""
        # Mock historical data
        mock_hist_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_hist.return_value = mock_hist_data
        
        # Mock no database connection
        mock_db.return_value = None
        
        result = process_and_store_data(self.test_ticker, 30)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('7_day_MA', result.columns)

if __name__ == '__main__':
    unittest.main()