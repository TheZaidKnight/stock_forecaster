# Configuration file for Stock Forecaster
import os

class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MONGODB_URL = os.environ.get('MONGODB_URL') or 'mongodb://localhost:27017/'
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY') or 'c8f0485fa7084f5f97232c6591e4c9d1'
    
    # Model parameters
    ARIMA_MAX_P = 5
    ARIMA_MAX_D = 2
    ARIMA_MAX_Q = 5
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 16
    
    # Data limits
    MIN_HISTORICAL_DAYS = 30
    MAX_HISTORICAL_DAYS = 365
    MIN_FORECAST_DAYS = 1
    MAX_FORECAST_DAYS = 30

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    MONGODB_URL = 'mongodb://localhost:27017/test_stock_data'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
