# 📈 Stock Price Forecaster

A comprehensive full-stack application that predicts stock prices using both traditional statistical methods (ARIMA) and modern neural networks (LSTM). The application features a robust data pipeline, interactive visualizations, and a user-friendly web interface.

## 🚀 Features

### Core Functionality
- **Dual Forecasting Models**: ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory) neural networks
- **Real-time Data Pipeline**: Fetches historical stock data and financial news using Yahoo Finance and NewsAPI
- **Interactive Visualizations**: Candlestick charts with forecast overlays, volume analysis, and confidence intervals
- **Performance Metrics**: RMSE, MAE, and MAPE evaluation for model comparison
- **MongoDB Integration**: Persistent storage for historical data and news articles

### Technical Highlights
- **Robust Error Handling**: Graceful fallbacks for API failures and insufficient data
- **Adaptive Model Parameters**: Automatic optimization of ARIMA parameters and LSTM architecture
- **Responsive Web Interface**: Modern, mobile-friendly design with real-time feedback
- **Comprehensive Testing**: Unit tests for all major components
- **Production Ready**: Configurable for different environments

## 🏗️ Architecture

```
stock_forecaster/
├── backend/
│   ├── app.py                 # Flask web application
│   ├── data_pipeline.py      # Data fetching and processing
│   ├── models.py             # ARIMA and LSTM forecasting models
│   ├── visualisation.py      # Interactive chart generation
│   ├── templates/            # HTML templates
│   │   ├── index.html        # Main input form
│   │   └── results.html      # Results display page
│   └── tests/               # Test suite
│       ├── test_data_pipeline.py
│       ├── test_models.py
│       └── test_visualization.py
├── requirements.txt          # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- MongoDB (optional, for data persistence)
- NewsAPI key (optional, for enhanced news fetching)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock_forecaster
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB (Optional)**
   ```bash
   # Install MongoDB
   # Windows: Download from https://www.mongodb.com/try/download/community
   # macOS: brew install mongodb-community
   # Linux: sudo apt-get install mongodb
   
   # Start MongoDB service
   mongod
   ```

4. **Configure NewsAPI (Optional)**
   - Get a free API key from [NewsAPI.org](https://newsapi.org/)
   - Update the `NEWS_API_KEY` in `backend/data_pipeline.py`

5. **Run the application**
   ```bash
   cd backend
   python app.py
   ```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`

## 📊 Usage

### Basic Workflow

1. **Enter Stock Information**
   - Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
   - Days of historical data (30-365 days)
   - Forecast horizon (1-30 days)

2. **Generate Forecast**
   - Click "Generate Forecast" to process the data
   - The application will fetch data, train models, and create visualizations

3. **View Results**
   - Interactive candlestick chart with forecast overlays
   - Model performance metrics (RMSE, MAE, MAPE)
   - Price predictions with percentage changes

### Supported Stock Symbols
- **Stocks**: AAPL, MSFT, GOOGL, TSLA, AMZN, etc.
- **Cryptocurrencies**: BTC-USD, ETH-USD, etc.
- **ETFs**: SPY, QQQ, etc.

## 🔬 Models & Methodology

### ARIMA Model
- **Purpose**: Traditional time series forecasting
- **Features**: 
  - Automatic parameter optimization (p, d, q)
  - Stationarity testing with Augmented Dickey-Fuller test
  - AIC-based model selection
- **Best For**: Linear trends and seasonal patterns

### LSTM Model
- **Purpose**: Deep learning-based forecasting
- **Features**:
  - Multi-layer LSTM architecture with dropout
  - Early stopping to prevent overfitting
  - Adaptive training parameters
- **Best For**: Complex non-linear patterns and long-term dependencies

### Model Evaluation
- **RMSE**: Root Mean Square Error - measures prediction accuracy
- **MAE**: Mean Absolute Error - measures average prediction error
- **MAPE**: Mean Absolute Percentage Error - measures relative error

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd backend
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python tests/test_data_pipeline.py
python tests/test_models.py
python tests/test_visualization.py
```

## 🔧 Configuration

### Environment Variables
- `MONGODB_URL`: MongoDB connection string (default: mongodb://localhost:27017/)
- `NEWS_API_KEY`: NewsAPI key for enhanced news fetching
- `FLASK_ENV`: Flask environment (development/production)

### Model Parameters
- **ARIMA**: Automatically optimized, but can be manually adjusted in `models.py`
- **LSTM**: Architecture and training parameters can be modified in `models.py`

## 📈 Performance Considerations

### Data Requirements
- **Minimum**: 30 days of historical data
- **Recommended**: 90+ days for better accuracy
- **Maximum**: 365 days (API limitations)

### Processing Time
- **ARIMA**: ~1-2 seconds
- **LSTM**: ~5-10 seconds (depending on data size)
- **Total**: ~10-15 seconds for complete forecast

### Accuracy Expectations
- **Short-term forecasts (1-7 days)**: Generally more accurate
- **Long-term forecasts (14-30 days)**: Lower accuracy, higher uncertainty
- **Volatile stocks**: Higher prediction errors

## 🚨 Limitations & Disclaimers

### Technical Limitations
- Predictions are based on historical data only
- Market sentiment and external factors not fully captured
- Models may not perform well during extreme market conditions

### Financial Disclaimer
- **This application is for educational and research purposes only**
- **Not intended as financial advice**
- **Past performance does not guarantee future results**
- **Always consult with qualified financial advisors before making investment decisions**

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Include type hints where appropriate

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free stock data API
- **NewsAPI**: For financial news data
- **Plotly**: For interactive visualization capabilities
- **TensorFlow/Keras**: For deep learning framework
- **Statsmodels**: For statistical modeling tools

---

**Remember**: This tool is for educational purposes. Always do your own research and consult with financial professionals before making investment decisions.
