# Zaid-Masood_22i-8793 (Modified for DB Integration)
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import feedparser
from dateutil import parser as date_parser
from pymongo import MongoClient

# You must get a free API key from https://newsapi.org/
NEWS_API_KEY = "c8f0485fa7084f5f97232c6591e4c9d1"

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the local MongoDB server."""
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        # Test the connection
        client.server_info()
        db = client['stock_data'] # Database named 'stock_data'
        print("Successfully connected to MongoDB")
        return db
    except Exception as e:
        print(f"Warning: Could not connect to MongoDB: {e}")
        print("Running in offline mode - data will not be persisted")
        return None

def get_news_from_api(ticker_symbol):
    """
    Fetch news for the given ticker from NewsAPI.org.
    """
    print(f"Trying News API for {ticker_symbol}...")
    try:
        query = ticker_symbol
        if ticker_symbol == 'BTC-USD':
            query = 'Bitcoin'
        
        url = (
            f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["articles"]:
            news_list = []
            for article in data["articles"][:20]:
                if 'title' in article and 'publishedAt' in article and article["title"].strip():
                    headline = article["title"]
                    # Use the actual publication date
                    pub_date = date_parser.parse(article["publishedAt"]).strftime('%Y-%m-%d')
                    news_list.append({
                        'date': pub_date,
                        'headline': headline,
                        'link': article.get('url', 'N/A'),
                        'publisher': article.get('source', {}).get('name', 'N/A')
                    })
            print(f"Successfully got {len(news_list)} articles from News API.")
            return news_list
    except Exception as e:
        print(f"Error fetching from News API: {e}")
    return []

def get_news_from_marketwatch_rss(ticker_symbol):
    """
    Fetches news from the MarketWatch RSS feed.
    """
    news_list = []
    rss_url = f"https://feeds.marketwatch.com/marketwatch/topstories/"
    
    try:
        print(f"Trying MarketWatch RSS feed...")
        feed = feedparser.parse(rss_url)
        
        if feed.entries:
            for entry in feed.entries:
                title = entry.title if hasattr(entry, 'title') else 'No title available'
                link = entry.link if hasattr(entry, 'link') else 'N/A'
                
                # Check for published date to make sure it's recent
                if 'published' in entry:
                    published_date = date_parser.parse(entry.published).strftime('%Y-%m-%d')
                else:
                    published_date = datetime.now().strftime('%Y-%m-%d')
                
                # Modified relevance check for crypto tickers
                relevance_check = ticker_symbol.upper() in title.upper()
                if ticker_symbol == 'BTC-USD':
                    relevance_check = any(word in title for word in ['Bitcoin', 'BTC', 'Crypto'])

                if title and title.strip() and relevance_check:
                    news_list.append({
                        'date': published_date,
                        'headline': title.strip(),
                        'link': link,
                        'publisher': "MarketWatch",
                        'publish_time': entry.published if hasattr(entry, 'published') else ''
                    })
    except Exception as e:
        print(f"Error fetching from MarketWatch RSS: {e}")
    
    print(f"Successfully got {len(news_list)} articles from MarketWatch RSS.")
    return news_list

def get_financial_news(ticker_symbol):
    """
    Main news fetching function with NewsAPI and MarketWatch RSS.
    """
    print(f"Fetching news for {ticker_symbol}...")
    news_list = []
    
    # Strategy 1: Dedicated News API (most reliable)
    api_news = get_news_from_api(ticker_symbol)
    if api_news:
        news_list.extend(api_news)
    
    # Strategy 2: MarketWatch RSS feed
    marketwatch_news = get_news_from_marketwatch_rss(ticker_symbol)
    if marketwatch_news:
        news_list.extend(marketwatch_news)
    
    # Aggregate news by date and remove duplicates
    news_by_date = {}
    for news_item in news_list:
        date_str = news_item['date']
        headline = news_item['headline'].strip()
        if date_str not in news_by_date:
            news_by_date[date_str] = set()
        news_by_date[date_str].add(headline)

    # Convert sets back to lists for final processing
    for date, headlines in news_by_date.items():
        news_by_date[date] = list(headlines)

    # Fallback if no news is found
    if not news_by_date:
        current_date = datetime.now().strftime('%Y-%m-%d')
        news_by_date[current_date] = [f'No recent news available for {ticker_symbol}']

    total_articles = sum(len(v) for v in news_by_date.values())
    print(f"Final result: {total_articles} unique news articles for {ticker_symbol}")
    return news_by_date

def get_historical_data(ticker_symbol: str, days: int, *, interval: str = '1d') -> pd.DataFrame:
    """
    Fetches historical stock or crypto data for a specified number of latest days.
    """
    try:
        print(f"Fetching {days} days of historical data for {ticker_symbol}...")
        
        # Fetch enough data to calculate a 7-day MA
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days + 10)).strftime('%Y-%m-%d') # Add buffer for non-trading days
        
        ticker = yf.Ticker(ticker_symbol)
        # Use interval for intraday (e.g., '60m') or daily ('1d')
        hist_data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if hist_data.empty:
            print(f"No data returned for {ticker_symbol}")
            return None
        
        # Trim the data to the user's requested number of days
        # For intraday, days approximates number of periods; otherwise use tail(days)
        hist_data = hist_data.tail(days)
        
        # Validate data quality
        if len(hist_data) < 10:
            print(f"Insufficient data for {ticker_symbol}: only {len(hist_data)} days available")
            return None
            
        # Check for missing values
        missing_values = hist_data.isnull().sum().sum()
        if missing_values > 0:
            print(f"Warning: {missing_values} missing values found in data for {ticker_symbol}")
            hist_data = hist_data.fillna(method='ffill')  # Forward fill missing values
        
        print(f"Successfully fetched {len(hist_data)} days of data for {ticker_symbol}")
        return hist_data
        
    except Exception as e:
        print(f"Error fetching historical data for {ticker_symbol}: {e}")
        return None


# --- Main Data Processing and Storage Function ---
def process_and_store_data(ticker_symbol, days, *, horizon_unit: str = 'days'):
    """
    Fetches, processes, and stores all data in MongoDB.
    """
    print(f"\n{'='*60}\nProcessing data for {ticker_symbol}\n{'='*60}")
    
    db = get_db_connection()
    
    # 1. Fetch Historical Data
    interval = '1d'
    if horizon_unit == 'hours':
        # Use hourly data for hourly forecasts
        interval = '60m'
    historical_df = get_historical_data(ticker_symbol, days, interval=interval)
    if historical_df is None or historical_df.empty:
        print("Could not retrieve historical data. Exiting.")
        return None

    # Calculate technical indicators (rename to be generic for intraday)
    window = 7
    historical_df['7_day_MA'] = historical_df['Close'].rolling(window=window).mean()
    historical_df['Daily_Return'] = historical_df['Close'].pct_change()
    historical_df['7_day_Volatility'] = historical_df['Daily_Return'].rolling(window=window).std()
    
    # Fill any remaining NaN values
    historical_df = historical_df.fillna(method='ffill').fillna(method='bfill')
    
    # 2. Fetch News Data
    news_by_date = get_financial_news(ticker_symbol)

    # 3. Store Data in MongoDB (if connection available)
    if db is not None:
        try:
            historical_prices_collection = db['historical_prices']
            news_articles_collection = db['news_articles']

            # Store historical data
            historical_df_copy = historical_df.copy()
            historical_df_copy.reset_index(inplace=True)
            records = historical_df_copy.to_dict('records')
            
            for record in records:
                record['ticker'] = ticker_symbol
                # Use date and ticker as a unique identifier to avoid duplicates
                historical_prices_collection.update_one(
                    {'Date': record['Date'], 'ticker': ticker_symbol},
                    {'$set': record},
                    upsert=True
                )
            print(f"Stored {len(records)} historical price points in MongoDB.")

            # Store news articles
            news_records = []
            for date, headlines in news_by_date.items():
                for headline in headlines:
                    news_records.append({
                        'date': datetime.strptime(date, '%Y-%m-%d'),
                        'headline': headline,
                        'ticker': ticker_symbol
                    })
            
            if news_records:
                # Delete old news for the ticker to avoid stale data
                news_articles_collection.delete_many({'ticker': ticker_symbol})
                news_articles_collection.insert_many(news_records)
                print(f"Stored {len(news_records)} news articles in MongoDB.")
                
        except Exception as e:
            print(f"Warning: Could not store data in MongoDB: {e}")
            print("Continuing without database storage...")
    else:
        print("Running in offline mode - data not stored in database")
        
    return historical_df # Return dataframe for further processing

if __name__ == '__main__':
    # This part can be moved to app.py later
    ticker = "AAPL"
    days = 20
    process_and_store_data(ticker, days)