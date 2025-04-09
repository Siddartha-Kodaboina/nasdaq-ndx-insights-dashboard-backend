"""
Example script demonstrating how to use the data transformation module.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta

from app.utils.data_transformer import (
    get_and_transform_stock_data,
    get_resampled_stock_data,
    normalize_data,
    calculate_returns,
    convert_to_dataframe,
    resample_ohlcv
)
from app.utils.stock_repository import (
    get_stock_data,
    get_tickers_with_data,
    StockDataNotFoundError
)
from app.models import DataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def populate_test_data():
    """
    Check if we have test data in the database.
    If not, run test_stock_repository.py to populate it.
    """
    try:
        tickers = get_tickers_with_data()
        if not tickers:
            logger.info("No tickers found. Running test_stock_repository.py to populate test data...")
            import subprocess
            subprocess.run(["python", "test_stock_repository.py"], check=True)
            logger.info("Test data populated successfully.")
        else:
            logger.info(f"Found {len(tickers)} tickers in the database.")
    except Exception as e:
        logger.error(f"Error checking or populating test data: {e}")
        raise

def example_transform_stock_data():
    """Example of transforming stock data to a DataFrame."""
    try:
        # Get available tickers
        tickers = get_tickers_with_data()
        if not tickers:
            logger.warning("No tickers found in the database.")
            return
        
        # Use the first ticker for demonstration
        ticker = tickers[0]
        logger.info(f"Using ticker: {ticker}")
        
        # Get stock data as a DataFrame - use a very wide date range to ensure we capture all test data
        start_date = datetime(1970, 1, 1)  # Start from Unix epoch
        end_date = datetime.now()          # Current date
        df = get_and_transform_stock_data(ticker, start_date=start_date, end_date=end_date)
        logger.info(f"Retrieved {df.shape[0]} rows of data for {ticker}")
        logger.info(f"DataFrame sample:\n{df.head()}")
        
        return df
    except StockDataNotFoundError as e:
        logger.warning(f"No stock data found: {e}")
    except Exception as e:
        logger.error(f"Error transforming stock data: {e}")

def example_resample_stock_data(ticker):
    """Example of resampling stock data to different frequencies."""
    try:
        # Since our test data all has the same date (1970-01-01), we'll create a sample DataFrame
        # with unique dates for demonstration purposes
        
        # Create sample data with proper dates
        dates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='D')
        sample_data = []
        
        for i, date in enumerate(dates):
            sample_data.append({
                'ticker': ticker,
                'date': date,
                'open': 100 + i * 0.1,
                'high': 105 + i * 0.1,
                'low': 95 + i * 0.1,
                'close': 102 + i * 0.1,
                'volume': 1000000 + i * 10000
            })
        
        # Convert to DataFrame
        df = convert_to_dataframe(sample_data)
        logger.info(f"Created sample DataFrame with {len(df)} unique dates for resampling demonstration")
        logger.info(f"Sample data:\n{df.head()}")
        
        # Resample to different frequencies
        weekly_df = resample_ohlcv(df, 'W')
        monthly_df = resample_ohlcv(df, 'ME')
        yearly_df = resample_ohlcv(df, 'YE')
        
        logger.info(f"Weekly resampled data ({weekly_df.shape[0]} rows):\n{weekly_df.head()}")
        logger.info(f"Monthly resampled data ({monthly_df.shape[0]} rows):\n{monthly_df.head()}")
        logger.info(f"Yearly resampled data ({yearly_df.shape[0]} rows):\n{yearly_df}")
        
        return weekly_df, monthly_df, yearly_df
    except StockDataNotFoundError as e:
        logger.warning(f"No stock data found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error resampling stock data: {e}")
        return None

def example_normalize_and_calculate_returns(df):
    """Example of normalizing data and calculating returns."""
    if df is None or df.empty:
        logger.warning("No data provided for normalization and returns calculation.")
        return
    
    try:
        # Normalize data using min-max scaling
        normalized_df = normalize_data(df, 'min-max')
        logger.info(f"Normalized data (min-max):\n{normalized_df.head()}")
        
        # Calculate returns
        returns_df = calculate_returns(df)
        logger.info(f"Returns data:\n{returns_df.head()}")
        
        return normalized_df, returns_df
    except Exception as e:
        logger.error(f"Error normalizing data or calculating returns: {e}")
        return None

def main():
    """Run all examples."""
    logger.info("Starting data transformation examples...")
    
    # Make sure we have test data
    populate_test_data()
    
    # Example 1: Transform stock data to DataFrame
    df = example_transform_stock_data()
    
    if df is not None and not df.empty:
        # Get ticker from DataFrame
        ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else None
        
        if ticker:
            # Example 3: Normalize data and calculate returns
            normalized_results = example_normalize_and_calculate_returns(df)
            
            # Example 2: Resample stock data
            resampled_results = example_resample_stock_data(ticker)
            
            # Show successful completion of all examples
            if normalized_results and resampled_results:
                logger.info("Successfully demonstrated all data transformation features!")
    
    logger.info("All examples completed!")

if __name__ == "__main__":
    main()
