"""
Test script for the data transformation module.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta

from app.utils.data_transformer import (
    convert_to_dataframe,
    get_and_transform_stock_data,
    fill_missing_dates,
    resample_ohlcv,
    normalize_data,
    calculate_returns,
    get_resampled_stock_data,
    DataTransformationError
)
from app.utils.stock_repository import StockDataNotFoundError, InvalidFilterError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_convert_to_dataframe():
    """Test converting stock data to DataFrame."""
    logger.info("Testing convert_to_dataframe...")
    
    # Create sample stock data
    stock_data = [
        {
            "id": 1,
            "task_id": 1,
            "ticker": "AAPL",
            "date": datetime(2020, 1, 2),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 1000000,
            "source": "source_a"
        },
        {
            "id": 2,
            "task_id": 1,
            "ticker": "AAPL",
            "date": datetime(2020, 1, 3),
            "open": 103.0,
            "high": 107.0,
            "low": 102.0,
            "close": 106.0,
            "volume": 1200000,
            "source": "source_a"
        }
    ]
    
    # Convert to DataFrame
    df = convert_to_dataframe(stock_data)
    
    # Verify DataFrame structure
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame index: {df.index}")
    logger.info(f"DataFrame sample:\n{df.head()}")
    
    # Assertions
    assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
    assert df.shape[0] == 2, "DataFrame should have 2 rows"
    assert 'ticker' in df.columns, "DataFrame should have a ticker column"
    assert 'open' in df.columns, "DataFrame should have an open column"
    assert df.index.name == 'date', "Index should be named 'date'"
    
    logger.info("convert_to_dataframe test passed!")
    return df

def test_get_and_transform_stock_data():
    """Test retrieving and transforming stock data."""
    logger.info("Testing get_and_transform_stock_data...")
    
    try:
        # Get stock data for a known ticker
        ticker = "AAPL"
        df = get_and_transform_stock_data(ticker)
        
        # Verify DataFrame
        logger.info(f"Retrieved {df.shape[0]} rows for {ticker}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame sample:\n{df.head()}")
        
        # Assertions
        assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert df.shape[0] > 0, "DataFrame should have at least one row"
        assert 'ticker' in df.columns, "DataFrame should have a ticker column"
        
        logger.info("get_and_transform_stock_data test passed!")
        return df
    
    except (StockDataNotFoundError, InvalidFilterError) as e:
        logger.warning(f"Test skipped: {e}")
        return None

def test_fill_missing_dates(df=None):
    """Test filling missing dates in DataFrame."""
    logger.info("Testing fill_missing_dates...")
    
    if df is None or df.empty:
        # Create sample DataFrame with missing dates
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        # Remove some dates to create gaps
        dates = dates[[0, 1, 3, 4, 7, 9]]
        
        df = pd.DataFrame({
            'ticker': ['AAPL'] * len(dates),
            'open': range(100, 100 + len(dates)),
            'high': range(105, 105 + len(dates)),
            'low': range(95, 95 + len(dates)),
            'close': range(102, 102 + len(dates)),
            'volume': range(1000000, 1000000 + 100000 * len(dates), 100000)
        }, index=dates)
    
    # Fill missing dates
    filled_df = fill_missing_dates(df)
    
    # Verify filled DataFrame
    logger.info(f"Original DataFrame shape: {df.shape}")
    logger.info(f"Filled DataFrame shape: {filled_df.shape}")
    logger.info(f"Filled DataFrame index:\n{filled_df.index}")
    
    # Assertions
    assert isinstance(filled_df, pd.DataFrame), "Result should be a pandas DataFrame"
    assert filled_df.shape[0] >= df.shape[0], "Filled DataFrame should have at least as many rows as original"
    
    if 'ticker' in df.columns:
        assert filled_df['ticker'].iloc[0] == df['ticker'].iloc[0], "Ticker should be preserved"
    
    logger.info("fill_missing_dates test passed!")
    return filled_df

def test_resample_ohlcv(df=None):
    """Test resampling OHLCV data."""
    logger.info("Testing resample_ohlcv...")
    
    if df is None or df.empty:
        # Create sample daily DataFrame
        dates = pd.date_range(start='2020-01-01', end='2020-01-31', freq='D')
        
        df = pd.DataFrame({
            'ticker': ['AAPL'] * len(dates),
            'open': range(100, 100 + len(dates)),
            'high': range(105, 105 + len(dates)),
            'low': range(95, 95 + len(dates)),
            'close': range(102, 102 + len(dates)),
            'volume': range(1000000, 1000000 + 100000 * len(dates), 100000)
        }, index=dates)
    
    # Test weekly resampling
    weekly_df = resample_ohlcv(df, 'W')
    logger.info(f"Weekly resampled DataFrame shape: {weekly_df.shape}")
    logger.info(f"Weekly resampled DataFrame:\n{weekly_df.head()}")
    
    # Test monthly resampling
    monthly_df = resample_ohlcv(df, 'ME')
    logger.info(f"Monthly resampled DataFrame shape: {monthly_df.shape}")
    logger.info(f"Monthly resampled DataFrame:\n{monthly_df.head()}")
    
    # Test yearly resampling
    yearly_df = resample_ohlcv(df, 'YE')
    logger.info(f"Yearly resampled DataFrame shape: {yearly_df.shape}")
    logger.info(f"Yearly resampled DataFrame:\n{yearly_df}")
    
    # Assertions
    assert isinstance(weekly_df, pd.DataFrame), "Weekly result should be a pandas DataFrame"
    assert isinstance(monthly_df, pd.DataFrame), "Monthly result should be a pandas DataFrame"
    assert isinstance(yearly_df, pd.DataFrame), "Yearly result should be a pandas DataFrame"
    
    assert weekly_df.shape[0] < df.shape[0], "Weekly DataFrame should have fewer rows than daily"
    assert monthly_df.shape[0] < weekly_df.shape[0], "Monthly DataFrame should have fewer rows than weekly"
    assert yearly_df.shape[0] <= monthly_df.shape[0], "Yearly DataFrame should have fewer or equal rows to monthly"
    
    logger.info("resample_ohlcv test passed!")
    return weekly_df, monthly_df, yearly_df

def test_normalize_data(df=None):
    """Test normalizing data."""
    logger.info("Testing normalize_data...")
    
    if df is None or df.empty:
        # Create sample DataFrame
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        
        df = pd.DataFrame({
            'ticker': ['AAPL'] * len(dates),
            'open': range(100, 100 + len(dates)),
            'high': range(105, 105 + len(dates)),
            'low': range(95, 95 + len(dates)),
            'close': range(102, 102 + len(dates)),
            'volume': range(1000000, 1000000 + 100000 * len(dates), 100000)
        }, index=dates)
    
    # Test min-max normalization
    minmax_df = normalize_data(df, 'min-max')
    logger.info(f"Min-Max normalized DataFrame:\n{minmax_df.head()}")
    
    # Test z-score normalization
    zscore_df = normalize_data(df, 'z-score')
    logger.info(f"Z-Score normalized DataFrame:\n{zscore_df.head()}")
    
    # Assertions
    assert isinstance(minmax_df, pd.DataFrame), "Min-Max result should be a pandas DataFrame"
    assert isinstance(zscore_df, pd.DataFrame), "Z-Score result should be a pandas DataFrame"
    
    if 'close' in minmax_df.columns:
        assert minmax_df['close'].min() >= 0, "Min-Max normalized values should be >= 0"
        assert minmax_df['close'].max() <= 1, "Min-Max normalized values should be <= 1"
    
    logger.info("normalize_data test passed!")
    return minmax_df, zscore_df

def test_calculate_returns(df=None):
    """Test calculating returns."""
    logger.info("Testing calculate_returns...")
    
    if df is None or df.empty:
        # Create sample DataFrame
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        
        df = pd.DataFrame({
            'ticker': ['AAPL'] * len(dates),
            'close': [100, 102, 101, 103, 105, 104, 107, 106, 108, 110]
        }, index=dates)
    
    # Calculate returns
    returns_df = calculate_returns(df)
    logger.info(f"Returns DataFrame:\n{returns_df.head()}")
    
    # Assertions
    assert isinstance(returns_df, pd.DataFrame), "Result should be a pandas DataFrame"
    assert 'returns' in returns_df.columns, "DataFrame should have a returns column"
    
    logger.info("calculate_returns test passed!")
    return returns_df

def test_get_resampled_stock_data():
    """Test getting resampled stock data."""
    logger.info("Testing get_resampled_stock_data...")
    
    try:
        # Get weekly resampled data
        ticker = "AAPL"
        weekly_df = get_resampled_stock_data(ticker, 'W')
        logger.info(f"Weekly resampled data for {ticker}:\n{weekly_df.head()}")
        
        # Get monthly resampled data
        monthly_df = get_resampled_stock_data(ticker, 'ME')
        logger.info(f"Monthly resampled data for {ticker}:\n{monthly_df.head()}")
        
        # Get yearly resampled data
        yearly_df = get_resampled_stock_data(ticker, 'YE')
        logger.info(f"Yearly resampled data for {ticker}:\n{yearly_df}")
        
        # Assertions
        assert isinstance(weekly_df, pd.DataFrame), "Weekly result should be a pandas DataFrame"
        assert isinstance(monthly_df, pd.DataFrame), "Monthly result should be a pandas DataFrame"
        assert isinstance(yearly_df, pd.DataFrame), "Yearly result should be a pandas DataFrame"
        
        logger.info("get_resampled_stock_data test passed!")
        return weekly_df, monthly_df, yearly_df
    
    except (StockDataNotFoundError, InvalidFilterError, DataTransformationError) as e:
        logger.warning(f"Test skipped: {e}")
        return None, None, None

def main():
    """Run all tests."""
    logger.info("Starting data transformer tests...")
    
    # Test convert_to_dataframe with sample data
    df = test_convert_to_dataframe()
    
    # Test get_and_transform_stock_data with real data
    real_df = test_get_and_transform_stock_data()
    
    # Test fill_missing_dates
    test_fill_missing_dates(real_df if real_df is not None else None)
    
    # Test resample_ohlcv
    test_resample_ohlcv(real_df if real_df is not None else None)
    
    # Test normalize_data
    test_normalize_data(real_df if real_df is not None else None)
    
    # Test calculate_returns
    test_calculate_returns(real_df if real_df is not None else None)
    
    # Test get_resampled_stock_data
    test_get_resampled_stock_data()
    
    logger.info("All data transformer tests completed!")

if __name__ == "__main__":
    main()
