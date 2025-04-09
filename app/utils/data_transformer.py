"""
Data Transformation Module

This module provides functions to transform and manipulate stock data,
including resampling time series data to different frequencies.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from app.utils.stock_repository import get_stock_data, StockDataNotFoundError, InvalidFilterError
from app.models import DataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataTransformationError(Exception):
    """Exception raised when data transformation fails."""
    pass

def convert_to_dataframe(stock_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert stock data from list of dictionaries to a Pandas DataFrame.
    
    Args:
        stock_data: List of dictionaries containing stock data
    
    Returns:
        Pandas DataFrame with stock data
    
    Raises:
        DataTransformationError: If conversion fails
    """
    try:
        if not stock_data:
            raise DataTransformationError("No stock data provided")
        
        # Create DataFrame from stock data
        df = pd.DataFrame(stock_data)
        
        # Set date as index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
        
        # Keep only OHLCV columns and ticker
        ohlcv_columns = ['ticker', 'open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in ohlcv_columns if col in df.columns]
        
        if len(available_columns) < 2:  # At least ticker and one OHLCV column
            raise DataTransformationError("Insufficient data columns for transformation")
        
        return df[available_columns]
    
    except Exception as e:
        logger.error(f"Error converting stock data to DataFrame: {e}")
        raise DataTransformationError(f"Failed to convert stock data to DataFrame: {e}")

def get_and_transform_stock_data(
    ticker: str,
    start_date: Optional[Union[datetime, str]] = None,
    end_date: Optional[Union[datetime, str]] = None,
    source: Optional[Union[DataSource, str]] = None
) -> pd.DataFrame:
    """
    Retrieve stock data and convert it to a Pandas DataFrame.
    
    Args:
        ticker: The stock ticker symbol
        start_date: Start date for the data range
        end_date: End date for the data range
        source: Data source to filter by
    
    Returns:
        Pandas DataFrame with stock data
    
    Raises:
        StockDataNotFoundError: If no data is found
        InvalidFilterError: If invalid filter parameters are provided
        DataTransformationError: If transformation fails
    """
    # Get stock data from repository
    # If start_date and end_date are None, it will retrieve all available data
    stock_data = get_stock_data(ticker, start_date, end_date, source)
    
    # Convert to DataFrame
    return convert_to_dataframe(stock_data)

def fill_missing_dates(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """
    Fill missing dates in the DataFrame with NaN values.
    
    Args:
        df: Pandas DataFrame with stock data
        freq: Frequency for date range ('D' for daily, 'W' for weekly, etc.)
    
    Returns:
        DataFrame with missing dates filled
    """
    if df.empty:
        return df
    
    # Create a complete date range
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    
    # Reindex the DataFrame to include all dates
    df_reindexed = df.reindex(date_range)
    
    # Ensure ticker column is filled
    if 'ticker' in df.columns and df['ticker'].nunique() == 1:
        df_reindexed['ticker'] = df['ticker'].iloc[0]
    
    return df_reindexed

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different frequency.
    
    Args:
        df: Pandas DataFrame with stock data
        freq: Target frequency ('W' for weekly, 'ME' for monthly, 'YE' for yearly)
    
    Returns:
        Resampled DataFrame
    
    Raises:
        DataTransformationError: If resampling fails
    """
    try:
        if df.empty:
            return df
        
        # Check if we have the necessary columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for proper OHLCV resampling: {missing_columns}")
        
        # Create a copy to avoid modifying the original
        resampled = pd.DataFrame()
        
        # Preserve ticker column if it exists
        if 'ticker' in df.columns:
            resampled['ticker'] = df['ticker'].iloc[0] if df['ticker'].nunique() == 1 else 'MULTIPLE'
        
        # Resample with proper OHLCV aggregation
        if 'open' in df.columns:
            resampled['open'] = df['open'].resample(freq).first()
        
        if 'high' in df.columns:
            resampled['high'] = df['high'].resample(freq).max()
        
        if 'low' in df.columns:
            resampled['low'] = df['low'].resample(freq).min()
        
        if 'close' in df.columns:
            resampled['close'] = df['close'].resample(freq).last()
        
        if 'volume' in df.columns:
            resampled['volume'] = df['volume'].resample(freq).sum()
        
        return resampled
    
    except Exception as e:
        logger.error(f"Error resampling data: {e}")
        raise DataTransformationError(f"Failed to resample data: {e}")

def normalize_data(df: pd.DataFrame, method: str = 'min-max') -> pd.DataFrame:
    """
    Normalize numerical columns in the DataFrame.
    
    Args:
        df: Pandas DataFrame with stock data
        method: Normalization method ('min-max' or 'z-score')
    
    Returns:
        DataFrame with normalized values
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    normalized = df.copy()
    
    # Identify numerical columns to normalize (exclude ticker)
    num_columns = [col for col in df.columns if col != 'ticker' and pd.api.types.is_numeric_dtype(df[col])]
    
    if method == 'min-max':
        for col in num_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:  # Avoid division by zero
                normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    elif method == 'z-score':
        for col in num_columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Avoid division by zero
                normalized[col] = (df[col] - mean) / std
    
    return normalized

def calculate_returns(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    Calculate returns over a specified period.
    
    Args:
        df: Pandas DataFrame with stock data
        period: Number of periods for return calculation
    
    Returns:
        DataFrame with an additional 'returns' column
    """
    if df.empty or 'close' not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Calculate returns
    result['returns'] = df['close'].pct_change(period)
    
    return result

def get_resampled_stock_data(
    ticker: str,
    freq: str,
    start_date: Optional[Union[datetime, str]] = None,
    end_date: Optional[Union[datetime, str]] = None,
    source: Optional[Union[DataSource, str]] = None,
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    Get stock data resampled to the specified frequency.
    
    Args:
        ticker: The stock ticker symbol
        freq: Target frequency ('W' for weekly, 'ME' for monthly, 'YE' for yearly)
        start_date: Start date for the data range
        end_date: End date for the data range
        source: Data source to filter by
        fill_missing: Whether to fill missing dates before resampling
    
    Returns:
        Resampled DataFrame
    
    Raises:
        StockDataNotFoundError: If no data is found
        InvalidFilterError: If invalid filter parameters are provided
        DataTransformationError: If transformation fails
    """
    # Get stock data as DataFrame
    df = get_and_transform_stock_data(ticker, start_date, end_date, source)
    
    # Fill missing dates if requested
    if fill_missing:
        df = fill_missing_dates(df)
    
    # Resample to target frequency
    return resample_ohlcv(df, freq)
