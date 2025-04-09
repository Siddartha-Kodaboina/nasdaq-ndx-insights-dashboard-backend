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


def merge_stock_data(
    tickers: List[str],
    column: str = 'close',
    start_date: Optional[Union[datetime, str]] = None,
    end_date: Optional[Union[datetime, str]] = None,
    source: Optional[Union[DataSource, str]] = None,
    fill_method: Optional[str] = 'ffill'
) -> pd.DataFrame:
    """
    Merge data from multiple tickers into a single DataFrame.
    
    Args:
        tickers: List of stock ticker symbols
        column: Column to extract from each ticker's data (default: 'close')
        start_date: Start date for the data range
        end_date: End date for the data range
        source: Data source to filter by
        fill_method: Method to fill missing values after merging ('ffill', 'bfill', None)
    
    Returns:
        DataFrame with merged data, where each column is a ticker
    
    Raises:
        StockDataNotFoundError: If no data is found for any ticker
        DataTransformationError: If merging fails
    """
    try:
        if not tickers:
            raise DataTransformationError("No tickers provided for merging")
        
        merged_df = pd.DataFrame()
        
        for ticker in tickers:
            try:
                # Get data for this ticker
                ticker_df = get_and_transform_stock_data(ticker, start_date, end_date, source)
                
                if ticker_df.empty or column not in ticker_df.columns:
                    logger.warning(f"No {column} data found for {ticker}, skipping")
                    continue
                
                # Extract the specified column and add to merged DataFrame
                merged_df[ticker] = ticker_df[column]
                
            except (StockDataNotFoundError, InvalidFilterError) as e:
                logger.warning(f"Could not retrieve data for {ticker}: {e}")
                continue
        
        if merged_df.empty:
            raise DataTransformationError("Could not retrieve data for any of the provided tickers")
        
        # Fill missing values if requested
        if fill_method:
            merged_df = merged_df.fillna(method=fill_method)
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Error merging stock data: {e}")
        raise DataTransformationError(f"Failed to merge stock data: {e}")


def merge_from_multiple_sources(
    ticker: str,
    sources: List[Union[DataSource, str]],
    start_date: Optional[Union[datetime, str]] = None,
    end_date: Optional[Union[datetime, str]] = None,
    priority_order: bool = True
) -> pd.DataFrame:
    """
    Merge data for a single ticker from multiple sources.
    
    Args:
        ticker: Stock ticker symbol
        sources: List of data sources
        start_date: Start date for the data range
        end_date: End date for the data range
        priority_order: If True, later sources in the list will overwrite earlier ones
                        If False, data will be merged without overwriting
    
    Returns:
        DataFrame with merged data from multiple sources
    
    Raises:
        StockDataNotFoundError: If no data is found for the ticker in any source
        DataTransformationError: If merging fails
    """
    try:
        if not sources:
            raise DataTransformationError("No sources provided for merging")
        
        all_data = []
        source_found = False
        
        for source in sources:
            try:
                # Get data for this source
                source_df = get_and_transform_stock_data(ticker, start_date, end_date, source)
                
                if not source_df.empty:
                    source_found = True
                    source_df['source'] = str(source)  # Add source information
                    all_data.append(source_df)
                
            except (StockDataNotFoundError, InvalidFilterError) as e:
                logger.warning(f"Could not retrieve data for {ticker} from {source}: {e}")
                continue
        
        if not source_found:
            raise StockDataNotFoundError(f"No data found for {ticker} in any of the provided sources")
        
        if len(all_data) == 1:
            return all_data[0].drop(columns=['source'], errors='ignore')
        
        # Combine data from all sources
        combined_df = pd.concat(all_data)
        
        if priority_order:
            # Sort by date and source (in the order provided)
            source_priority = {str(source): i for i, source in enumerate(sources)}
            combined_df['source_priority'] = combined_df['source'].map(source_priority)
            combined_df.sort_values(['source_priority'], inplace=True)
            
            # Keep the last occurrence of each date (highest priority source)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.drop(columns=['source', 'source_priority'], inplace=True)
        else:
            # Remove duplicates, keeping the first occurrence
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df.drop(columns=['source'], inplace=True)
        
        # Sort by date
        combined_df.sort_index(inplace=True)
        
        return combined_df
    
    except Exception as e:
        logger.error(f"Error merging data from multiple sources: {e}")
        raise DataTransformationError(f"Failed to merge data from multiple sources: {e}")


def align_time_series(
    dataframes: List[pd.DataFrame],
    method: str = 'inner',
    fill_method: Optional[str] = None
) -> List[pd.DataFrame]:
    """
    Align multiple time series to a common date range.
    
    Args:
        dataframes: List of DataFrames to align
        method: Join method ('inner', 'outer', 'left', 'right')
        fill_method: Method to fill missing values after alignment ('ffill', 'bfill', None)
    
    Returns:
        List of aligned DataFrames
    
    Raises:
        DataTransformationError: If alignment fails
    """
    try:
        if not dataframes:
            return []
        
        if len(dataframes) == 1:
            return dataframes
        
        # Filter out empty DataFrames
        valid_dfs = [df for df in dataframes if not df.empty]
        
        if not valid_dfs:
            return []
        
        # Get all unique dates from all DataFrames based on the join method
        if method == 'inner':
            # Find common dates across all DataFrames
            common_dates = valid_dfs[0].index
            for df in valid_dfs[1:]:
                common_dates = common_dates.intersection(df.index)
            
            # Reindex all DataFrames to the common dates
            aligned_dfs = [df.reindex(common_dates) for df in valid_dfs]
        
        elif method == 'outer':
            # Find union of all dates
            all_dates = valid_dfs[0].index
            for df in valid_dfs[1:]:
                all_dates = all_dates.union(df.index)
            
            # Reindex all DataFrames to include all dates
            aligned_dfs = [df.reindex(all_dates) for df in valid_dfs]
        
        elif method == 'left':
            # Use the first DataFrame's dates
            reference_dates = valid_dfs[0].index
            aligned_dfs = [valid_dfs[0]] + [df.reindex(reference_dates) for df in valid_dfs[1:]]
        
        elif method == 'right':
            # Use the last DataFrame's dates
            reference_dates = valid_dfs[-1].index
            aligned_dfs = [df.reindex(reference_dates) for df in valid_dfs[:-1]] + [valid_dfs[-1]]
        
        else:
            raise ValueError(f"Invalid alignment method: {method}")
        
        # Fill missing values if requested
        if fill_method:
            if fill_method == 'ffill':
                aligned_dfs = [df.ffill() for df in aligned_dfs]
            elif fill_method == 'bfill':
                aligned_dfs = [df.bfill() for df in aligned_dfs]
            else:
                logger.warning(f"Unknown fill method: {fill_method}, using ffill")
                aligned_dfs = [df.ffill() for df in aligned_dfs]
        
        return aligned_dfs
    
    except Exception as e:
        logger.error(f"Error aligning time series: {e}")
        raise DataTransformationError(f"Failed to align time series: {e}")


def resample_and_align(
    dataframes: List[pd.DataFrame],
    freq: str,
    method: str = 'inner',
    fill_method: Optional[str] = None
) -> List[pd.DataFrame]:
    """
    Resample multiple DataFrames to a common frequency and align dates.
    
    Args:
        dataframes: List of DataFrames to resample and align
        freq: Target frequency ('W' for weekly, 'ME' for monthly, 'YE' for yearly)
        method: Join method ('inner', 'outer', 'left', 'right')
        fill_method: Method to fill missing values after alignment ('ffill', 'bfill', None)
    
    Returns:
        List of resampled and aligned DataFrames
    
    Raises:
        DataTransformationError: If resampling or alignment fails
    """
    try:
        if not dataframes:
            return []
        
        # Resample each DataFrame
        resampled_dfs = []
        for df in dataframes:
            if not df.empty:
                # For DataFrames with only 'close' column, handle differently
                if set(df.columns) - {'ticker'} == {'close'}:
                    # Simple resampling for DataFrames with only close prices
                    resampled = df.copy()
                    if 'ticker' in df.columns:
                        ticker_val = df['ticker'].iloc[0] if df['ticker'].nunique() == 1 else 'MULTIPLE'
                        resampled_close = df['close'].resample(freq).last()
                        resampled = pd.DataFrame({'ticker': ticker_val, 'close': resampled_close})
                        resampled.index = resampled_close.index
                    else:
                        resampled = pd.DataFrame({'close': df['close'].resample(freq).last()})
                    resampled_dfs.append(resampled)
                else:
                    # Use the full OHLCV resampling for complete DataFrames
                    resampled_dfs.append(resample_ohlcv(df, freq))
            else:
                resampled_dfs.append(df)  # Keep empty DataFrames as is
        
        # Align the resampled DataFrames
        return align_time_series(resampled_dfs, method, fill_method)
    
    except Exception as e:
        logger.error(f"Error resampling and aligning DataFrames: {e}")
        raise DataTransformationError(f"Failed to resample and align DataFrames: {e}")


def normalize_for_comparison(
    dataframes: List[pd.DataFrame],
    column: str = 'close',
    method: str = 'percent_change',
    base_date: Optional[Union[datetime, str]] = None
) -> List[pd.DataFrame]:
    """
    Normalize multiple stocks for direct comparison.
    
    Args:
        dataframes: List of DataFrames to normalize
        column: Column to normalize (default: 'close')
        method: Normalization method ('percent_change', 'first_value', 'min-max', 'z-score')
        base_date: Base date for percent change calculation (default: first date in each DataFrame)
    
    Returns:
        List of DataFrames with normalized values
    
    Raises:
        DataTransformationError: If normalization fails
    """
    try:
        if not dataframes:
            return []
        
        normalized_dfs = []
        
        for df in dataframes:
            if df.empty or column not in df.columns:
                normalized_dfs.append(df)
                continue
            
            normalized_df = df.copy()
            
            if method == 'percent_change':
                # Calculate percent change from base date or first date
                if base_date:
                    # Convert base_date to datetime if it's a string
                    if isinstance(base_date, str):
                        base_date = pd.to_datetime(base_date)
                    
                    # Find the closest date to base_date in the DataFrame
                    if base_date in df.index:
                        base_value = df.loc[base_date, column]
                    else:
                        # Find the closest date after base_date
                        valid_dates = df.index[df.index >= base_date]
                        if len(valid_dates) > 0:
                            closest_date = valid_dates[0]
                            base_value = df.loc[closest_date, column]
                        else:
                            # If no date after base_date, use the first available date
                            base_value = df[column].iloc[0]
                else:
                    # Use the first value as base
                    base_value = df[column].iloc[0]
                
                # Calculate percent change
                if base_value != 0:
                    normalized_df[f'{column}_normalized'] = (df[column] / base_value - 1) * 100
                else:
                    # Handle division by zero
                    normalized_df[f'{column}_normalized'] = np.nan
            
            elif method == 'first_value':
                # Normalize to first value (similar to percent_change but as a ratio instead of percentage)
                first_value = df[column].iloc[0]
                if first_value != 0:
                    normalized_df[f'{column}_normalized'] = df[column] / first_value
                else:
                    normalized_df[f'{column}_normalized'] = np.nan
            
            elif method == 'min-max':
                # Min-max normalization
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val > min_val:
                    normalized_df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
                else:
                    normalized_df[f'{column}_normalized'] = np.nan
            
            elif method == 'z-score':
                # Z-score normalization
                mean = df[column].mean()
                std = df[column].std()
                if std > 0:
                    normalized_df[f'{column}_normalized'] = (df[column] - mean) / std
                else:
                    normalized_df[f'{column}_normalized'] = np.nan
            
            else:
                raise ValueError(f"Invalid normalization method: {method}")
            
            normalized_dfs.append(normalized_df)
        
        return normalized_dfs
    
    except Exception as e:
        logger.error(f"Error normalizing data for comparison: {e}")
        raise DataTransformationError(f"Failed to normalize data for comparison: {e}")
