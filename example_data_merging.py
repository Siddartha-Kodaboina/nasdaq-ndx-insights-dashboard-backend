#!/usr/bin/env python3
"""
Example script to demonstrate data merging features.

This script shows how to use the data merging, alignment, and comparison functions
from the data_transformer module.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.utils.data_transformer import (
    get_and_transform_stock_data,
    merge_stock_data,
    merge_from_multiple_sources,
    align_time_series,
    resample_and_align,
    normalize_for_comparison
)
from app.utils.stock_repository import (
    get_stock_data,
    get_tickers_with_data,
    get_data_sources_for_ticker,
    StockDataNotFoundError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("__main__")

def create_sample_data(tickers, start_date='2020-01-01', num_days=90):
    """
    Create sample data for demonstration purposes.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for the sample data
        num_days: Number of days to generate
        
    Returns:
        Dictionary of DataFrames, one for each ticker
    """
    start = pd.to_datetime(start_date)
    dates = [start + timedelta(days=i) for i in range(num_days)]
    
    dataframes = {}
    
    for ticker in tickers:
        # Create base price and add some randomness
        base_price = 100 + np.random.randint(0, 50)
        
        data = []
        for i, date in enumerate(dates):
            # Skip some dates randomly to simulate missing data
            if np.random.random() > 0.9:  # 10% chance to skip a date
                continue
                
            # Create price with some trend and randomness
            trend = i * 0.1  # Upward trend
            noise = np.random.normal(0, 1)  # Random noise
            
            open_price = base_price + trend + noise
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price + np.random.normal(0, 0.5)
            volume = int(100000 + np.random.normal(0, 10000))
            
            data.append({
                'date': date,
                'ticker': ticker,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        dataframes[ticker] = df
    
    return dataframes

def example_merge_stock_data(tickers):
    """
    Example of merging data from multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Merged DataFrame if successful, None otherwise
    """
    try:
        logger.info(f"Merging data for tickers: {tickers}")
        
        # Try to get real data first
        try:
            merged_df = merge_stock_data(
                tickers=tickers,
                column='close',
                start_date=None,
                end_date=None,
                fill_method='ffill'
            )
            
            logger.info(f"Successfully merged data for {len(merged_df.columns)} tickers")
            logger.info(f"Merged data shape: {merged_df.shape}")
            logger.info(f"Sample of merged data:\n{merged_df.head()}")
            
            return merged_df
            
        except (StockDataNotFoundError, Exception) as e:
            logger.warning(f"Could not merge real data: {e}")
            logger.info("Creating sample data for demonstration...")
            
            # Create sample data
            sample_data = create_sample_data(tickers)
            
            # Manually merge the sample data
            merged_df = pd.DataFrame()
            for ticker, df in sample_data.items():
                if 'close' in df.columns:
                    merged_df[ticker] = df['close']
            
            logger.info(f"Created sample merged data with shape: {merged_df.shape}")
            logger.info(f"Sample of merged data:\n{merged_df.head()}")
            
            return merged_df
            
    except Exception as e:
        logger.error(f"Error in example_merge_stock_data: {e}")
        return None

def example_merge_from_multiple_sources(ticker, sources):
    """
    Example of merging data for a single ticker from multiple sources.
    
    Args:
        ticker: Ticker symbol
        sources: List of data sources
        
    Returns:
        Merged DataFrame if successful, None otherwise
    """
    try:
        logger.info(f"Merging data for {ticker} from sources: {sources}")
        
        # Try to get real data first
        try:
            merged_df = merge_from_multiple_sources(
                ticker=ticker,
                sources=sources,
                start_date=None,
                end_date=None,
                priority_order=True
            )
            
            logger.info(f"Successfully merged data from {len(sources)} sources")
            logger.info(f"Merged data shape: {merged_df.shape}")
            logger.info(f"Sample of merged data:\n{merged_df.head()}")
            
            return merged_df
            
        except (StockDataNotFoundError, Exception) as e:
            logger.warning(f"Could not merge real data from multiple sources: {e}")
            logger.info("Creating sample data for demonstration...")
            
            # Create sample data for each source
            sample_dfs = []
            for i, source in enumerate(sources):
                # Create slightly different data for each source
                df = create_sample_data([ticker], start_date=f"2020-01-{i+1}", num_days=90-i*10)[ticker]
                df['source'] = str(source)
                sample_dfs.append(df)
            
            # Combine the sample data
            combined_df = pd.concat(sample_dfs)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            
            logger.info(f"Created sample merged data with shape: {combined_df.shape}")
            logger.info(f"Sample of merged data:\n{combined_df.head()}")
            
            return combined_df
            
    except Exception as e:
        logger.error(f"Error in example_merge_from_multiple_sources: {e}")
        return None

def example_align_and_normalize(tickers):
    """
    Example of aligning and normalizing data from multiple tickers for comparison.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Tuple of (aligned DataFrames, normalized DataFrames) if successful, None otherwise
    """
    try:
        logger.info(f"Aligning and normalizing data for tickers: {tickers}")
        
        # Get data for each ticker (use sample data for demonstration)
        dataframes = []
        
        # Create sample data with different date ranges
        sample_data = {}
        for i, ticker in enumerate(tickers):
            # Stagger start dates to demonstrate alignment
            start_date = pd.to_datetime('2020-01-01') + timedelta(days=i*5)
            num_days = 90 - i*10  # Different lengths
            
            # Create sample data for this ticker
            df = create_sample_data([ticker], start_date=start_date, num_days=num_days)[ticker]
            sample_data[ticker] = df
            dataframes.append(df)
        
        # Align the time series
        logger.info("Aligning time series...")
        aligned_dfs = align_time_series(
            dataframes=dataframes,
            method='outer',  # Use outer join to include all dates
            fill_method='ffill'  # Forward fill missing values
        )
        
        logger.info(f"Successfully aligned {len(aligned_dfs)} DataFrames")
        for i, df in enumerate(aligned_dfs):
            logger.info(f"Aligned DataFrame {i} shape: {df.shape}")
        
        # Normalize for comparison
        logger.info("Normalizing data for comparison...")
        normalized_dfs = normalize_for_comparison(
            dataframes=aligned_dfs,
            column='close',
            method='percent_change',
            base_date=None  # Use first date in each DataFrame
        )
        
        logger.info(f"Successfully normalized {len(normalized_dfs)} DataFrames")
        
        # Create a combined DataFrame for visualization
        combined_df = pd.DataFrame()
        for i, df in enumerate(normalized_dfs):
            if not df.empty and 'close_normalized' in df.columns:
                combined_df[tickers[i]] = df['close_normalized']
        
        logger.info(f"Combined normalized data:\n{combined_df.head()}")
        
        return aligned_dfs, normalized_dfs, combined_df
        
    except Exception as e:
        logger.error(f"Error in example_align_and_normalize: {e}")
        return None, None, None

def example_resample_and_align(tickers, freq='W'):
    """
    Example of resampling and aligning data from multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        freq: Frequency for resampling
        
    Returns:
        List of resampled and aligned DataFrames if successful, None otherwise
    """
    try:
        logger.info(f"Resampling and aligning data for tickers: {tickers} to frequency: {freq}")
        
        # Get data for each ticker (use sample data for demonstration)
        dataframes = []
        
        # Create sample data with different date ranges
        for i, ticker in enumerate(tickers):
            # Stagger start dates to demonstrate alignment
            start_date = pd.to_datetime('2020-01-01') + timedelta(days=i*3)
            num_days = 90 - i*5  # Different lengths
            
            # Create sample data for this ticker
            df = create_sample_data([ticker], start_date=start_date, num_days=num_days)[ticker]
            dataframes.append(df)
        
        # Resample and align
        logger.info(f"Resampling to {freq} frequency and aligning...")
        resampled_aligned_dfs = resample_and_align(
            dataframes=dataframes,
            freq=freq,
            method='outer',  # Use outer join to include all dates
            fill_method='ffill'  # Forward fill missing values
        )
        
        logger.info(f"Successfully resampled and aligned {len(resampled_aligned_dfs)} DataFrames")
        for i, df in enumerate(resampled_aligned_dfs):
            logger.info(f"Resampled DataFrame {i} shape: {df.shape}")
            logger.info(f"Sample of resampled data for {tickers[i]}:\n{df.head()}")
        
        return resampled_aligned_dfs
        
    except Exception as e:
        logger.error(f"Error in example_resample_and_align: {e}")
        return None

def main():
    """Run all examples."""
    logger.info("Starting data merging examples...")
    
    # Get available tickers
    try:
        available_tickers = get_tickers_with_data()
        logger.info(f"Found {len(available_tickers)} tickers in the database.")
        
        # Use at least 2 tickers for examples
        if len(available_tickers) >= 2:
            tickers = available_tickers[:3]  # Use up to 3 tickers
        else:
            tickers = ['AAPL', 'MSFT', 'GOOGL']  # Default tickers if not enough in database
            
        logger.info(f"Using tickers: {tickers}")
        
        # Get available sources for the first ticker
        if tickers:
            available_sources = get_data_sources_for_ticker(tickers[0])
            logger.info(f"Found {len(available_sources)} data sources for {tickers[0]} in the database.")
        else:
            available_sources = []
        
        # Use at least 2 sources for examples
        if len(available_sources) >= 2:
            sources = available_sources[:2]  # Use up to 2 sources
        else:
            sources = ['SOURCE_A', 'SOURCE_B']  # Default sources if not enough in database
            
        logger.info(f"Using sources: {sources}")
        
        # Example 1: Merge data from multiple tickers
        merged_df = example_merge_stock_data(tickers)
        
        # Example 2: Merge data from multiple sources
        if tickers:
            merged_sources_df = example_merge_from_multiple_sources(tickers[0], sources)
        
        # Example 3: Align and normalize data for comparison
        aligned_dfs, normalized_dfs, combined_df = example_align_and_normalize(tickers)
        
        # Example 4: Resample and align data
        resampled_dfs = example_resample_and_align(tickers, freq='W')
        
        # Show successful completion of all examples
        if merged_df is not None and aligned_dfs is not None and resampled_dfs is not None:
            logger.info("Successfully demonstrated all data merging features!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    
    logger.info("All examples completed!")

if __name__ == "__main__":
    main()
