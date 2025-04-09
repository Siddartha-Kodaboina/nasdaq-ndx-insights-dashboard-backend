import logging
import pandas as pd
import sys
from datetime import datetime, timedelta
from app.utils.data_transformer import resample_ohlcv
from app.schemas import ResampleFrequency, AggregatedStockData

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   stream=sys.stdout)
logger = logging.getLogger("debug_aggregation")

def create_test_data(test_case='standard'):
    """Create test data similar to what we'd have in the application
    
    Args:
        test_case: Type of test data to create
                  'standard': Regular data with some NaN values
                  'multi_ticker': Data with multiple tickers
                  'multi_source': Data with multiple sources
                  'month_boundary': Data that spans month boundaries
    """
    if test_case == 'standard':
        # Create sample data spanning multiple months
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        data = {
            'ticker': ['AAPL'] * 60,
            'open': [150 + i * 0.5 for i in range(60)],
            'high': [155 + i * 0.5 for i in range(60)],
            'low': [145 + i * 0.5 for i in range(60)],
            'close': [152 + i * 0.5 for i in range(60)],
            'volume': [1000000 + i * 50000 for i in range(60)],
            'source': ['source_a'] * 60
        }
        
        # Create a DataFrame
        df = pd.DataFrame(data, index=dates)
        
        # Add some NaN values to test robustness
        df.loc[dates[5:10], 'ticker'] = None
        df.loc[dates[15:20], 'source'] = None
        
    elif test_case == 'multi_ticker':
        # Create data with multiple tickers
        dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
        tickers = ['AAPL'] * 30 + ['MSFT'] * 30 + ['GOOG'] * 30
        data = {
            'ticker': tickers,
            'open': [150 + i * 0.5 for i in range(90)],
            'high': [155 + i * 0.5 for i in range(90)],
            'low': [145 + i * 0.5 for i in range(90)],
            'close': [152 + i * 0.5 for i in range(90)],
            'volume': [1000000 + i * 50000 for i in range(90)],
            'source': ['source_a'] * 90
        }
        df = pd.DataFrame(data, index=dates)
        
    elif test_case == 'multi_source':
        # Create data with multiple sources
        dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
        sources = ['source_a'] * 30 + ['source_b'] * 30 + ['source_c'] * 30
        data = {
            'ticker': ['AAPL'] * 90,
            'open': [150 + i * 0.5 for i in range(90)],
            'high': [155 + i * 0.5 for i in range(90)],
            'low': [145 + i * 0.5 for i in range(90)],
            'close': [152 + i * 0.5 for i in range(90)],
            'volume': [1000000 + i * 50000 for i in range(90)],
            'source': sources
        }
        df = pd.DataFrame(data, index=dates)
        
    elif test_case == 'month_boundary':
        # Create data that specifically spans month boundaries to test ME frequency
        # Create dates that span across multiple month boundaries
        dates = pd.date_range(start='2023-01-25', end='2023-04-05', freq='D')
        data = {
            'ticker': ['AAPL'] * len(dates),
            'open': [150 + i * 0.5 for i in range(len(dates))],
            'high': [155 + i * 0.5 for i in range(len(dates))],
            'low': [145 + i * 0.5 for i in range(len(dates))],
            'close': [152 + i * 0.5 for i in range(len(dates))],
            'volume': [1000000 + i * 50000 for i in range(len(dates))],
            'source': ['source_a'] * len(dates)
        }
        df = pd.DataFrame(data, index=dates)
        
        # Add some NaN values near month boundaries to test edge cases
        month_ends = [d for d in dates if d.day == pd.Timestamp(d).days_in_month]
        for end_date in month_ends:
            # Add NaN values around month end
            df.loc[end_date - pd.Timedelta(days=1):end_date + pd.Timedelta(days=1), 'ticker'] = None
    
    logger.info(f"Created test DataFrame for case '{test_case}' with shape {df.shape}")
    logger.info(f"DataFrame sample:\n{df.head(5)}")
    if not df.empty:
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def test_frequency(df, freq):
    """Test a specific resampling frequency"""
    logger.info(f"\n\n--- Testing frequency: {freq} ---")
    try:
        # Log the input data shape and columns
        logger.info(f"Input data shape: {df.shape}, columns: {df.columns.tolist()}")
        if not df.empty:
            logger.info(f"Input data first row: {df.iloc[0].to_dict()}")
            logger.info(f"Input data index type: {type(df.index)}")
        
        # Check for NaN values in ticker and source columns
        if 'ticker' in df.columns:
            logger.info(f"NaN values in ticker column: {df['ticker'].isna().sum()}")
        if 'source' in df.columns:
            logger.info(f"NaN values in source column: {df['source'].isna().sum()}")
        
        # Perform resampling
        resampled = resample_ohlcv(df, freq)
        logger.info(f"Resampled shape: {resampled.shape}")
        logger.info(f"Resampled columns: {resampled.columns.tolist()}")
        if not resampled.empty:
            logger.info(f"Resampled head:\n{resampled.head()}")
        
        # Check for NaN values in resampled ticker and source columns
        if 'ticker' in resampled.columns:
            logger.info(f"NaN values in resampled ticker column: {resampled['ticker'].isna().sum()}")
        if 'source' in resampled.columns:
            logger.info(f"NaN values in resampled source column: {resampled['source'].isna().sum()}")
        
        # Try creating AggregatedStockData objects
        aggregated_data = []
        for index, row in resampled.iterrows():
            logger.info(f"Processing row with index {index}, data: {row.to_dict()}")
            
            # Determine period start and end based on frequency
            if freq == ResampleFrequency.DAILY.value:
                period_start = index
                period_end = index
            elif freq == ResampleFrequency.WEEKLY.value:
                period_start = index - timedelta(days=6)
                period_end = index
            elif freq == ResampleFrequency.MONTHLY.value:
                # Handle ME (month-end) frequency specifically
                period_start = index.replace(day=1)
                # Calculate the last day of the month properly
                if index.month == 12:
                    next_month_year = index.year + 1
                    next_month = 1
                else:
                    next_month_year = index.year
                    next_month = index.month + 1
                period_end = datetime(next_month_year, next_month, 1) - timedelta(days=1)
                logger.info(f"Monthly period: start={period_start}, end={period_end}")
            elif freq == ResampleFrequency.QUARTERLY.value:
                quarter = (index.month - 1) // 3
                period_start = datetime(index.year, quarter * 3 + 1, 1)
                if quarter == 3:
                    period_end = datetime(index.year + 1, 1, 1) - timedelta(days=1)
                else:
                    period_end = datetime(index.year, (quarter + 1) * 3 + 1, 1) - timedelta(days=1)
            elif freq == ResampleFrequency.YEARLY.value:
                period_start = datetime(index.year, 1, 1)
                period_end = datetime(index.year, 12, 31)
            
            # Get source from row or use transformed as default
            import pandas as pd
            
            source_value = row.get('source', 'transformed')
            if pd.isna(source_value):
                source_value = 'transformed'
                logger.warning(f"Found NaN source value, using default: {source_value}")
            
            # Handle NaN values for ticker
            ticker_str = row.get('ticker', 'unknown')
            if pd.isna(ticker_str):
                ticker_str = 'unknown'
                logger.warning(f"Found NaN ticker value, using default: {ticker_str}")
            
            # Ensure data_points is an integer
            data_points = row.get('data_points', 0)
            if pd.isna(data_points):
                data_points = 0
                logger.warning("Found NaN data_points value, using default: 0")
            else:
                data_points = int(data_points)
            
            # Log the values we're using to create the AggregatedStockData
            logger.info(f"Creating AggregatedStockData with: ticker={ticker_str}, period_start={period_start}, "
                        f"period_end={period_end}, data_points={data_points}, source={source_value}")
            
            # Create the aggregated data object
            try:
                aggregated_item = AggregatedStockData(
                    ticker=ticker_str,
                    period_start=period_start,
                    period_end=period_end,
                    data_points=data_points,
                    source=str(source_value)
                )
                
                # Add OHLC data
                open_val = row.get('open')
                high_val = row.get('high')
                low_val = row.get('low')
                close_val = row.get('close')
                
                # Handle NaN values for OHLC
                aggregated_item.open = None if pd.isna(open_val) else float(open_val)
                aggregated_item.high = None if pd.isna(high_val) else float(high_val)
                aggregated_item.low = None if pd.isna(low_val) else float(low_val)
                aggregated_item.close = None if pd.isna(close_val) else float(close_val)
                
                # Add volume
                volume_val = row.get('volume')
                aggregated_item.volume = None if pd.isna(volume_val) else float(volume_val)
                
                # Add returns if available
                if 'returns' in row:
                    returns_val = row.get('returns')
                    aggregated_item.returns = None if pd.isna(returns_val) else float(returns_val)
                
                aggregated_data.append(aggregated_item)
                logger.info(f"Successfully created AggregatedStockData: {aggregated_item}")
            except Exception as e:
                logger.error(f"Error creating AggregatedStockData: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"Created {len(aggregated_data)} AggregatedStockData objects")
        
    except Exception as e:
        logger.error(f"Error with frequency {freq}: {e}")

def test_resampling():
    """Test resampling with different frequencies, focusing on monthly"""
    # Test with standard data
    logger.info("\n\n=== TESTING WITH STANDARD DATA ===")
    df_standard = create_test_data('standard')
    
    # Test monthly frequency first (which was problematic)
    logger.info("\n\n--- FOCUSING ON MONTHLY FREQUENCY (ME) ---")
    test_frequency(df_standard, ResampleFrequency.MONTHLY.value)
    
    # Test with month boundary data specifically for ME frequency
    logger.info("\n\n=== TESTING WITH MONTH BOUNDARY DATA ===")
    df_month_boundary = create_test_data('month_boundary')
    test_frequency(df_month_boundary, ResampleFrequency.MONTHLY.value)
    
    # Test with multi-ticker data
    logger.info("\n\n=== TESTING WITH MULTIPLE TICKERS ===")
    df_multi_ticker = create_test_data('multi_ticker')
    test_frequency(df_multi_ticker, ResampleFrequency.MONTHLY.value)
    
    # Test with multi-source data
    logger.info("\n\n=== TESTING WITH MULTIPLE SOURCES ===")
    df_multi_source = create_test_data('multi_source')
    test_frequency(df_multi_source, ResampleFrequency.MONTHLY.value)
    
    # Test other frequencies with standard data
    logger.info("\n\n--- TESTING OTHER FREQUENCIES WITH STANDARD DATA ---")
    for freq in [f.value for f in ResampleFrequency if f != ResampleFrequency.MONTHLY]:
        test_frequency(df_standard, freq)

def test_single_frequency(freq):
    """Test a single frequency with all test cases"""
    logger.info(f"\n\n=== TESTING FREQUENCY: {freq} ===")
    
    test_cases = ['standard', 'month_boundary', 'multi_ticker', 'multi_source']
    for test_case in test_cases:
        logger.info(f"\n--- Test case: {test_case} ---")
        df = create_test_data(test_case)
        test_frequency(df, freq)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data aggregation functionality')
    parser.add_argument('--freq', choices=[f.value for f in ResampleFrequency], 
                        help='Test a specific frequency')
    parser.add_argument('--test-case', choices=['standard', 'month_boundary', 'multi_ticker', 'multi_source'],
                        help='Test a specific test case')
    
    args = parser.parse_args()
    
    if args.freq and args.test_case:
        # Test specific frequency with specific test case
        logger.info(f"Testing frequency {args.freq} with test case {args.test_case}")
        df = create_test_data(args.test_case)
        test_frequency(df, args.freq)
    elif args.freq:
        # Test specific frequency with all test cases
        test_single_frequency(args.freq)
    elif args.test_case:
        # Test all frequencies with specific test case
        logger.info(f"Testing all frequencies with test case {args.test_case}")
        df = create_test_data(args.test_case)
        for freq in [f.value for f in ResampleFrequency]:
            test_frequency(df, freq)
    else:
        # Run all tests
        test_resampling()
