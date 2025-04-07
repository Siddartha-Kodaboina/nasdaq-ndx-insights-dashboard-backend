from app.utils.data_sources import get_data_source
from app.models import DataSource as DataSourceEnum
from datetime import datetime
import pandas as pd

def test_csv_source():
    """Test the CSV data source implementation."""
    print("Testing CSV data source...")
    
    # Test Source A
    print("\nTesting Source A:")
    with get_data_source(DataSourceEnum.SOURCE_A) as source:
        # Get available tickers
        tickers = source.get_tickers()
        print(f"Available tickers: {tickers}")
        
        # Get data for a ticker
        ticker = tickers[0]
        data = source.get_data(ticker)
        print(f"Data for {ticker}:")
        print(data.head())
        
        # Test date filtering
        from_date = datetime(2020, 1, 1)
        to_date = datetime(2020, 12, 31)
        filtered_data = source.get_data(ticker, from_date, to_date)
        print(f"\nFiltered data for {ticker} ({from_date.date()} to {to_date.date()}):")
        print(filtered_data.head())
    
    # Test Source B
    print("\nTesting Source B:")
    with get_data_source(DataSourceEnum.SOURCE_B) as source:
        # Get available tickers
        tickers = source.get_tickers()
        print(f"Available tickers: {tickers}")
        
        # Get data for a ticker
        ticker = tickers[0]
        data = source.get_data(ticker)
        print(f"Data for {ticker}:")
        print(data.head())
    
    print("\nCSV data source test completed successfully!")

if __name__ == "__main__":
    test_csv_source()