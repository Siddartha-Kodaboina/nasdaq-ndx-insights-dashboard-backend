import os
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime
import logging
from app.utils.data_sources.base import DataSource
from app.models import DataSource as DataSourceEnum

logger = logging.getLogger(__name__)

class CSVDataSource(DataSource):
    """Implementation of DataSource for CSV files.
    
    This class reads stock data from CSV files and provides methods
    to access and filter the data.
    """
    
    def __init__(self, file_path: str, source_type: DataSourceEnum):
        """Initialize the CSV data source.
        
        Args:
            file_path: Path to the CSV file
            source_type: The enum value for the source type
        """
        self.file_path = file_path
        self.source_type = source_type
        self.data = None
        self._tickers = None
    
    def __enter__(self):
        """Context manager entry method.
        
        Loads the CSV file into memory when entering the context.
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.data = self._load_csv()
            return self
        except Exception as e:
            logger.error(f"Error loading CSV file {self.file_path}: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method.
        
        Cleans up resources when exiting the context.
        """
        # No resources to clean up for CSV files
        pass
    
    def _load_csv(self) -> pd.DataFrame:
        """Load the CSV file into a pandas DataFrame.
        
        Returns:
            DataFrame with the CSV data
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            pd.errors.ParserError: If the CSV file is malformed
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(self.file_path)
            
            # Normalize the data
            df = self._normalize_data(df)
            
            return df
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file {self.file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading CSV file {self.file_path}: {e}")
            raise
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data from the CSV file.
        
        Args:
            df: DataFrame with the raw CSV data
            
        Returns:
            Normalized DataFrame
        """
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure all numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date and ticker
        df = df.sort_values(['ticker', 'date'])
        
        return df
    
    def get_tickers(self) -> List[str]:
        """Get a list of available tickers in the data source.
        
        Returns:
            List of ticker symbols
        """
        if self.data is None:
            raise ValueError("Data not loaded. Use with statement to load data.")
        
        if self._tickers is None:
            self._tickers = sorted(self.data['ticker'].unique().tolist())
        
        return self._tickers
    
    def get_data(self, 
                ticker: str, 
                from_date: Optional[datetime] = None, 
                to_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get stock data for a specific ticker and date range.
        
        Args:
            ticker: The stock ticker symbol
            from_date: Start date for data retrieval (inclusive)
            to_date: End date for data retrieval (inclusive)
            
        Returns:
            DataFrame with stock data (date, open, high, low, close, volume)
            
        Raises:
            ValueError: If the ticker is not available or data is not loaded
        """
        if self.data is None:
            raise ValueError("Data not loaded. Use with statement to load data.")
        
        # Filter by ticker
        ticker_data = self.data[self.data['ticker'] == ticker]
        
        if ticker_data.empty:
            raise ValueError(f"Ticker {ticker} not found in data source")
        
        # Filter by date range if provided
        if from_date:
            ticker_data = ticker_data[ticker_data['date'] >= from_date]
        
        if to_date:
            ticker_data = ticker_data[ticker_data['date'] <= to_date]
        
        return ticker_data
    
    def get_source_name(self) -> str:
        """Get the name of this data source.
        
        Returns:
            The source name
        """
        return self.source_type.value