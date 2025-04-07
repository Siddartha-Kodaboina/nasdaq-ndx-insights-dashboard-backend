from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class DataSource(ABC):
    """Abstract base class for data sources.
    
    This class defines the interface that all data sources must implement.
    It uses Python's ABC (Abstract Base Class) to enforce implementation
    of required methods in subclasses.
    """
    
    @abstractmethod
    def __enter__(self):
        """Context manager entry method.
        
        This method is called when entering a 'with' statement.
        It should handle resource acquisition and return self.
        """
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method.
        
        This method is called when exiting a 'with' statement.
        It should handle resource cleanup and exception handling.
        """
        pass
    
    @abstractmethod
    def get_tickers(self) -> List[str]:
        """Get a list of available tickers in the data source."""
        pass
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this data source."""
        pass