from contextlib import contextmanager
from typing import Generator, List, Dict, Any
from app.models import DataSource as DataSourceEnum
from app.utils.data_sources.factory import DataSourceFactory
from app.utils.data_sources.base import DataSource

@contextmanager
def get_data_source(source_type: DataSourceEnum) -> Generator[DataSource, None, None]:
    """Context manager for getting and using a data source.
    
    This function creates a context manager that automatically handles
    resource acquisition and cleanup for a data source.
    
    Args:
        source_type: The enum value for the source type
        
    Yields:
        The data source instance
        
    Example:
        with get_data_source(DataSourceEnum.SOURCE_A) as source:
            data = source.get_data("AAPL", from_date, to_date)
    """
    source = DataSourceFactory.get_source(source_type)
    with source:
        yield source

@contextmanager
def get_all_data_sources() -> Generator[Dict[DataSourceEnum, DataSource], None, None]:
    """Context manager for getting and using all data sources.
    
    This function creates a context manager that automatically handles
    resource acquisition and cleanup for all registered data sources.
    
    Yields:
        A dictionary mapping source types to data source instances
        
    Example:
        with get_all_data_sources() as sources:
            source_a = sources[DataSourceEnum.SOURCE_A]
            data = source_a.get_data("AAPL", from_date, to_date)
    """
    sources = DataSourceFactory.get_all_sources()
    # Use a list to keep track of sources that were successfully entered
    entered_sources = []
    
    try:
        # Enter all sources
        for source_type, source in sources.items():
            source.__enter__()
            entered_sources.append((source_type, source))
        
        yield {source_type: source for source_type, source in entered_sources}
    
    finally:
        # Exit all sources that were successfully entered
        for source_type, source in reversed(entered_sources):
            try:
                source.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing data source {source_type}: {e}")