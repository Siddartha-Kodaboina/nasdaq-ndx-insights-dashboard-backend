"""
Stock Data Repository Module

This module provides functions to retrieve stock data from the database
with various filtering options and caching capabilities.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from app.models import StockData, DataSource
from app.utils.db_client import get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple in-memory cache
_cache = {}
_cache_ttl = {}  # Time-to-live for each cache key
_cache_hits = 0
_cache_misses = 0
DEFAULT_CACHE_TTL = 300  # 5 minutes in seconds

class StockDataNotFoundError(Exception):
    """Exception raised when no stock data is found for the given filters."""
    pass

class InvalidFilterError(Exception):
    """Exception raised when invalid filter parameters are provided."""
    pass

def _generate_cache_key(ticker: str, start_date: datetime, end_date: datetime, 
                       source: Optional[Union[DataSource, str]] = None) -> str:
    """Generate a cache key from the query parameters."""
    source_str = str(source) if source else "all"
    return f"{ticker}_{start_date.isoformat()}_{end_date.isoformat()}_{source_str}"

def _get_from_cache(key: str) -> Optional[Any]:
    """Get data from cache if it exists and is not expired."""
    global _cache_hits, _cache_misses
    
    if key in _cache and key in _cache_ttl:
        if time.time() < _cache_ttl[key]:
            _cache_hits += 1
            logger.debug(f"Cache hit for key: {key}")
            return _cache[key]
    
    _cache_misses += 1
    logger.debug(f"Cache miss for key: {key}")
    return None

def _store_in_cache(key: str, data: Any, ttl: int = DEFAULT_CACHE_TTL) -> None:
    """Store data in cache with a time-to-live."""
    _cache[key] = data
    _cache_ttl[key] = time.time() + ttl
    logger.debug(f"Stored in cache: {key} with TTL: {ttl} seconds")

def clear_cache() -> None:
    """Clear the entire cache."""
    global _cache, _cache_ttl, _cache_hits, _cache_misses
    _cache = {}
    _cache_ttl = {}
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Cache cleared")

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "size": len(_cache),
    }

def get_stock_data(
    ticker: str,
    start_date: Optional[Union[datetime, str]] = None,
    end_date: Optional[Union[datetime, str]] = None,
    source: Optional[Union[DataSource, str]] = None,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve stock data for a specific ticker within a date range.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        start_date: Start date for the data range (inclusive)
        end_date: End date for the data range (inclusive)
        source: Data source to filter by (optional)
        use_cache: Whether to use the cache (default: True)
    
    Returns:
        List of dictionaries containing stock data
    
    Raises:
        StockDataNotFoundError: If no data is found for the given filters
        InvalidFilterError: If invalid filter parameters are provided
    """
    # Validate inputs
    if not ticker:
        raise InvalidFilterError("Ticker symbol is required")
    
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        try:
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise InvalidFilterError(f"Invalid start_date format: {start_date}")
    
    if isinstance(end_date, str):
        try:
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise InvalidFilterError(f"Invalid end_date format: {end_date}")
    
    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)  # Default to last 30 days
    
    # Validate date range
    if start_date > end_date:
        raise InvalidFilterError("start_date cannot be after end_date")
    
    # Convert string source to enum if needed
    if isinstance(source, str):
        try:
            source = DataSource(source)
        except ValueError:
            raise InvalidFilterError(f"Invalid source: {source}")
    
    # Check cache first if enabled
    if use_cache:
        cache_key = _generate_cache_key(ticker, start_date, end_date, source)
        cached_data = _get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
    
    # Query the database
    with get_db() as db:
        query = db.query(StockData).filter(StockData.ticker == ticker)
        
        # Apply date filters - note: we need to handle the case where dates might be incorrect
        # For now, we'll still apply the filter but be aware it might not work as expected
        if start_date:
            query = query.filter(StockData.date >= start_date)
        if end_date:
            query = query.filter(StockData.date <= end_date)
        
        # Apply source filter if provided
        if source:
            query = query.filter(StockData.source == source)
        
        # Execute query and get results
        results = query.order_by(StockData.date).all()
        
        if not results:
            logger.warning(f"No stock data found for ticker: {ticker} between {start_date} and {end_date}")
            raise StockDataNotFoundError(f"No stock data found for ticker: {ticker}")
        
        # Convert SQLAlchemy objects to dictionaries
        data = []
        for row in results:
            data.append({
                "id": row.id,
                "task_id": row.task_id,
                "ticker": row.ticker,
                "date": row.date,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "source": row.source.value if hasattr(row.source, 'value') else str(row.source)
            })
        
        # Cache the results if enabled
        if use_cache:
            _store_in_cache(cache_key, data)
        
        return data

def get_latest_stock_data(
    ticker: str,
    days: int = 1,
    source: Optional[Union[DataSource, str]] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Get the latest stock data for a ticker.
    
    Args:
        ticker: The stock ticker symbol
        days: Number of days to retrieve (default: 1)
        source: Data source to filter by (optional)
        use_cache: Whether to use the cache (default: True)
    
    Returns:
        Dictionary containing the latest stock data
    
    Raises:
        StockDataNotFoundError: If no data is found for the given ticker
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = get_stock_data(ticker, start_date, end_date, source, use_cache)
    
    if not data:
        raise StockDataNotFoundError(f"No recent stock data found for ticker: {ticker}")
    
    # Return the most recent data point
    return data[-1]

def get_tickers_with_data() -> List[str]:
    """
    Get a list of all tickers that have data in the database.
    
    Returns:
        List of ticker symbols
    """
    with get_db() as db:
        tickers = db.query(StockData.ticker).distinct().all()
        return [t[0] for t in tickers]

def get_data_sources_for_ticker(ticker: str) -> List[str]:
    """
    Get a list of all data sources available for a specific ticker.
    
    Args:
        ticker: The stock ticker symbol
    
    Returns:
        List of data source names
    """
    with get_db() as db:
        sources = db.query(StockData.source).filter(StockData.ticker == ticker).distinct().all()
        return [s[0].value if hasattr(s[0], 'value') else str(s[0]) for s in sources]

def get_date_range_for_ticker(ticker: str, source: Optional[Union[DataSource, str]] = None) -> Tuple[datetime, datetime]:
    """
    Get the earliest and latest dates available for a specific ticker.
    
    Args:
        ticker: The stock ticker symbol
        source: Data source to filter by (optional)
    
    Returns:
        Tuple of (earliest_date, latest_date)
    
    Raises:
        StockDataNotFoundError: If no data is found for the given ticker
    """
    with get_db() as db:
        query = db.query(
            func.min(StockData.date).label("min_date"),
            func.max(StockData.date).label("max_date")
        ).filter(StockData.ticker == ticker)
        
        if source:
            if isinstance(source, str):
                try:
                    source = DataSource(source)
                except ValueError:
                    raise InvalidFilterError(f"Invalid source: {source}")
            query = query.filter(StockData.source == source)
        
        result = query.one()
        
        if not result.min_date or not result.max_date:
            raise StockDataNotFoundError(f"No data found for ticker: {ticker}")
        
        return (result.min_date, result.max_date)

def aggregate_stock_data(
    ticker: str,
    start_date: Optional[Union[datetime, str]] = None,
    end_date: Optional[Union[datetime, str]] = None,
    source: Optional[Union[DataSource, str]] = None,
    interval: str = 'daily',  # 'daily', 'weekly', 'monthly'
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Aggregate stock data by the specified interval.
    
    Args:
        ticker: The stock ticker symbol
        start_date: Start date for the data range
        end_date: End date for the data range
        source: Data source to filter by
        interval: Aggregation interval ('daily', 'weekly', 'monthly')
        use_cache: Whether to use the cache
    
    Returns:
        List of dictionaries containing aggregated stock data
    
    Raises:
        StockDataNotFoundError: If no data is found for the given filters
        InvalidFilterError: If invalid filter parameters are provided
    """
    # For now, this is a placeholder that just returns daily data
    # In a real implementation, we would aggregate the data based on the interval
    return get_stock_data(ticker, start_date, end_date, source, use_cache)
