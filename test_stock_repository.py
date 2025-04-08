"""
Test script for the stock data repository module.
"""

import logging
import time
import asyncio
from datetime import datetime, timedelta

from app.utils.stock_repository import (
    get_stock_data,
    get_latest_stock_data,
    get_tickers_with_data,
    get_data_sources_for_ticker,
    get_date_range_for_ticker,
    aggregate_stock_data,
    clear_cache,
    get_cache_stats,
    StockDataNotFoundError,
    InvalidFilterError
)

# Import for populating test data
from app.models import Task, TaskType, TaskStatus, DataSource
from app.async_workers import process_task_async
from app.utils.async_db_client import get_async_db, create_async_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_get_tickers_with_data():
    """Test retrieving all tickers with data."""
    logger.info("Testing get_tickers_with_data...")
    
    tickers = get_tickers_with_data()
    logger.info(f"Found {len(tickers)} tickers with data: {tickers}")
    
    if len(tickers) == 0:
        logger.warning("No tickers found in the database. Database may be empty.")
        return None
    return tickers[0]  # Return first ticker for use in other tests

def test_get_data_sources_for_ticker(ticker):
    """Test retrieving data sources for a ticker."""
    logger.info(f"Testing get_data_sources_for_ticker for {ticker}...")
    
    sources = get_data_sources_for_ticker(ticker)
    logger.info(f"Found {len(sources)} sources for {ticker}: {sources}")
    
    assert len(sources) > 0, f"No data sources found for ticker: {ticker}"
    return sources[0]  # Return first source for use in other tests

def test_get_date_range_for_ticker(ticker, source=None):
    """Test retrieving date range for a ticker."""
    logger.info(f"Testing get_date_range_for_ticker for {ticker}...")
    
    try:
        start_date, end_date = get_date_range_for_ticker(ticker, source)
        logger.info(f"Date range for {ticker}: {start_date} to {end_date}")
        
        assert start_date <= end_date, "Start date should be before or equal to end date"
        return start_date, end_date
    except StockDataNotFoundError as e:
        logger.warning(f"Test failed: {e}")
        return None, None

def test_get_stock_data(ticker, source=None):
    """Test retrieving stock data for a ticker."""
    logger.info(f"Testing get_stock_data for {ticker}...")
    
    try:
        # Get date range first
        start_date, end_date = get_date_range_for_ticker(ticker, source)
        if not start_date or not end_date:
            logger.warning("Skipping test_get_stock_data due to missing date range")
            return
        
        # Get stock data
        data = get_stock_data(ticker, start_date, end_date, source)
        logger.info(f"Retrieved {len(data)} data points for {ticker}")
        
        # Print first and last data point
        if data:
            logger.info(f"First data point: {data[0]}")
            logger.info(f"Last data point: {data[-1]}")
        
        assert len(data) > 0, f"No data retrieved for {ticker}"
    except (StockDataNotFoundError, InvalidFilterError) as e:
        logger.warning(f"Test failed: {e}")

def test_get_latest_stock_data(ticker, source=None):
    """Test retrieving latest stock data for a ticker."""
    logger.info(f"Testing get_latest_stock_data for {ticker}...")
    
    try:
        data = get_latest_stock_data(ticker, days=30, source=source)
        logger.info(f"Latest data for {ticker}: {data}")
        
        assert data is not None, f"No latest data retrieved for {ticker}"
    except StockDataNotFoundError as e:
        logger.warning(f"Test failed: {e}")

def test_cache_functionality(ticker, source=None):
    """Test the caching functionality."""
    logger.info("Testing cache functionality...")
    
    # Clear cache first
    clear_cache()
    stats_before = get_cache_stats()
    logger.info(f"Cache stats before: {stats_before}")
    
    # Get date range
    start_date, end_date = get_date_range_for_ticker(ticker, source)
    if not start_date or not end_date:
        logger.warning("Skipping cache test due to missing date range")
        return
    
    # First call should miss cache
    logger.info("First call (should miss cache)...")
    get_stock_data(ticker, start_date, end_date, source, use_cache=True)
    
    # Second call should hit cache
    logger.info("Second call (should hit cache)...")
    get_stock_data(ticker, start_date, end_date, source, use_cache=True)
    
    # Check cache stats
    stats_after = get_cache_stats()
    logger.info(f"Cache stats after: {stats_after}")
    
    assert stats_after["hits"] > stats_before["hits"], "Cache hit count should increase"
    assert stats_after["size"] > stats_before["size"], "Cache size should increase"

def test_error_handling():
    """Test error handling for invalid inputs."""
    logger.info("Testing error handling...")
    
    # Test invalid ticker
    try:
        get_stock_data("INVALID_TICKER")
        logger.error("Test failed: Should have raised StockDataNotFoundError for invalid ticker")
    except StockDataNotFoundError:
        logger.info("Successfully caught StockDataNotFoundError for invalid ticker")
    
    # Test invalid date range
    try:
        end_date = datetime.now()
        start_date = end_date + timedelta(days=10)  # Start date after end date
        get_stock_data("AAPL", start_date, end_date)
        logger.error("Test failed: Should have raised InvalidFilterError for invalid date range")
    except InvalidFilterError:
        logger.info("Successfully caught InvalidFilterError for invalid date range")
    
    # Test invalid source
    try:
        get_stock_data("AAPL", source="INVALID_SOURCE")
        logger.error("Test failed: Should have raised InvalidFilterError for invalid source")
    except InvalidFilterError:
        logger.info("Successfully caught InvalidFilterError for invalid source")

async def populate_test_data():
    """Populate the database with test data."""
    logger.info("Populating database with test data...")
    
    # Create tables if they don't exist
    await create_async_tables()
    
    # Create test tasks
    tasks = [
        # Explore stock task for AAPL
        {
            "task_type": TaskType.EXPLORE_STOCK,
            "parameters": {
                "ticker": "AAPL",
                "source": "source_a",
                "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "to_date": datetime.now().isoformat()
            }
        },
        # Compare stocks task for AAPL and MSFT
        {
            "task_type": TaskType.COMPARE_STOCKS,
            "parameters": {
                "ticker1": "AAPL",
                "ticker2": "MSFT",
                "field": "close",
                "source": "source_b",
                "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "to_date": datetime.now().isoformat()
            }
        },
        # Stock vs index task for AAPL and ^NDX
        {
            "task_type": TaskType.STOCK_VS_INDEX,
            "parameters": {
                "ticker": "AAPL",
                "index_ticker": "^NDX",
                "source": "source_a",
                "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "to_date": datetime.now().isoformat()
            }
        }
    ]
    
    # Create tasks in the database and process them
    task_ids = []
    async with get_async_db() as db:
        for task_data in tasks:
            task = Task(
                task_type=task_data["task_type"],
                status=TaskStatus.PENDING,
                parameters=task_data["parameters"]
            )
            db.add(task)
            await db.commit()
            await db.refresh(task)
            task_ids.append(task.id)
    
    # Process tasks
    for task_id in task_ids:
        logger.info(f"Processing task {task_id}...")
        result = await process_task_async(task_id)
        logger.info(f"Task {task_id} processed with status: {result['status']}")
    
    logger.info("Test data population completed!")

async def async_main():
    """Run all tests with async support."""
    # Check if database is empty
    tickers = get_tickers_with_data()
    if len(tickers) == 0:
        logger.info("Database is empty. Populating with test data...")
        await populate_test_data()
    
    # Run tests
    logger.info("Starting stock repository tests...")
    
    # Test getting tickers
    ticker = test_get_tickers_with_data()
    
    if ticker is None:
        logger.error("Cannot continue tests without data. Please check database setup.")
        return
    
    # Test getting sources for ticker
    source = test_get_data_sources_for_ticker(ticker)
    
    # Test getting date range for ticker
    test_get_date_range_for_ticker(ticker, source)
    
    # Test getting stock data
    test_get_stock_data(ticker, source)
    
    # Test getting latest stock data
    test_get_latest_stock_data(ticker, source)
    
    # Test cache functionality
    test_cache_functionality(ticker, source)
    
    # Test error handling
    test_error_handling()
    
    logger.info("All tests completed!")

def main():
    """Run the async main function."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
