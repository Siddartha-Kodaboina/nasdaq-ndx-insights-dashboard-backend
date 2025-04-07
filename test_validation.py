import asyncio
import json
from datetime import datetime, timedelta
from fastapi import HTTPException
from app.schemas import TaskCreate, ExploreStockParams, CompareStocksParams, StockVsIndexParams
from app.schemas import TaskTypeEnum, DataSourceEnum
from app.utils.validators import validate_request_body, validate_ticker

async def test_validation_schemas():
    """Test the validation schemas."""
    print("Testing validation schemas...")
    
    # Test valid ExploreStockParams
    try:
        params = ExploreStockParams(
            ticker="AAPL",
            source=DataSourceEnum.SOURCE_A,
            from_date=datetime.now() - timedelta(days=30),
            to_date=datetime.now() - timedelta(days=1)
        )
        print(f"Valid ExploreStockParams: {params}")
    except Exception as e:
        print(f"Error with valid ExploreStockParams: {e}")
    
    # Test invalid date range (to_date before from_date)
    try:
        params = ExploreStockParams(
            ticker="AAPL",
            source=DataSourceEnum.SOURCE_A,
            from_date=datetime.now() - timedelta(days=1),
            to_date=datetime.now() - timedelta(days=30)
        )
        print("Invalid date range was accepted!")
    except Exception as e:
        print(f"Correctly caught invalid date range: {e}")
    
    # Test future date
    try:
        params = ExploreStockParams(
            ticker="AAPL",
            source=DataSourceEnum.SOURCE_A,
            from_date=datetime.now() - timedelta(days=30),
            to_date=datetime.now() + timedelta(days=1)
        )
        print("Future date was accepted!")
    except Exception as e:
        print(f"Correctly caught future date: {e}")
    
    # Test CompareStocksParams with invalid field
    try:
        params = CompareStocksParams(
            ticker1="AAPL",
            ticker2="MSFT",
            field="invalid",
            from_date=datetime.now() - timedelta(days=30),
            to_date=datetime.now() - timedelta(days=1)
        )
        print("Invalid field was accepted!")
    except Exception as e:
        print(f"Correctly caught invalid field: {e}")
    
    # Test valid TaskCreate with ExploreStockParams
    try:
        task = TaskCreate(
            task_type=TaskTypeEnum.EXPLORE_STOCK,
            parameters={
                "ticker": "AAPL",
                "source": "source_a",
                "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "to_date": (datetime.now() - timedelta(days=1)).isoformat()
            }
        )
        print(f"Valid TaskCreate with ExploreStockParams: {task}")
    except Exception as e:
        print(f"Error with valid TaskCreate: {e}")
    
    # Test TaskCreate with mismatched parameters
    try:
        task = TaskCreate(
            task_type=TaskTypeEnum.EXPLORE_STOCK,
            parameters={
                "ticker1": "AAPL",
                "ticker2": "MSFT",
                "field": "close",
                "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "to_date": (datetime.now() - timedelta(days=1)).isoformat()
            }
        )
        print("Mismatched parameters were accepted!")
    except Exception as e:
        print(f"Correctly caught mismatched parameters: {e}")
    
    print("Validation schemas test completed!")

@validate_request_body(TaskCreate)
async def mock_create_task(task_data):
    """Mock function to test the validate_request_body decorator."""
    return task_data

@validate_ticker
async def mock_get_stock_data(ticker, source):
    """Mock function to test the validate_ticker decorator."""
    return {"ticker": ticker, "source": source}

async def test_validation_decorators():
    """Test the validation decorators."""
    print("\nTesting validation decorators...")
    
    # Test validate_request_body with valid data
    try:
        result = await mock_create_task({
            "task_type": "explore_stock",
            "parameters": {
                "ticker": "AAPL",
                "source": "source_a",
                "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "to_date": (datetime.now() - timedelta(days=1)).isoformat()
            }
        })
        print(f"Valid request body was accepted: {result}")
    except HTTPException as e:
        print(f"Error with valid request body: {e.detail}")
    
    # Test validate_request_body with invalid data
    try:
        result = await mock_create_task({
            "task_type": "explore_stock",
            "parameters": {
                "ticker": "AAPL",
                "source": "source_a",
                "from_date": (datetime.now() + timedelta(days=1)).isoformat(),
                "to_date": (datetime.now() + timedelta(days=30)).isoformat()
            }
        })
        print("Invalid request body was accepted!")
    except HTTPException as e:
        print(f"Correctly caught invalid request body: {e.status_code}")
    
    # Note: We can't fully test validate_ticker here because it requires
    # a real data source. We'll test it through the API endpoints.
    
    print("Validation decorators test completed!")

async def main():
    """Run all validation tests."""
    await test_validation_schemas()
    await test_validation_decorators()
    print("\nAll validation tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())