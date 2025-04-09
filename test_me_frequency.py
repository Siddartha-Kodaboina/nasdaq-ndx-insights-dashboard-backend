"""
Test script to verify the ME (month-end) frequency aggregation functionality.

This script tests the aggregated data retrieval endpoint with the ME frequency
to ensure it works correctly after the fixes.
"""
import asyncio
import httpx
import json
from datetime import datetime, timedelta
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_me_frequency")

# API base URL
BASE_URL = "http://localhost:8000"

async def create_test_task():
    """Create a test task for stock exploration."""
    async with httpx.AsyncClient() as client:
        # Use dates in the past to avoid validation errors
        # Current date for end_date and 6 months before that for start_date
        end_date = datetime.now() - timedelta(days=30)  # Yesterday
        start_date = end_date - timedelta(days=180)  # 6 months before end_date
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_date_str = end_date.strftime("%Y-%m-%dT23:59:59")
        
        # Task data
        task_data = {
            "task_type": "explore_stock",
            "parameters": {
                "ticker": "AAPL",
                "source": "source_a",
                "from_date": start_date_str,
                "to_date": end_date_str
            }
        }
        
        logger.info(f"Creating test task with data: {task_data}")
        
        # Create task
        response = await client.post(f"{BASE_URL}/tasks/", json=task_data)
        
        if response.status_code != 201:
            logger.error(f"Failed to create task: {response.status_code} - {response.text}")
            return None
        
        task = response.json()
        task_id = task["id"]
        logger.info(f"Created task with ID: {task_id}")
        
        return task_id

async def wait_for_task_completion(task_id):
    """Wait for the task to complete."""
    async with httpx.AsyncClient() as client:
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            response = await client.get(f"{BASE_URL}/tasks/{task_id}")
            
            if response.status_code != 200:
                logger.error(f"Failed to get task status: {response.status_code} - {response.text}")
                return False
            
            task = response.json()
            status = task["status"]
            logger.info(f"Task {task_id} status: {status}, progress: {task['progress']}")
            
            if status == "completed":
                return True
            elif status == "failed":
                logger.error(f"Task failed with error: {task.get('error_message')}")
                return False
            
            # Wait before checking again
            await asyncio.sleep(2)
            attempt += 1
        
        logger.error(f"Task did not complete within the expected time")
        return False

async def test_me_frequency(task_id):
    """Test the ME frequency aggregation."""
    async with httpx.AsyncClient() as client:
        # Test parameters
        params = {
            "group_by": "ME",  # Month-end frequency
            "include_ohlc": "true",
            "include_volume": "true",
            "include_returns": "true"
        }
        
        logger.info(f"Testing ME frequency aggregation for task {task_id} with params: {params}")
        
        # Get aggregated data
        response = await client.get(f"{BASE_URL}/tasks/{task_id}/aggregated", params=params)
        
        if response.status_code != 200:
            logger.error(f"Failed to get aggregated data: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        logger.info(f"Successfully retrieved {len(data)} aggregated data points")
        
        # Log the first few data points for inspection
        for i, item in enumerate(data[:3]):
            logger.info(f"Data point {i+1}:")
            logger.info(f"  Ticker: {item['ticker']}")
            logger.info(f"  Period: {item['period_start']} to {item['period_end']}")
            logger.info(f"  OHLC: open={item.get('open')}, high={item.get('high')}, low={item.get('low')}, close={item.get('close')}")
            logger.info(f"  Volume: {item.get('volume')}")
            logger.info(f"  Returns: {item.get('returns')}")
            logger.info(f"  Data points: {item['data_points']}")
        
        return len(data) > 0

async def main():
    """Main test function."""
    # Create a test task
    task_id = await create_test_task()
    if not task_id:
        logger.error("Failed to create test task")
        return
    
    # Wait for task to complete
    completed = await wait_for_task_completion(task_id)
    if not completed:
        logger.error("Task did not complete successfully")
        return
    
    # Test ME frequency
    success = await test_me_frequency(task_id)
    
    if success:
        logger.info("✅ ME frequency test passed successfully!")
    else:
        logger.error("❌ ME frequency test failed")

if __name__ == "__main__":
    asyncio.run(main())
