"""
Test script for core backend features.

This script tests the most critical features of the backend to ensure it's working correctly.
"""

import asyncio
import json
import sys
import logging
import time
from datetime import datetime, timedelta
import httpx
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:8000"

# Test data
TEST_TICKER = "AAPL"
TEST_TASK_TYPE = "explore_stock"

async def create_task(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Create a task for testing."""
    task_data = {
        "task_type": TEST_TASK_TYPE,
        "parameters": {
            "ticker": TEST_TICKER,
            "from_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "to_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "source": "source_a"
        }
    }
    
    response = await client.post(f"{BASE_URL}/tasks/", json=task_data)
    
    if response.status_code != 201:
        logger.error(f"Failed to create task: {response.text}")
        raise Exception(f"Failed to create task: {response.text}")
    
    return response.json()

async def wait_for_task_completion(client: httpx.AsyncClient, task_id: int, max_wait_seconds: int = 60) -> bool:
    """Wait for a task to complete."""
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        # Get task status
        response = await client.get(f"{BASE_URL}/tasks/{task_id}")
        if response.status_code != 200:
            logger.error(f"Failed to get task status: {response.text}")
            return False
        
        status_data = response.json()
        status = status_data.get("status")
        progress = status_data.get("progress", 0)
        
        # Check if task is completed
        if status.lower() == "completed":
            logger.info(f"Task {task_id} completed successfully")
            return True
            
        # Check if task failed or was cancelled
        if status.lower() in ["failed", "cancelled"]:
            logger.error(f"Task {task_id} ended with status: {status}")
            return False
        
        logger.info(f"Task {task_id} still processing. Status: {status}, Progress: {progress}%")
        
        # Wait before checking again
        await asyncio.sleep(2)
    
    logger.error(f"Timeout waiting for task {task_id} to complete")
    return False

async def test_basic_data_retrieval(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test basic data retrieval."""
    logger.info(f"Testing basic data retrieval for task {task_id}")
    
    response = await client.get(f"{BASE_URL}/tasks/{task_id}/data")
    if response.status_code != 200:
        logger.error(f"Failed to get task data: {response.text}")
        return False
    
    data = response.json()
    if not data:
        logger.error("No data returned")
        return False
    
    logger.info(f"Retrieved {len(data)} data points")
    return True

async def test_me_frequency_aggregation(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test ME frequency aggregation."""
    logger.info(f"Testing ME frequency aggregation for task {task_id}")
    
    response = await client.get(
        f"{BASE_URL}/tasks/{task_id}/aggregated",
        params={
            "group_by": "ME",
            "include_ohlc": "true",
            "include_volume": "true",
            "include_returns": "true"
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get ME frequency aggregated data: {response.text}")
        return False
    
    data = response.json()
    if not data:
        logger.error("No data returned for ME frequency aggregation")
        return False
    
    logger.info(f"Successfully retrieved {len(data)} aggregated data points for ME frequency")
    logger.info(f"Sample ME aggregated data: {json.dumps(data[0], indent=2)}")
    return True

async def test_weekly_resampling(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test weekly resampling."""
    logger.info(f"Testing weekly resampling for task {task_id}")
    
    response = await client.get(
        f"{BASE_URL}/tasks/{task_id}/data/advanced",
        params={
            "resample_freq": "W",
            "fill_missing_dates": "true",
            "limit": 10
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get weekly resampled data: {response.text}")
        return False
    
    data = response.json()
    if not data or "data" not in data:
        logger.error("No data returned for weekly resampling")
        return False
    
    logger.info(f"Successfully retrieved {len(data['data'])} weekly resampled data points")
    logger.info(f"Resampling metadata: {json.dumps(data['metadata']['statistics']['resampling'], indent=2)}")
    return True

async def test_missing_data_handling(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test missing data handling."""
    logger.info(f"Testing missing data handling for task {task_id}")
    
    response = await client.get(
        f"{BASE_URL}/tasks/{task_id}/data/advanced",
        params={
            "fill_missing_dates": "true",
            "limit": 10
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get data with missing dates filled: {response.text}")
        return False
    
    data = response.json()
    if not data or "data" not in data:
        logger.error("No data returned for missing data handling")
        return False
    
    logger.info(f"Successfully retrieved {len(data['data'])} data points with missing dates handling")
    logger.info(f"Missing dates filled: {data['metadata']['statistics']['missing_dates_filled']}")
    return True

async def run_all_tests():
    """Run all core feature tests."""
    try:
        async with httpx.AsyncClient() as client:
            # Create a test task
            task = await create_task(client)
            task_id = task["id"]
            
            # Wait for the task to complete
            logger.info(f"Waiting for task {task_id} to complete...")
            if not await wait_for_task_completion(client, task_id):
                logger.error("Task did not complete successfully")
                return False
            
            # Run all tests
            results = {}
            
            # Test basic data retrieval
            logger.info("Running basic data retrieval test...")
            results["basic_data_retrieval"] = await test_basic_data_retrieval(client, task_id)
            
            # Test ME frequency aggregation
            logger.info("Running ME frequency aggregation test...")
            results["me_frequency_aggregation"] = await test_me_frequency_aggregation(client, task_id)
            
            # Test weekly resampling
            logger.info("Running weekly resampling test...")
            results["weekly_resampling"] = await test_weekly_resampling(client, task_id)
            
            # Test missing data handling
            logger.info("Running missing data handling test...")
            results["missing_data_handling"] = await test_missing_data_handling(client, task_id)
            
            # Check if all tests passed
            all_passed = all(results.values())
            if all_passed:
                logger.info("All core feature tests passed!")
            else:
                logger.error("Some core feature tests failed:")
                for test_name, passed in results.items():
                    logger.error(f"  {test_name}: {'PASSED' if passed else 'FAILED'}")
            
            return all_passed
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running tests: {e}")
        sys.exit(1)
