"""
Test script for all frequency aggregations.

This script tests all available frequency aggregations to ensure they're working correctly.
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

# All frequencies to test
FREQUENCIES = ["D", "W", "ME", "QE", "YE"]

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

async def test_frequency_aggregation(client: httpx.AsyncClient, task_id: int, frequency: str) -> bool:
    """Test aggregation with a specific frequency."""
    logger.info(f"Testing {frequency} frequency aggregation for task {task_id}")
    
    try:
        response = await client.get(
            f"{BASE_URL}/tasks/{task_id}/aggregated",
            params={
                "group_by": frequency,
                "include_ohlc": "true",
                "include_volume": "true",
                "include_returns": "true"
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get aggregated data for {frequency} frequency: {response.text}")
            return False
        
        data = response.json()
        if not data:
            logger.warning(f"No data returned for {frequency} frequency")
            return True  # Consider this a success if the API returns an empty list
        
        logger.info(f"Successfully retrieved {len(data)} aggregated data points for {frequency} frequency")
        
        # Log the first data point as a sample
        if data:
            logger.info(f"Sample {frequency} aggregated data: {json.dumps(data[0], indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"Exception during {frequency} frequency test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def run_all_tests():
    """Run tests for all frequencies."""
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
            
            # Test each frequency
            results = {}
            for frequency in FREQUENCIES:
                logger.info(f"Testing {frequency} frequency...")
                results[frequency] = await test_frequency_aggregation(client, task_id, frequency)
            
            # Check if all tests passed
            all_passed = all(results.values())
            if all_passed:
                logger.info("All frequency tests passed!")
            else:
                logger.error("Some frequency tests failed:")
                for freq, passed in results.items():
                    logger.error(f"  {freq}: {'PASSED' if passed else 'FAILED'}")
            
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
