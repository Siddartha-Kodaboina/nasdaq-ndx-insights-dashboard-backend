#!/usr/bin/env python3
"""
Test script for data retrieval endpoints.

This script tests the enhanced data retrieval endpoints for the Stack Analysis API,
including filtering, transformation, and aggregation capabilities.
"""

import asyncio
import json
import sys
import logging
import time
from datetime import datetime, timedelta
import httpx
import pandas as pd
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:8000"

# Test data
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL"]
TEST_TASK_TYPE = "explore_stock"

async def create_task(client: httpx.AsyncClient, ticker: str) -> Dict[str, Any]:
    """Create a task for testing."""
    task_data = {
        "task_type": TEST_TASK_TYPE,
        "parameters": {
            "ticker": ticker,
            "from_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "to_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "source": "source_a"  # Using a valid source from DataSourceEnum
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
        response = await client.get(f"{BASE_URL}/tasks/{task_id}/status")
        if response.status_code != 200:
            logger.error(f"Failed to get task status: {response.text}")
            return False
        
        status_data = response.json()
        status = status_data.get("status")
        progress = status_data.get("progress", 0)
        
        # Log the full status response for debugging
        logger.info(f"Task {task_id} status response: {status_data}")
        
        # Check if task is completed (case-insensitive comparison)
        if status.upper() == "COMPLETED":
            logger.info(f"Task {task_id} completed successfully")
            return True
            
        # Check if task failed or was cancelled (case-insensitive comparison)
        if status.upper() in ["FAILED", "CANCELLED"]:
            logger.error(f"Task {task_id} ended with status: {status}")
            return False
        
        # If progress is 100% but status is not completed (case-insensitive), there might be an issue
        if progress == 100.0 and status.upper() != "COMPLETED":
            logger.warning(f"Task {task_id} shows 100% progress but status is still {status}. This might indicate an issue.")
        
        logger.info(f"Task {task_id} still processing. Status: {status}, Progress: {progress}%")
        
        # Wait before checking again
        await asyncio.sleep(2)
    
    logger.error(f"Timeout waiting for task {task_id} to complete")
    return False

async def test_basic_data_retrieval(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test basic data retrieval."""
    logger.info(f"Testing basic data retrieval for task {task_id}")
    
    # Get task data
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

async def test_filtered_data_retrieval(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test filtered data retrieval."""
    logger.info(f"Testing filtered data retrieval for task {task_id}")
    
    # Test with date range filter
    from_date = (datetime.now() - timedelta(days=180)).isoformat()
    to_date = datetime.now().isoformat()
    
    response = await client.get(
        f"{BASE_URL}/tasks/{task_id}/data/advanced",
        params={
            "start_date": from_date,
            "end_date": to_date,
            "limit": 10
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get filtered task data: {response.text}")
        return False
    
    data = response.json()
    if not data or "data" not in data:
        logger.error("No data returned from filtered data retrieval endpoint")
        return False
    
    logger.info(f"Filtered data retrieval successful. Received {len(data['data'])} records.")
    logger.info(f"Metadata: {json.dumps(data['metadata'], indent=2)}")
    return True

async def test_transformed_data_retrieval(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test transformed data retrieval endpoint."""
    logger.info(f"Testing transformed data retrieval for task {task_id}")
    
    # Test with resampling transformation
    response = await client.get(
        f"{BASE_URL}/tasks/{task_id}/data/advanced",
        params={
            "resample_freq": "W",  # Weekly resampling
            "fill_missing_dates": "true",
            "limit": 10
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get transformed task data: {response.text}")
        return False
    
    data = response.json()
    if not data or "data" not in data:
        logger.error("No data returned from transformed data retrieval endpoint")
        return False
    
    logger.info(f"Transformed data retrieval successful. Received {len(data['data'])} records.")
    logger.info(f"Transformation metadata: {json.dumps(data['metadata']['statistics'], indent=2)}")
    return True

async def test_aggregated_data_retrieval(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test aggregated data retrieval endpoint."""
    logger.info(f"Testing aggregated data retrieval for task {task_id}")
    
    # Test with monthly aggregation
    try:
        logger.info("Sending request to aggregated data endpoint with ME frequency...")
        response = await client.get(
            f"{BASE_URL}/tasks/{task_id}/aggregated",
            params={
                "group_by": "ME",  # Month-end aggregation
                "include_ohlc": "true",
                "include_volume": "true",
                "include_returns": "true"
            }
        )
        
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        if response.status_code != 200:
            logger.error(f"Failed to get aggregated task data: {response.text}")
            # Try to parse the error response if possible
            try:
                error_data = response.json()
                logger.error(f"Error details: {json.dumps(error_data, indent=2)}")
            except Exception as e:
                logger.error(f"Could not parse error response as JSON: {e}")
                logger.error(f"Raw response text: {response.text}")
            return False
        
        data = response.json()
        if not data:
            logger.error("No data returned from aggregated data retrieval endpoint")
            return False
        
        logger.info(f"Aggregated data retrieval successful. Received {len(data)} records.")
        logger.info(f"Sample aggregated data: {json.dumps(data[0], indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Exception during aggregated data retrieval test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def run_all_tests():
    """Run all tests."""
    try:
        # Create a test client
        async with httpx.AsyncClient() as client:
            # Create a test task
            task = await create_task(client, TEST_TICKERS[0])
            task_id = task["id"]
            
            # Wait for the task to complete
            logger.info(f"Waiting for task {task_id} to complete...")
            if not await wait_for_task_completion(client, task_id, max_wait_seconds=120):
                logger.error("Task did not complete successfully")
                return False
            
            # Add a small delay to ensure data is fully processed
            logger.info("Task completed. Waiting 2 seconds before testing data retrieval...")
            await asyncio.sleep(2)
            
            # Run all tests
            results = {}
            
            # Test basic data retrieval
            logger.info("Running basic data retrieval test...")
            results["basic_data_retrieval"] = await test_basic_data_retrieval(client, task_id)
            
            # Test filtered data retrieval
            logger.info("Running filtered data retrieval test...")
            results["filtered_data_retrieval"] = await test_filtered_data_retrieval(client, task_id)
            
            # Test transformed data retrieval
            logger.info("Running transformed data retrieval test...")
            results["transformed_data_retrieval"] = await test_transformed_data_retrieval(client, task_id)
            
            # Test aggregated data retrieval
            logger.info("Running aggregated data retrieval test...")
            results["aggregated_data_retrieval"] = await test_aggregated_data_retrieval(client, task_id)
            
            # Check if all tests passed
            all_passed = all(results.values())
            if all_passed:
                logger.info("All tests passed!")
            else:
                logger.error("Some tests failed:")
                for test_name, passed in results.items():
                    logger.error(f"  {test_name}: {'PASSED' if passed else 'FAILED'}")
            
            return all_passed
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        raise

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
