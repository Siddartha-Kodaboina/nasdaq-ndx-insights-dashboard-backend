"""
Comprehensive Backend Testing Script

This script performs a full test of all critical backend functionality:
1. Task creation and monitoring
2. Data retrieval (raw and advanced)
3. All frequency aggregations (D, W, ME, QE, YE)
4. Data transformation features (resampling, missing data handling)
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

# Supported frequencies
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

async def test_task_creation_and_monitoring(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Test task creation and monitoring."""
    logger.info("Testing task creation and monitoring...")
    
    # Create task
    task = await create_task(client)
    task_id = task["id"]
    logger.info(f"Created task with ID: {task_id}")
    
    # Wait for task completion
    success = await wait_for_task_completion(client, task_id)
    if not success:
        raise Exception(f"Task {task_id} did not complete successfully")
    
    return task

async def test_raw_data_retrieval(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test raw data retrieval."""
    logger.info(f"Testing raw data retrieval for task {task_id}")
    
    response = await client.get(f"{BASE_URL}/tasks/{task_id}/data")
    if response.status_code != 200:
        logger.error(f"Failed to get raw data: {response.text}")
        return False
    
    data = response.json()
    if not data:
        logger.error("No raw data returned")
        return False
    
    logger.info(f"Successfully retrieved {len(data)} raw data points")
    logger.info(f"Sample raw data: {json.dumps(data[0], indent=2)}")
    return True

async def test_advanced_data_retrieval(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test advanced data retrieval with parameters."""
    logger.info(f"Testing advanced data retrieval for task {task_id}")
    
    response = await client.get(
        f"{BASE_URL}/tasks/{task_id}/data/advanced",
        params={
            "limit": 10,
            "fill_missing_dates": "true"
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get advanced data: {response.text}")
        return False
    
    data = response.json()
    if not data or "data" not in data:
        logger.error("No advanced data returned")
        return False
    
    logger.info(f"Successfully retrieved {len(data['data'])} advanced data points")
    logger.info(f"Advanced data metadata: {json.dumps(data['metadata'], indent=2)}")
    return True

async def test_frequency_aggregations(client: httpx.AsyncClient, task_id: int) -> Dict[str, bool]:
    """Test all frequency aggregations."""
    logger.info(f"Testing all frequency aggregations for task {task_id}")
    
    results = {}
    
    for freq in FREQUENCIES:
        logger.info(f"Testing {freq} frequency aggregation...")
        
        response = await client.get(
            f"{BASE_URL}/tasks/{task_id}/aggregated",
            params={
                "group_by": freq,
                "include_ohlc": "true",
                "include_volume": "true",
                "include_returns": "true"
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get {freq} frequency aggregated data: {response.text}")
            results[freq] = False
            continue
        
        data = response.json()
        if not data:
            logger.error(f"No data returned for {freq} frequency aggregation")
            results[freq] = False
            continue
        
        logger.info(f"Successfully retrieved {len(data)} aggregated data points for {freq} frequency")
        logger.info(f"Sample {freq} aggregated data: {json.dumps(data[0], indent=2)}")
        results[freq] = True
    
    return results

async def test_resampling_features(client: httpx.AsyncClient, task_id: int) -> Dict[str, bool]:
    """Test resampling features."""
    logger.info(f"Testing resampling features for task {task_id}")
    
    results = {}
    
    # Test weekly resampling
    logger.info("Testing weekly resampling...")
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
        results["weekly_resampling"] = False
    else:
        data = response.json()
        if not data or "data" not in data:
            logger.error("No data returned for weekly resampling")
            results["weekly_resampling"] = False
        else:
            logger.info(f"Successfully retrieved {len(data['data'])} weekly resampled data points")
            logger.info(f"Weekly resampling metadata: {json.dumps(data['metadata']['statistics']['resampling'], indent=2)}")
            results["weekly_resampling"] = True
    
    # Test missing data handling
    logger.info("Testing missing data handling...")
    response = await client.get(
        f"{BASE_URL}/tasks/{task_id}/data/advanced",
        params={
            "fill_missing_dates": "true",
            "limit": 10
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get data with missing dates filled: {response.text}")
        results["missing_data_handling"] = False
    else:
        data = response.json()
        if not data or "data" not in data:
            logger.error("No data returned for missing data handling")
            results["missing_data_handling"] = False
        else:
            logger.info(f"Successfully retrieved {len(data['data'])} data points with missing dates handling")
            logger.info(f"Missing dates filled: {data['metadata']['statistics']['missing_dates_filled']}")
            results["missing_data_handling"] = True
    
    return results

async def run_comprehensive_tests():
    """Run comprehensive backend tests."""
    try:
        async with httpx.AsyncClient() as client:
            # Test task creation and monitoring
            task = await test_task_creation_and_monitoring(client)
            task_id = task["id"]
            
            # Run all tests
            test_results = {}
            
            # Test raw data retrieval
            logger.info("Running raw data retrieval test...")
            test_results["raw_data_retrieval"] = await test_raw_data_retrieval(client, task_id)
            
            # Test advanced data retrieval
            logger.info("Running advanced data retrieval test...")
            test_results["advanced_data_retrieval"] = await test_advanced_data_retrieval(client, task_id)
            
            # Test frequency aggregations
            logger.info("Running frequency aggregation tests...")
            frequency_results = await test_frequency_aggregations(client, task_id)
            for freq, result in frequency_results.items():
                test_results[f"{freq}_frequency"] = result
            
            # Test resampling features
            logger.info("Running resampling feature tests...")
            resampling_results = await test_resampling_features(client, task_id)
            for feature, result in resampling_results.items():
                test_results[feature] = result
            
            # Check if all tests passed
            all_passed = all(test_results.values())
            
            # Print test summary
            logger.info("\n" + "="*50)
            logger.info("COMPREHENSIVE BACKEND TEST SUMMARY")
            logger.info("="*50)
            
            for test_name, passed in test_results.items():
                logger.info(f"{test_name.ljust(30)}: {'PASSED' if passed else 'FAILED'}")
            
            logger.info("="*50)
            logger.info(f"OVERALL RESULT: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED'}")
            logger.info("="*50)
            
            return all_passed
    except Exception as e:
        logger.error(f"Error running comprehensive tests: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(run_comprehensive_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running tests: {e}")
        sys.exit(1)
