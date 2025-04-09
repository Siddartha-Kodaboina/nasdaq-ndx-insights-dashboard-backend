"""
Test script for data transformation capabilities.

This script tests various data transformation features including:
1. Resampling (weekly, monthly, yearly)
2. Normalization (min-max, z-score)
3. Return calculations
4. Missing data handling
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

async def test_resampling(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test data resampling capabilities."""
    logger.info(f"Testing data resampling for task {task_id}")
    
    # Test different resampling frequencies
    resample_freqs = ["D", "W", "M", "Q", "Y"]
    results = {}
    
    for freq in resample_freqs:
        try:
            logger.info(f"Testing {freq} resampling...")
            response = await client.get(
                f"{BASE_URL}/tasks/{task_id}/data/advanced",
                params={
                    "resample_freq": freq,
                    "fill_missing_dates": "true",
                    "limit": 10
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get resampled data for {freq} frequency: {response.text}")
                results[freq] = False
                continue
            
            data = response.json()
            if not data or "data" not in data:
                logger.error(f"No data returned for {freq} resampling")
                results[freq] = False
                continue
            
            logger.info(f"Successfully retrieved {len(data['data'])} resampled data points for {freq} frequency")
            logger.info(f"Resampling metadata for {freq}: {json.dumps(data['metadata']['statistics']['resampling'], indent=2)}")
            results[freq] = True
        except Exception as e:
            logger.error(f"Exception during {freq} resampling test: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            results[freq] = False
    
    # Check if all tests passed
    all_passed = all(results.values())
    if all_passed:
        logger.info("All resampling tests passed!")
    else:
        logger.error("Some resampling tests failed:")
        for freq, passed in results.items():
            logger.error(f"  {freq}: {'PASSED' if passed else 'FAILED'}")
    
    return all_passed

async def test_normalization(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test data normalization capabilities."""
    logger.info(f"Testing data normalization for task {task_id}")
    
    # Test different normalization methods
    normalization_methods = ["min_max", "z_score"]
    results = {}
    
    for method in normalization_methods:
        try:
            logger.info(f"Testing {method} normalization...")
            response = await client.get(
                f"{BASE_URL}/tasks/{task_id}/data/advanced",
                params={
                    "normalize_method": method,
                    "normalize_columns": "open,high,low,close",
                    "limit": 10
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get normalized data for {method} method: {response.text}")
                results[method] = False
                continue
            
            data = response.json()
            if not data or "data" not in data:
                logger.error(f"No data returned for {method} normalization")
                results[method] = False
                continue
            
            logger.info(f"Successfully retrieved {len(data['data'])} normalized data points for {method} method")
            if "normalization" in data["metadata"]["statistics"]:
                logger.info(f"Normalization metadata for {method}: {json.dumps(data['metadata']['statistics']['normalization'], indent=2)}")
            results[method] = True
        except Exception as e:
            logger.error(f"Exception during {method} normalization test: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            results[method] = False
    
    # Check if all tests passed
    all_passed = all(results.values())
    if all_passed:
        logger.info("All normalization tests passed!")
    else:
        logger.error("Some normalization tests failed:")
        for method, passed in results.items():
            logger.error(f"  {method}: {'PASSED' if passed else 'FAILED'}")
    
    return all_passed

async def test_returns_calculation(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test returns calculation capabilities."""
    logger.info(f"Testing returns calculation for task {task_id}")
    
    try:
        response = await client.get(
            f"{BASE_URL}/tasks/{task_id}/data/advanced",
            params={
                "calculate_returns": "true",
                "return_periods": "1,5,20",
                "limit": 10
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get data with returns calculation: {response.text}")
            return False
        
        data = response.json()
        if not data or "data" not in data:
            logger.error("No data returned for returns calculation")
            return False
        
        logger.info(f"Successfully retrieved {len(data['data'])} data points with returns calculation")
        if "returns" in data["metadata"]["statistics"]:
            logger.info(f"Returns calculation metadata: {json.dumps(data['metadata']['statistics']['returns'], indent=2)}")
        
        # Check if returns columns exist in the data
        if data["data"] and len(data["data"]) > 0:
            sample_data = data["data"][0]
            has_returns = any(key.endswith("_return") for key in sample_data.keys())
            if has_returns:
                logger.info("Returns columns found in data")
            else:
                logger.warning("No returns columns found in data")
        
        return True
    except Exception as e:
        logger.error(f"Exception during returns calculation test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def test_missing_data_handling(client: httpx.AsyncClient, task_id: int) -> bool:
    """Test missing data handling capabilities."""
    logger.info(f"Testing missing data handling for task {task_id}")
    
    try:
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
        if "missing_dates_filled" in data["metadata"]["statistics"]:
            logger.info(f"Missing dates filled: {data['metadata']['statistics']['missing_dates_filled']}")
        
        return True
    except Exception as e:
        logger.error(f"Exception during missing data handling test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def run_all_tests():
    """Run all data transformation tests."""
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
            
            # Run all transformation tests
            results = {}
            
            # Test resampling
            logger.info("Running resampling tests...")
            results["resampling"] = await test_resampling(client, task_id)
            
            # Test normalization
            logger.info("Running normalization tests...")
            results["normalization"] = await test_normalization(client, task_id)
            
            # Test returns calculation
            logger.info("Running returns calculation tests...")
            results["returns_calculation"] = await test_returns_calculation(client, task_id)
            
            # Test missing data handling
            logger.info("Running missing data handling tests...")
            results["missing_data_handling"] = await test_missing_data_handling(client, task_id)
            
            # Check if all tests passed
            all_passed = all(results.values())
            if all_passed:
                logger.info("All data transformation tests passed!")
            else:
                logger.error("Some data transformation tests failed:")
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
