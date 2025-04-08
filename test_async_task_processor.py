#!/usr/bin/env python
"""
Test script for the async task processor.

This script tests the async task processor by creating test tasks and
processing them asynchronously.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from sqlalchemy import text

from app.models import Task, TaskType, TaskStatus, DataSource
from app.utils.async_db_client import get_async_db, create_async_tables
from app.async_workers import process_task_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_test_task(task_type: TaskType, parameters: dict) -> int:
    """Create a test task in the database.
    
    Args:
        task_type: Type of task to create
        parameters: Task parameters
        
    Returns:
        ID of the created task
    """
    async with get_async_db() as db:
        task = Task(
            task_type=task_type,
            parameters=parameters,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        
        return task.id

async def test_explore_stock():
    """Test the async explore stock task processor."""
    logger.info("Testing async explore stock task processor...")
    
    # Create a test task
    parameters = {
        "ticker": "AAPL",
        "source": DataSource.SOURCE_A.value,  # Use the string value for JSON serialization
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = await create_test_task(TaskType.EXPLORE_STOCK, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Process the task
    result = await process_task_async(task_id)
    
    # Print the result
    logger.info(f"Task processing result: {json.dumps(result, indent=2)}")
    
    # Verify the task status in the database
    async with get_async_db() as db:
        stmt = text("SELECT status FROM tasks WHERE id = :task_id")
        result_proxy = await db.execute(stmt, {"task_id": task_id})
        task_status = result_proxy.scalar_one_or_none()
        logger.info(f"Task status in database: {task_status}")
    
    logger.info("Async explore stock task test completed!")
    return result

async def test_compare_stocks():
    """Test the async compare stocks task processor."""
    logger.info("Testing async compare stocks task processor...")
    
    # Create a test task
    parameters = {
        "ticker1": "AAPL",
        "ticker2": "MSFT",
        "field": "close",
        "source": DataSource.SOURCE_B.value,  # Use the string value for JSON serialization
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = await create_test_task(TaskType.COMPARE_STOCKS, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Process the task
    result = await process_task_async(task_id)
    
    # Print the result
    logger.info(f"Task processing result: {json.dumps(result, indent=2)}")
    
    # Verify the task status in the database
    async with get_async_db() as db:
        stmt = text("SELECT status FROM tasks WHERE id = :task_id")
        result_proxy = await db.execute(stmt, {"task_id": task_id})
        task_status = result_proxy.scalar_one_or_none()
        logger.info(f"Task status in database: {task_status}")
    
    logger.info("Async compare stocks task test completed!")
    return result

async def test_stock_vs_index():
    """Test the async stock vs index task processor."""
    logger.info("Testing async stock vs index task processor...")
    
    # Create a test task
    parameters = {
        "ticker": "AAPL",
        "index_ticker": "^NDX",
        "source": DataSource.SOURCE_A.value,  # Use the string value for JSON serialization
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = await create_test_task(TaskType.STOCK_VS_INDEX, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Process the task
    result = await process_task_async(task_id)
    
    # Print the result
    logger.info(f"Task processing result: {json.dumps(result, indent=2)}")
    
    # Verify the task status in the database
    async with get_async_db() as db:
        stmt = text("SELECT status FROM tasks WHERE id = :task_id")
        result_proxy = await db.execute(stmt, {"task_id": task_id})
        task_status = result_proxy.scalar_one_or_none()
        logger.info(f"Task status in database: {task_status}")
    
    logger.info("Async stock vs index task test completed!")
    return result

async def test_task_cancellation():
    """Test task cancellation."""
    logger.info("Testing task cancellation...")
    
    # Create a test task
    parameters = {
        "ticker": "AAPL",
        "source": DataSource.SOURCE_A.value,
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = await create_test_task(TaskType.EXPLORE_STOCK, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Create a task for processing
    task = asyncio.create_task(process_task_async(task_id))
    
    # Wait a bit to let the task start
    await asyncio.sleep(0.5)
    
    # Cancel the task
    task.cancel()
    
    try:
        # Wait for the task to be cancelled
        result = await task
        logger.info(f"Task result: {json.dumps(result, indent=2)}")
    except asyncio.CancelledError:
        logger.info("Task was cancelled as expected")
    
    # Verify the task status in the database
    async with get_async_db() as db:
        stmt = text("SELECT status FROM tasks WHERE id = :task_id")
        result_proxy = await db.execute(stmt, {"task_id": task_id})
        task_status = result_proxy.scalar_one_or_none()
        logger.info(f"Task status in database: {task_status}")
    
    logger.info("Task cancellation test completed!")

async def test_task_timeout():
    """Test task timeout."""
    logger.info("Testing task timeout...")
    
    # Create a test task
    parameters = {
        "ticker": "AAPL",
        "source": DataSource.SOURCE_A.value,
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = await create_test_task(TaskType.EXPLORE_STOCK, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Process the task with a very short timeout
    result = await process_task_async(task_id, timeout=0.1)
    
    # Print the result
    logger.info(f"Task processing result: {json.dumps(result, indent=2)}")
    
    # Verify the task status in the database
    async with get_async_db() as db:
        stmt = text("SELECT status FROM tasks WHERE id = :task_id")
        result_proxy = await db.execute(stmt, {"task_id": task_id})
        task_status = result_proxy.scalar_one_or_none()
        logger.info(f"Task status in database: {task_status}")
    
    logger.info("Task timeout test completed!")

async def main():
    """Run all tests."""
    # Create tables if they don't exist
    await create_async_tables()
    
    # Clear the database before running tests
    async with get_async_db() as db:
        await db.execute(text("DELETE FROM stock_data"))
        await db.execute(text("DELETE FROM tasks"))
        await db.commit()
    
    # Run tests
    await test_explore_stock()
    await test_compare_stocks()
    await test_stock_vs_index()
    await test_task_cancellation()
    await test_task_timeout()

if __name__ == "__main__":
    asyncio.run(main())
