import logging
import time
from datetime import datetime, timedelta
import json

from app.models import Task, TaskStatus, TaskType, DataSource
from app.utils.db_client import get_db, create_tables
from app.workers import process_task, TaskProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_task(task_type, parameters):
    """Create a test task in the database."""
    with get_db() as db:
        task = Task(
            task_type=task_type,
            status=TaskStatus.PENDING,
            parameters=parameters
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return task.id

def test_explore_stock():
    """Test the explore stock task processor."""
    logger.info("Testing explore stock task processor...")
    
    # Create a test task
    parameters = {
        "ticker": "AAPL",
        "source": DataSource.SOURCE_A.value,  # Use the string value for JSON serialization
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = create_test_task(TaskType.EXPLORE_STOCK, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Process the task
    result = process_task(task_id)
    
    # Print the result
    logger.info(f"Task processing result: {json.dumps(result, indent=2)}")
    
    # Verify the task status in the database
    with get_db() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        logger.info(f"Task status in database: {task.status}")
    
    logger.info("Explore stock task test completed!")
    return result

def test_compare_stocks():
    """Test the compare stocks task processor."""
    logger.info("Testing compare stocks task processor...")
    
    # Create a test task
    parameters = {
        "ticker1": "AAPL",
        "ticker2": "MSFT",
        "field": "close",
        "source": DataSource.SOURCE_B.value,  # Use the string value for JSON serialization
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = create_test_task(TaskType.COMPARE_STOCKS, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Process the task
    result = process_task(task_id)
    
    # Print the result
    logger.info(f"Task processing result: {json.dumps(result, indent=2)}")
    
    # Verify the task status in the database
    with get_db() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        logger.info(f"Task status in database: {task.status}")
    
    logger.info("Compare stocks task test completed!")
    return result

def test_stock_vs_index():
    """Test the stock vs index task processor."""
    logger.info("Testing stock vs index task processor...")
    
    # Create a test task
    parameters = {
        "ticker": "AAPL",
        "index_ticker": "^NDX",
        "source": DataSource.SOURCE_A.value,  # Use the string value for JSON serialization
        "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "to_date": datetime.now().isoformat()
    }
    
    task_id = create_test_task(TaskType.STOCK_VS_INDEX, parameters)
    logger.info(f"Created test task with ID: {task_id}")
    
    # Process the task
    result = process_task(task_id)
    
    # Print the result
    logger.info(f"Task processing result: {json.dumps(result, indent=2)}")
    
    # Verify the task status in the database
    with get_db() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        logger.info(f"Task status in database: {task.status}")
    
    logger.info("Stock vs index task test completed!")
    return result

if __name__ == "__main__":
    # Ensure database tables are created
    create_tables()
    
    # Run the tests
    try:
        test_explore_stock()
        test_compare_stocks()
        test_stock_vs_index()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.exception(f"Error during testing: {e}")
