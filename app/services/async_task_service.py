from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, asc
import logging
import json

from app.models import Task, TaskStatus, TaskType, StockData
from app.schemas import TaskCreate, TaskResponse, TaskStatusResponse
from app.queue.job_queue import job_queue
from app.utils.async_db_client import get_async_db, AsyncSessionLocal
from app.async_workers import process_task_async

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger(__name__)

class AsyncTaskService:
    """Service for managing tasks asynchronously."""
    
    @staticmethod
    async def create_task(task_data: Dict[str, Any], db: AsyncSession) -> Task:
        """Create a new task and enqueue it for processing.
        
        Args:
            task_data: Task data
            db: Database session
            
        Returns:
            Created task
        """
        # Log the type of db object we received
        logger.info(f"DB object type: {type(db)}, ID: {id(db)}")
        
        # Serialize the parameters to handle datetime objects
        serialized_parameters = json.loads(json.dumps(task_data["parameters"], cls=DateTimeEncoder))
        
        # Convert task_type string to enum value
        task_type_str = task_data["task_type"]
        logger.info(f"Converting task_type string '{task_type_str}' to enum value")
        
        # Get the appropriate enum value
        try:
            task_type_enum = TaskType[task_type_str.upper()]
            logger.info(f"Converted to enum value: {task_type_enum}")
        except KeyError:
            # Try direct lookup by value
            try:
                task_type_enum = TaskType(task_type_str)
                logger.info(f"Found enum by value: {task_type_enum}")
            except ValueError:
                logger.error(f"Invalid task_type: {task_type_str}")
                raise ValueError(f"Invalid task_type: {task_type_str}. Valid types are: {[t.value for t in TaskType]}")
        
        # Create a new task in the database
        task = Task(
            task_type=task_type_enum,
            status=TaskStatus.PENDING,
            parameters=serialized_parameters,
            progress=0,
            error_message=None,
            estimated_completion_time=None
        )
        
        try:
            # Add the task to the session
            logger.info(f"Attempting to add task to session: {task.task_type}")
            db.add(task)
            
            # Flush the session to get the ID without committing
            logger.info("Flushing session to get task ID")
            await db.flush()
            await db.refresh(task)
            logger.info(f"Task created with ID: {task.id}")
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            logger.exception("Detailed traceback:")
            raise
        
        logger.info(f"Created task {task.id} of type {task.task_type}")
        
        # Enqueue the task for processing
        job_queue.enqueue(task.id, process_task_async, task.id)
        
        return task
    
    @staticmethod
    async def get_task(task_id: int, db: AsyncSession) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: ID of the task to get
            db: Database session
            
        Returns:
            Task if found, None otherwise
        """
        result = await db.execute(select(Task).filter(Task.id == task_id))
        return result.scalars().first()
    
    @staticmethod
    async def get_task_status(task_id: int, db: AsyncSession) -> Optional[TaskStatusResponse]:
        """Get detailed status information for a task.
        
        Args:
            task_id: ID of the task to get status for
            db: Database session
            
        Returns:
            TaskStatusResponse if found, None otherwise
        """
        task = await AsyncTaskService.get_task(task_id, db)
        if not task:
            return None
            
        # Get job information from the queue
        job = job_queue.get_job_by_task(task_id)
        
        # Calculate elapsed time if the task is in progress
        elapsed_time = None
        if task.status == TaskStatus.IN_PROGRESS and job and job.started_at:
            elapsed_time = (datetime.now() - job.started_at).total_seconds()
        
        # Calculate estimated time remaining
        time_remaining = None
        if task.status == TaskStatus.IN_PROGRESS and task.progress > 0 and task.estimated_completion_time:
            time_remaining = (task.estimated_completion_time - datetime.now()).total_seconds()
            if time_remaining < 0:
                time_remaining = None
        
        return TaskStatusResponse(
            task_id=task.id,
            status=task.status,
            progress=task.progress,
            elapsed_time=elapsed_time,
            estimated_time_remaining=time_remaining,
            error_message=task.error_message,
            created_at=task.created_at,
            updated_at=task.updated_at
        )
    
    @staticmethod
    async def list_tasks(
        status: Optional[str] = None, 
        task_type: Optional[str] = None, 
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        db: AsyncSession = None
    ) -> List[Task]:
        """List tasks with optional filtering.
        
        Args:
            status: Filter by status
            task_type: Filter by task type
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return (for pagination)
            sort_by: Field to sort by (created_at, updated_at, id, status, task_type)
            sort_desc: Sort in descending order if True, ascending if False
            db: Database session
            
        Returns:
            List of tasks
        """
        # Start with a base query
        query = select(Task)
        
        # Apply filters
        if status:
            query = query.filter(Task.status == status)
        
        if task_type:
            query = query.filter(Task.task_type == task_type)
        
        # Apply sorting
        sort_column = getattr(Task, sort_by, Task.created_at)
        if sort_desc:
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def get_task_data(
        task_id: int, 
        ticker: Optional[str] = None, 
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = None
    ) -> List[StockData]:
        """Get data for a task.
        
        Args:
            task_id: ID of the task
            ticker: Filter by ticker
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return (for pagination)
            db: Database session
            
        Returns:
            List of stock data records
        """
        query = select(StockData).filter(StockData.task_id == task_id)
        
        if ticker:
            query = query.filter(StockData.ticker == ticker)
        
        query = query.order_by(StockData.date).offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
