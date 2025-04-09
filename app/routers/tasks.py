from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
import logging

from app.utils.async_db_client import get_async_db
from app.models import Task, TaskStatus, TaskType
from app.schemas import TaskCreate, TaskResponse, TaskStatusResponse, StockDataResponse
from app.utils.validators import validate_request_body
from app.services.async_task_service import AsyncTaskService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
)

@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
@validate_request_body(TaskCreate)
async def create_task(task_data: Dict[str, Any], db: AsyncSession = Depends(get_async_db)):
    """Create a new data analysis task."""
    logger.info(f"Router received create_task request with task_type: {task_data.get('task_type')}")
    logger.info(f"Router received db object type: {type(db)}, ID: {id(db)}")
    
    try:
        task = await AsyncTaskService.create_task(task_data, db)
        logger.info(f"Task created successfully with ID: {task.id}")
        
        # Convert the SQLAlchemy model to a dictionary manually
        task_dict = {
            "id": task.id,
            "task_type": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
            "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
            "parameters": task.parameters,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "progress": task.progress,
            "error_message": task.error_message,
            "estimated_completion_time": task.estimated_completion_time
        }
        
        # Create the response using the dictionary
        response = TaskResponse(**task_dict)
        logger.info(f"Returning response: {response.model_dump()}")
        return response
    except Exception as e:
        logger.error(f"Error in create_task endpoint: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_async_db)):
    """Get a specific task by ID."""
    task = await AsyncTaskService.get_task(task_id, db)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Convert the SQLAlchemy model to a dictionary manually
    task_dict = {
        "id": task.id,
        "task_type": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
        "parameters": task.parameters,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "progress": task.progress,
        "error_message": task.error_message,
        "estimated_completion_time": task.estimated_completion_time
    }
    
    # Create the response using the dictionary
    return TaskResponse(**task_dict)

@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: int, db: AsyncSession = Depends(get_async_db)):
    """Get detailed status information for a specific task.
    
    This endpoint provides more detailed status information than the basic task endpoint,
    including progress percentage, elapsed time, and estimated time remaining.
    """
    task_status = await AsyncTaskService.get_task_status(task_id, db)
    if not task_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    # The task_status is already a Pydantic model, so no need to convert
    return task_status

@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "created_at",
    sort_desc: bool = True,
    db: AsyncSession = Depends(get_async_db)
):
    """List all tasks with optional filtering.
    
    Args:
        status: Filter by task status
        task_type: Filter by task type
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (for pagination)
        sort_by: Field to sort by (created_at, updated_at, id, status, task_type)
        sort_desc: Sort in descending order if True, ascending if False
        db: Database session
    """
    tasks = await AsyncTaskService.list_tasks(
        status=status, 
        task_type=task_type, 
        skip=skip,
        limit=limit,
        sort_by=sort_by,
        sort_desc=sort_desc,
        db=db
    )
    # Convert SQLAlchemy models to Pydantic models
    result = []
    for task in tasks:
        task_dict = {
            "id": task.id,
            "task_type": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
            "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
            "parameters": task.parameters,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "progress": task.progress,
            "error_message": task.error_message,
            "estimated_completion_time": task.estimated_completion_time
        }
        result.append(TaskResponse(**task_dict))
    return result

@router.get("/{task_id}/data", response_model=List[StockDataResponse])
async def get_task_data(
    task_id: int,
    ticker: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_async_db)
):
    """Get data for a specific task with pagination and filtering.
    
    Args:
        task_id: ID of the task to get data for
        ticker: Filter by ticker symbol
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (for pagination)
        db: Database session
    """
    # First check if the task exists
    task = await AsyncTaskService.get_task(task_id, db)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Then check if the task is completed
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not completed yet. Current status: {task.status}"
        )
    
    # Get the task data
    data = await AsyncTaskService.get_task_data(
        task_id=task_id,
        ticker=ticker,
        skip=skip,
        limit=limit,
        db=db
    )
    
    # Convert SQLAlchemy models to Pydantic models
    result = []
    for item in data:
        stock_data_dict = {
            "id": item.id,
            "task_id": item.task_id,
            "ticker": item.ticker,
            "date": item.date,
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume,
            "source": item.source.value if hasattr(item.source, 'value') else str(item.source)
        }
        result.append(StockDataResponse(**stock_data_dict))
    return result