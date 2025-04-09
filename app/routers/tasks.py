from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import logging

from app.utils.async_db_client import get_async_db
from app.models import Task, TaskStatus, TaskType, StockData
from app.schemas import (
    TaskCreate, TaskResponse, TaskStatusResponse, StockDataResponse,
    StockDataFilter, StockDataTransform, StockDataAggregation,
    PaginationParams, StockDataResponseWithMetadata, AggregatedStockData,
    ResampleFrequency, NormalizationMethod
)
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
    """Get data for a specific task with basic pagination and filtering.
    
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
    try:
        data = await AsyncTaskService.get_task_data(
            task_id=task_id,
            ticker=ticker,
            skip=skip,
            limit=limit,
            db=db
        )
    except Exception as e:
        logger.exception(f"Error retrieving data for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving data: {str(e)}"
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

@router.get("/{task_id}/data/advanced", response_model=StockDataResponseWithMetadata)
async def get_filtered_task_data(
    task_id: int,
    db: AsyncSession = Depends(get_async_db),
    # Filter parameters
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_volume: Optional[float] = None,
    max_volume: Optional[float] = None,
    price_field: str = "close",
    # Pagination parameters
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "date",
    sort_desc: bool = False,
    # Transformation parameters
    resample_freq: Optional[str] = None,
    normalize_method: Optional[str] = None,
    calculate_returns_period: Optional[int] = None,
    fill_missing_dates: bool = False
):
    """Get data for a specific task with advanced filtering, pagination, and transformation options.
    
    Args:
        task_id: ID of the task to get data for
        ticker: Filter by ticker symbol
        start_date: Filter by start date
        end_date: Filter by end date
        min_price: Filter by minimum price
        max_price: Filter by maximum price
        min_volume: Filter by minimum volume
        max_volume: Filter by maximum volume
        price_field: Price field to filter on (open, high, low, close)
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (for pagination)
        sort_by: Field to sort by (date, open, high, low, close, volume)
        sort_desc: Sort in descending order if True, ascending if False
        resample_freq: Frequency to resample data to (D, W, ME, QE, YE)
        normalize_method: Method to normalize data (min-max, z-score, percent_change, first_value)
        calculate_returns_period: Period for return calculation
        fill_missing_dates: Fill missing dates with NaN values
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
    
    # Create filter parameters
    filter_params = StockDataFilter(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        min_price=min_price,
        max_price=max_price,
        min_volume=min_volume,
        max_volume=max_volume,
        price_field=price_field
    )
    
    # Create pagination parameters
    pagination_params = PaginationParams(
        skip=skip,
        limit=limit,
        sort_by=sort_by,
        sort_desc=sort_desc
    )
    
    # Create transformation parameters
    transform_params = None
    if resample_freq or normalize_method or calculate_returns_period or fill_missing_dates:
        transform_params = StockDataTransform(
            resample_freq=ResampleFrequency(resample_freq) if resample_freq else None,
            normalize_method=NormalizationMethod(normalize_method) if normalize_method else None,
            calculate_returns=calculate_returns_period,
            fill_missing_dates=fill_missing_dates
        )
    
    # Get the transformed task data
    try:
        result = await AsyncTaskService.get_transformed_task_data(
            task_id=task_id,
            filter_params=filter_params,
            transform_params=transform_params,
            pagination=pagination_params,
            db=db
        )
        return result
    except Exception as e:
        logger.exception(f"Error retrieving transformed data for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving transformed data: {str(e)}"
        )

@router.get("/{task_id}/aggregated", response_model=List[AggregatedStockData])
async def get_aggregated_task_data(
    task_id: int,
    group_by: ResampleFrequency,
    include_ohlc: bool = True,
    include_volume: bool = True,
    include_returns: bool = False,
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_volume: Optional[float] = None,
    max_volume: Optional[float] = None,
    price_field: str = "close",
    db: AsyncSession = Depends(get_async_db)
):
    """Get aggregated data for a specific task.
    
    Args:
        task_id: ID of the task to get data for
        group_by: Frequency to group data by (D, W, ME, QE, YE)
        include_ohlc: Include OHLC data in aggregation
        include_volume: Include volume data in aggregation
        include_returns: Include returns data in aggregation
        ticker: Filter by ticker symbol
        start_date: Filter by start date
        end_date: Filter by end date
        min_price: Filter by minimum price
        max_price: Filter by maximum price
        min_volume: Filter by minimum volume
        max_volume: Filter by maximum volume
        price_field: Price field to filter on (open, high, low, close)
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
    
    # Create filter parameters
    filter_params = StockDataFilter(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        min_price=min_price,
        max_price=max_price,
        min_volume=min_volume,
        max_volume=max_volume,
        price_field=price_field
    )
    
    # Create aggregation parameters
    aggregation_params = StockDataAggregation(
        group_by=group_by,
        include_ohlc=include_ohlc,
        include_volume=include_volume,
        include_returns=include_returns
    )
    
    # Check if there is any data for this task
    data_count_query = select(func.count()).select_from(StockData).filter(StockData.task_id == task_id)
    data_count_result = await db.execute(data_count_query)
    data_count = data_count_result.scalar()
    
    if data_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for task {task_id}. The task might have completed but failed to store data."
        )
    
    # Get the aggregated task data
    try:
        logger.debug(f"Calling AsyncTaskService.get_aggregated_task_data with task_id={task_id}")
        logger.debug(f"Filter params: {filter_params}")
        logger.debug(f"Aggregation params: {aggregation_params}")
        
        result = await AsyncTaskService.get_aggregated_task_data(
            task_id=task_id,
            filter_params=filter_params,
            aggregation_params=aggregation_params,
            db=db
        )
        
        logger.debug(f"Successfully retrieved {len(result)} aggregated data points")
        if result:
            logger.debug(f"First result item: ticker={result[0].ticker}, period_start={result[0].period_start}, "
                        f"period_end={result[0].period_end}, data_points={result[0].data_points}")
        
        return result
    except Exception as e:
        logger.exception(f"Error retrieving aggregated data for task {task_id}: {str(e)}")
        import traceback
        logger.error(f"Detailed traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving aggregated data: {str(e)}"
        )