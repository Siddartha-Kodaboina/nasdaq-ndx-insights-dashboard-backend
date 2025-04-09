from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, asc, func, and_, or_, between, text
from sqlalchemy.sql import Select
import logging
import json
import pandas as pd

from app.models import Task, TaskStatus, TaskType, StockData
from app.schemas import (
    TaskCreate, TaskResponse, TaskStatusResponse, 
    StockDataFilter, StockDataTransform, StockDataAggregation,
    PaginationParams, StockDataResponseWithMetadata, AggregatedStockData,
    ResampleFrequency, NormalizationMethod
)
from app.queue.job_queue import job_queue
from app.utils.async_db_client import get_async_db, AsyncSessionLocal
from app.async_workers import process_task_async
from app.utils.data_transformer import (
    convert_to_dataframe, resample_ohlcv, normalize_data, 
    calculate_returns, fill_missing_dates
)

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
        """Get data for a task (basic version).
        
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
    
    @staticmethod
    async def get_filtered_task_data(
        task_id: int,
        filter_params: StockDataFilter,
        pagination: PaginationParams,
        db: AsyncSession
    ) -> Tuple[List[StockData], int]:
        """Get filtered data for a task with advanced filtering.
        
        Args:
            task_id: ID of the task
            filter_params: Filter parameters
            pagination: Pagination parameters
            db: Database session
            
        Returns:
            Tuple of (list of stock data records, total count)
        """
        # Build the base query for the task
        query = select(StockData).filter(StockData.task_id == task_id)
        count_query = select(func.count()).select_from(StockData).filter(StockData.task_id == task_id)
        
        # Apply filters
        if filter_params.ticker:
            query = query.filter(StockData.ticker == filter_params.ticker)
            count_query = count_query.filter(StockData.ticker == filter_params.ticker)
        
        if filter_params.start_date:
            query = query.filter(StockData.date >= filter_params.start_date)
            count_query = count_query.filter(StockData.date >= filter_params.start_date)
        
        if filter_params.end_date:
            query = query.filter(StockData.date <= filter_params.end_date)
            count_query = count_query.filter(StockData.date <= filter_params.end_date)
        
        # Apply price filters based on the selected price field
        price_field = getattr(StockData, filter_params.price_field)
        
        if filter_params.min_price is not None:
            query = query.filter(price_field >= filter_params.min_price)
            count_query = count_query.filter(price_field >= filter_params.min_price)
        
        if filter_params.max_price is not None:
            query = query.filter(price_field <= filter_params.max_price)
            count_query = count_query.filter(price_field <= filter_params.max_price)
        
        # Apply volume filters
        if filter_params.min_volume is not None:
            query = query.filter(StockData.volume >= filter_params.min_volume)
            count_query = count_query.filter(StockData.volume >= filter_params.min_volume)
        
        if filter_params.max_volume is not None:
            query = query.filter(StockData.volume <= filter_params.max_volume)
            count_query = count_query.filter(StockData.volume <= filter_params.max_volume)
        
        # Get total count before pagination
        count_result = await db.execute(count_query)
        total_count = count_result.scalar()
        
        # Apply sorting
        sort_column = getattr(StockData, pagination.sort_by)
        if pagination.sort_desc:
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        query = query.offset(pagination.skip).limit(pagination.limit)
        
        # Execute query
        result = await db.execute(query)
        data = result.scalars().all()
        
        return data, total_count
    
    @staticmethod
    async def get_transformed_task_data(
        task_id: int,
        filter_params: StockDataFilter,
        transform_params: StockDataTransform,
        pagination: PaginationParams,
        db: AsyncSession
    ) -> StockDataResponseWithMetadata:
        """Get filtered and transformed data for a task.
        
        Args:
            task_id: ID of the task
            filter_params: Filter parameters
            transform_params: Transformation parameters
            pagination: Pagination parameters
            db: Database session
            
        Returns:
            StockDataResponseWithMetadata containing transformed data and metadata
        """
        logger.info(f"Getting transformed data for task {task_id} with filters: {filter_params} and transformations: {transform_params}")
    
        # Get filtered data
        try:
            data, total_count = await AsyncTaskService.get_filtered_task_data(
                task_id, filter_params, pagination, db
            )
            logger.info(f"Retrieved {len(data)} records for task {task_id} (total: {total_count})")
        except Exception as e:
            logger.error(f"Error retrieving filtered data for task {task_id}: {str(e)}")
            raise
        
        # If no data or no transformations requested, return early
        if not data or not transform_params:
            return StockDataResponseWithMetadata(
                data=data,
                metadata={
                    "total_count": total_count,
                    "page_count": len(data),
                    "has_more": total_count > (pagination.skip + len(data)),
                    "statistics": {}
                }
            )
        
        # Convert to DataFrame for transformations
        try:
            logger.info(f"Converting {len(data)} records to DataFrame for transformations")
            df = convert_to_dataframe([{
                "ticker": item.ticker,
                "date": item.date,
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume,
                "source": item.source
            } for item in data])
            
            # Check for and handle duplicate date indices before any transformations
            if df.index.duplicated().any():
                logger.warning(f"Found {df.index.duplicated().sum()} duplicate date indices in the original data. Taking the last occurrence.")
                df = df[~df.index.duplicated(keep='last')]
                
            logger.info(f"Successfully converted data to DataFrame with shape {df.shape}")
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            raise
        
        # Apply transformations
        statistics = {}
        logger.info(f"Applying transformations: {transform_params}")
        
        # Fill missing dates if requested
        if transform_params.fill_missing_dates:
            df = fill_missing_dates(df)
            statistics["missing_dates_filled"] = len(df) - len(data)
            logger.info(f"Filled missing dates, new shape: {df.shape}")
        
        # Resample if requested
        if transform_params.resample_freq:
            try:
                logger.info(f"Resampling data to frequency: {transform_params.resample_freq.value}")
                original_len = len(df)
                df = resample_ohlcv(df, transform_params.resample_freq.value)
                logger.info(f"Successfully resampled data from {original_len} to {len(df)} records")
                statistics["resampling"] = {
                    "original_count": original_len,
                    "resampled_count": len(df),
                    "frequency": transform_params.resample_freq.value
                }
            except Exception as e:
                logger.error(f"Error resampling data: {str(e)}")
                raise
        
        # Normalize if requested
        if transform_params.normalize_method:
            try:
                logger.info(f"Normalizing data using method: {transform_params.normalize_method.value}")
                df = normalize_data(df, transform_params.normalize_method.value)
                logger.info(f"Successfully normalized data")
                statistics["normalization"] = {
                    "method": transform_params.normalize_method.value
                }
            except Exception as e:
                logger.error(f"Error normalizing data: {str(e)}")
                raise
        
        # Calculate returns if requested
        if transform_params.calculate_returns:
            try:
                logger.info(f"Calculating returns with period: {transform_params.calculate_returns}")
                df = calculate_returns(df, transform_params.calculate_returns)
                logger.info(f"Successfully calculated returns")
                statistics["returns"] = {
                    "period": transform_params.calculate_returns
                }
            except Exception as e:
                logger.error(f"Error calculating returns: {str(e)}")
                raise
        
        # Convert back to StockDataResponse objects
        try:
            logger.info(f"Converting DataFrame with shape {df.shape} back to StockDataResponse objects")
            transformed_data = []
            for index, row in df.iterrows():
                # Create a StockDataResponse object directly
                from app.schemas import StockDataResponse, DataSourceEnum
                import pandas as pd
                
                # Convert source string to DataSourceEnum
                source_str = row.get('source', 'transformed')
                try:
                    source = DataSourceEnum(source_str)
                except ValueError:
                    source = DataSourceEnum.TRANSFORMED
                
                # Handle NaN values for ticker
                ticker_value = row.get('ticker', '')
                if pd.isna(ticker_value):
                    ticker_value = 'unknown'
                
                stock_data = StockDataResponse(
                    id=0,  # Placeholder ID
                    task_id=task_id,
                    ticker=ticker_value,
                    date=index,
                    open=row.get('open'),
                    high=row.get('high'),
                    low=row.get('low'),
                    close=row.get('close'),
                    volume=row.get('volume'),
                    source=source
                )
                transformed_data.append(stock_data)
            logger.info(f"Successfully converted {len(transformed_data)} records back to StockDataResponse objects")
        except Exception as e:
            logger.error(f"Error converting DataFrame back to StockData objects: {str(e)}")
            raise
        
        # Add basic statistics
        if len(df) > 0:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    statistics[f"{col}_stats"] = {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                    }
        
        return StockDataResponseWithMetadata(
            data=transformed_data,
            metadata={
                "total_count": total_count,
                "page_count": len(transformed_data),
                "has_more": total_count > (pagination.skip + len(data)),
                "statistics": statistics
            }
        )
    
    @staticmethod
    async def get_aggregated_task_data(
        task_id: int,
        filter_params: StockDataFilter,
        aggregation_params: StockDataAggregation,
        db: AsyncSession
    ) -> List[AggregatedStockData]:
        """Get aggregated data for a task.
        
        Args:
            task_id: ID of the task
            filter_params: Filter parameters
            aggregation_params: Aggregation parameters
            db: Database session
            
        Returns:
            List of aggregated stock data
        """
        logger.debug(f"Starting get_aggregated_task_data for task_id={task_id} with frequency={aggregation_params.group_by.value}")
        
        # Get all data for aggregation (no pagination for aggregation)
        pagination = PaginationParams(skip=0, limit=10000, sort_by="date", sort_desc=False)
        try:
            data, _ = await AsyncTaskService.get_filtered_task_data(
                task_id, filter_params, pagination, db
            )
            logger.debug(f"Retrieved {len(data)} data points for aggregation")
        except Exception as e:
            logger.error(f"Error retrieving filtered task data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        
        if not data:
            logger.warning(f"No data found for task {task_id} with the given filters")
            return []
        
        # Convert to DataFrame for aggregation
        try:
            df = convert_to_dataframe([{
                "ticker": item.ticker,
                "date": item.date,
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume,
                "source": item.source
            } for item in data])
            logger.debug(f"Successfully converted data to DataFrame with shape {df.shape}")
            
            # Log the first few rows to help with debugging
            if not df.empty:
                logger.debug(f"First row of DataFrame: {df.iloc[0].to_dict()}")
                logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                logger.debug(f"DataFrame index type: {type(df.index)}")
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        
        # Check for and handle duplicate date indices before any transformations
        if df.index.duplicated().any():
            logger.warning(f"Found {df.index.duplicated().sum()} duplicate date indices in the aggregation data. Taking the last occurrence.")
            df = df[~df.index.duplicated(keep='last')]
        
        # Group by ticker to handle multiple tickers
        try:
            # Ensure ticker column exists and has no NaN values
            if 'ticker' not in df.columns:
                logger.error("'ticker' column not found in DataFrame")
                return []
            
            # Fill NaN values in ticker column with 'unknown'
            if df['ticker'].isna().any():
                logger.warning(f"Found {df['ticker'].isna().sum()} NaN values in ticker column. Filling with 'unknown'")
                df['ticker'] = df['ticker'].fillna('unknown')
            
            grouped = df.groupby('ticker')
            logger.debug(f"Successfully grouped data by ticker. Found {len(grouped)} unique tickers")
        except Exception as e:
            logger.error(f"Error grouping data by ticker: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        aggregated_data = []
        for ticker, group in grouped:
            logger.info(f"Processing ticker {ticker} with {len(group)} data points")
            
            # Resample to the requested frequency
            try:
                # Log the group data before resampling
                logger.debug(f"Group data before resampling - shape: {group.shape}, columns: {group.columns.tolist()}")
                if not group.empty:
                    logger.debug(f"First row before resampling: {group.iloc[0].to_dict()}")
                    logger.debug(f"Index type: {type(group.index)}, first index: {group.index[0]}")
                
                # Ensure source column has no NaN values
                if 'source' in group.columns and group['source'].isna().any():
                    logger.warning(f"Found {group['source'].isna().sum()} NaN values in source column. Filling with 'transformed'")
                    group['source'] = group['source'].fillna('transformed')
                
                # Log the frequency being used
                logger.debug(f"Resampling with frequency: {aggregation_params.group_by.value}")
                
                # Ensure the index is sorted before resampling
                if not group.index.is_monotonic_increasing:
                    logger.warning(f"Index for ticker {ticker} is not sorted. Sorting before resampling.")
                    group = group.sort_index()
                
                resampled = resample_ohlcv(group, aggregation_params.group_by.value)
                
                # Log the resampled data
                logger.info(f"Successfully resampled data for ticker {ticker} to {len(resampled)} data points")
                logger.debug(f"Resampled data columns: {resampled.columns.tolist()}")
                if not resampled.empty:
                    logger.debug(f"First row after resampling: {resampled.iloc[0].to_dict()}")
                    logger.debug(f"Resampled index type: {type(resampled.index)}, first index: {resampled.index[0]}")
            except Exception as e:
                logger.error(f"Error resampling data for ticker {ticker}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
            
            # Calculate returns if requested
            if aggregation_params.include_returns:
                try:
                    resampled = calculate_returns(resampled)
                except Exception as e:
                    logger.error(f"Error calculating returns: {str(e)}")
                    # Continue without returns
        
            # Convert to AggregatedStockData objects
            for index, row in resampled.iterrows():
                try:
                    # Log the index we're working with
                    logger.debug(f"Processing index: {index}, type: {type(index)}")
                    
                    # Determine period start and end based on frequency
                    period_start = None
                    period_end = None
                    
                    if aggregation_params.group_by == ResampleFrequency.DAILY:
                        period_start = index
                        period_end = index
                        logger.debug(f"Daily period: start={period_start}, end={period_end}")
                    elif aggregation_params.group_by == ResampleFrequency.WEEKLY:
                        period_start = index - timedelta(days=6)
                        period_end = index
                        logger.debug(f"Weekly period: start={period_start}, end={period_end}")
                    elif aggregation_params.group_by == ResampleFrequency.MONTHLY:
                        # Handle ME (month-end) frequency specifically
                        logger.debug(f"Processing monthly frequency for index: {index}")
                        
                        # For ME frequency, index should already be the last day of the month
                        # First, ensure we're working with a pandas Timestamp
                        if not isinstance(index, pd.Timestamp):
                            try:
                                logger.debug(f"Converting index {index} to Timestamp")
                                index = pd.Timestamp(index)
                                logger.debug(f"Converted to Timestamp: {index}")
                            except Exception as e:
                                logger.error(f"Error converting index to Timestamp: {e}")
                                # Use the raw index as a fallback
                                period_start = index
                                period_end = index
                                logger.debug(f"Using fallback for monthly period: start={period_start}, end={period_end}")
                                continue
                        
                        try:
                            # Calculate the first day of the month
                            period_start = index.replace(day=1)
                            
                            # Use pandas to calculate the last day of the month safely
                            # This avoids issues with different month lengths
                            period_end = pd.Timestamp(index.year, index.month, 1) + pd.offsets.MonthEnd(1)
                            
                            logger.debug(f"Monthly period calculation: index={index}, start={period_start}, end={period_end}")
                        except Exception as e:
                            logger.error(f"Error calculating monthly period: {e}")
                            # Fallback to using the index for both start and end
                            period_start = index
                            period_end = index
                            logger.debug(f"Using fallback for monthly period after calculation error: start={period_start}, end={period_end}")
                    elif aggregation_params.group_by == ResampleFrequency.QUARTERLY:
                        # Handle quarterly frequency
                        if not isinstance(index, pd.Timestamp):
                            try:
                                index = pd.Timestamp(index)
                            except Exception as e:
                                logger.error(f"Error converting index to Timestamp for quarterly: {e}")
                                period_start = index
                                period_end = index
                                continue
                        
                        # Calculate the quarter
                        quarter = (index.month - 1) // 3
                        period_start = datetime(index.year, quarter * 3 + 1, 1)
                        if quarter == 3:  # Q4
                            period_end = datetime(index.year + 1, 1, 1) - timedelta(days=1)
                        else:
                            period_end = datetime(index.year, (quarter + 1) * 3 + 1, 1) - timedelta(days=1)
                        
                        logger.debug(f"Quarterly period calculation: index={index}, quarter={quarter+1}, start={period_start}, end={period_end}")
                    elif aggregation_params.group_by == ResampleFrequency.YEARLY:
                        period_start = datetime(index.year, 1, 1)
                        period_end = datetime(index.year, 12, 31)
                        logger.debug(f"Yearly period: start={period_start}, end={period_end}")
                    else:
                        # Default case for unknown frequency
                        logger.warning(f"Unknown frequency: {aggregation_params.group_by}. Using index as both start and end.")
                        period_start = index
                        period_end = index
                    
                    if period_start is None or period_end is None:
                        logger.error(f"Failed to determine period for index {index} with frequency {aggregation_params.group_by}")
                        continue
                        
                    # Ensure both period_start and period_end are datetime objects
                    if isinstance(period_start, pd.Timestamp):
                        period_start = period_start.to_pydatetime()
                    if isinstance(period_end, pd.Timestamp):
                        period_end = period_end.to_pydatetime()
                    
                    logger.debug(f"Determined period for index {index}: start={period_start}, end={period_end}")
                except Exception as e:
                    logger.error(f"Error determining period for index {index}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Try to recover with a fallback approach
                    try:
                        # Use the index as both start and end as a fallback
                        if isinstance(index, pd.Timestamp):
                            period_start = index.to_pydatetime()
                            period_end = index.to_pydatetime()
                        else:
                            period_start = index
                            period_end = index
                        logger.warning(f"Using fallback period calculation: start={period_start}, end={period_end}")
                    except Exception as fallback_error:
                        logger.error(f"Failed even with fallback period calculation: {fallback_error}")
                        continue
                
                # Create aggregated data object
                try:
                    # Get source from row or use transformed as default
                    source_value = row.get('source', 'transformed')
                    if pd.isna(source_value):
                        source_value = 'transformed'
                    
                    # Handle NaN values for ticker
                    ticker_str = row.get('ticker', 'unknown')
                    if pd.isna(ticker_str):
                        ticker_str = 'unknown'
                    
                    # Ensure data_points is an integer
                    data_points = row.get('data_points', 0)
                    if pd.isna(data_points):
                        data_points = 0
                    else:
                        data_points = int(data_points)
                    
                    # Log the values we're using
                    logger.debug(f"Creating AggregatedStockData with: ticker={ticker_str}, period_start={period_start}, "
                                f"period_end={period_end}, data_points={data_points}, source={source_value}")
                    
                    # Log the row data for debugging
                    logger.debug(f"Row data: {row.to_dict()}")
                    
                    aggregated_item = AggregatedStockData(
                        ticker=ticker_str,
                        period_start=period_start,
                        period_end=period_end,
                        data_points=data_points,
                        source=str(source_value)
                    )
                except Exception as e:
                    logger.error(f"Error creating AggregatedStockData: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
                
                # Add OHLC data if requested
                if aggregation_params.include_ohlc:
                    open_val = row.get('open')
                    high_val = row.get('high')
                    low_val = row.get('low')
                    close_val = row.get('close')
                    
                    # Handle NaN values for OHLC
                    aggregated_item.open = None if pd.isna(open_val) else float(open_val)
                    aggregated_item.high = None if pd.isna(high_val) else float(high_val)
                    aggregated_item.low = None if pd.isna(low_val) else float(low_val)
                    aggregated_item.close = None if pd.isna(close_val) else float(close_val)
                
                # Add volume if requested
                if aggregation_params.include_volume:
                    volume_val = row.get('volume')
                    aggregated_item.volume = None if pd.isna(volume_val) else float(volume_val)
                
                # Add returns if requested
                if aggregation_params.include_returns:
                    daily_return_val = row.get('daily_return')
                    weekly_return_val = row.get('weekly_return')
                    monthly_return_val = row.get('monthly_return')
                    
                    # Handle NaN values for returns
                    aggregated_item.daily_return = None if pd.isna(daily_return_val) else float(daily_return_val)
                    aggregated_item.weekly_return = None if pd.isna(weekly_return_val) else float(weekly_return_val)
                    aggregated_item.monthly_return = None if pd.isna(monthly_return_val) else float(monthly_return_val)
                
                # Add to result list
                aggregated_data.append(aggregated_item)
                logger.debug(f"Added aggregated item for {ticker_str} at {index}")
        
        logger.info(f"Returning {len(aggregated_data)} aggregated data points")
        return aggregated_data
