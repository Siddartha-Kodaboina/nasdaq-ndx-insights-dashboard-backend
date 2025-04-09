from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum

class TaskStatusEnum(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskTypeEnum(str, Enum):
    EXPLORE_STOCK = "explore_stock"
    COMPARE_STOCKS = "compare_stocks"
    STOCK_VS_INDEX = "stock_vs_index"

class DataSourceEnum(str, Enum):
    SOURCE_A = "source_a"
    SOURCE_B = "source_b"
    TRANSFORMED = "transformed"

# Specific parameter schemas for different task types
class DateRangeParams(BaseModel):
    """Base parameters for date range selection."""
    from_date: datetime = Field(..., description="Start date for data retrieval (inclusive)")
    to_date: datetime = Field(..., description="End date for data retrieval (inclusive)")
    
    @validator('to_date')
    def validate_date_range(cls, to_date, values):
        """Validate that to_date is after from_date."""
        if 'from_date' in values and to_date < values['from_date']:
            raise ValueError("End date must be after start date")
        return to_date
    
    @validator('from_date', 'to_date')
    def validate_dates_not_future(cls, value):
        """Validate that dates are not in the future."""
        if value > datetime.now():
            raise ValueError("Date cannot be in the future")
        return value

class ExploreStockParams(DateRangeParams):
    """Parameters for exploring a single stock."""
    ticker: str = Field(..., description="Stock ticker symbol")
    source: DataSourceEnum = Field(..., description="Data source to use")

class CompareStocksParams(DateRangeParams):
    """Parameters for comparing two stocks."""
    ticker1: str = Field(..., description="First stock ticker symbol")
    ticker2: str = Field(..., description="Second stock ticker symbol")
    field: str = Field("close", description="Field to compare (open, high, low, close)")
    
    @validator('field')
    def validate_field(cls, value):
        """Validate that field is a valid OHLC field."""
        valid_fields = ['open', 'high', 'low', 'close']
        if value.lower() not in valid_fields:
            raise ValueError(f"Field must be one of: {', '.join(valid_fields)}")
        return value.lower()

class StockVsIndexParams(DateRangeParams):
    """Parameters for comparing a stock against an index."""
    ticker: str = Field(..., description="Stock ticker symbol")
    index: str = Field("^NDX", description="Index to compare against")

# Task schemas
class TaskBase(BaseModel):
    """Base schema for tasks."""
    task_type: TaskTypeEnum = Field(..., description="Type of task to create")

class TaskCreate(TaskBase):
    """Schema for creating a new task."""
    parameters: Union[Dict[str, Any], ExploreStockParams, CompareStocksParams, StockVsIndexParams] = Field(
        ..., description="Parameters for the task"
    )
    
    @root_validator(skip_on_failure=True)
    def validate_parameters(cls, values):
        """Validate that parameters match the task type."""
        task_type = values.get('task_type')
        parameters = values.get('parameters')
        
        if not task_type or not parameters:
            return values
        
        # Convert dict parameters to appropriate model if needed
        if isinstance(parameters, dict):
            if task_type == TaskTypeEnum.EXPLORE_STOCK:
                values['parameters'] = ExploreStockParams(**parameters)
            elif task_type == TaskTypeEnum.COMPARE_STOCKS:
                values['parameters'] = CompareStocksParams(**parameters)
            elif task_type == TaskTypeEnum.STOCK_VS_INDEX:
                values['parameters'] = StockVsIndexParams(**parameters)
        
        # Validate parameter type matches task type
        if task_type == TaskTypeEnum.EXPLORE_STOCK and not isinstance(values['parameters'], ExploreStockParams):
            raise ValueError("Parameters must be ExploreStockParams for EXPLORE_STOCK task")
        elif task_type == TaskTypeEnum.COMPARE_STOCKS and not isinstance(values['parameters'], CompareStocksParams):
            raise ValueError("Parameters must be CompareStocksParams for COMPARE_STOCKS task")
        elif task_type == TaskTypeEnum.STOCK_VS_INDEX and not isinstance(values['parameters'], StockVsIndexParams):
            raise ValueError("Parameters must be StockVsIndexParams for STOCK_VS_INDEX task")
        
        return values

class TaskResponse(BaseModel):
    """Response model for tasks."""
    id: int
    task_type: str
    status: str
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    progress: Optional[float] = None
    error_message: Optional[str] = None
    estimated_completion_time: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class TaskStatusResponse(BaseModel):
    """Schema for detailed task status response."""
    task_id: int
    status: TaskStatusEnum
    progress: float = 0.0
    elapsed_time: Optional[float] = None  # in seconds
    estimated_time_remaining: Optional[float] = None  # in seconds
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

# Data transformation and aggregation enums
class ResampleFrequency(str, Enum):
    """Enum for resampling frequencies."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "ME"
    QUARTERLY = "QE"
    YEARLY = "YE"

class NormalizationMethod(str, Enum):
    """Enum for normalization methods."""
    MIN_MAX = "min-max"
    Z_SCORE = "z-score"
    PERCENT_CHANGE = "percent_change"
    FIRST_VALUE = "first_value"

# Stock data schemas
class StockDataBase(BaseModel):
    """Base schema for stock data."""
    ticker: str
    date: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    source: DataSourceEnum

class StockDataCreate(StockDataBase):
    """Schema for creating stock data."""
    task_id: int

class StockDataResponse(StockDataBase):
    """Schema for stock data response."""
    id: int
    task_id: int

    class Config:
        orm_mode = True

class StockDataFilter(BaseModel):
    """Schema for filtering stock data."""
    ticker: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    price_field: Optional[str] = Field("close", description="Price field to filter on (open, high, low, close)")
    
    @validator('price_field')
    def validate_price_field(cls, v):
        valid_fields = ['open', 'high', 'low', 'close']
        if v not in valid_fields:
            raise ValueError(f"Price field must be one of: {', '.join(valid_fields)}")
        return v

class StockDataTransform(BaseModel):
    """Schema for transforming stock data."""
    resample_freq: Optional[ResampleFrequency] = None
    normalize_method: Optional[NormalizationMethod] = None
    calculate_returns: Optional[int] = Field(None, description="Period for return calculation")
    fill_missing_dates: Optional[bool] = Field(False, description="Fill missing dates with NaN values")

class StockDataAggregation(BaseModel):
    """Schema for aggregating stock data."""
    group_by: ResampleFrequency
    include_ohlc: bool = Field(True, description="Include OHLC data in aggregation")
    include_volume: bool = Field(True, description="Include volume data in aggregation")
    include_returns: bool = Field(False, description="Include returns data in aggregation")

class AggregatedStockData(BaseModel):
    """Schema for aggregated stock data response."""
    ticker: str
    period_start: datetime
    period_end: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    returns: Optional[float] = None
    daily_return: Optional[float] = None
    weekly_return: Optional[float] = None
    monthly_return: Optional[float] = None
    data_points: int
    source: str

class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    skip: int = Field(0, description="Number of records to skip")
    limit: int = Field(100, description="Maximum number of records to return")
    sort_by: str = Field("date", description="Field to sort by")
    sort_desc: bool = Field(True, description="Sort in descending order if True")
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        valid_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        if v not in valid_fields:
            raise ValueError(f"Sort field must be one of: {', '.join(valid_fields)}")
        return v

class StockDataResponseWithMetadata(BaseModel):
    """Schema for stock data response with metadata."""
    data: List[StockDataResponse]
    metadata: Dict[str, Any] = Field(
        {
            "total_count": 0,
            "page_count": 0,
            "has_more": False,
            "statistics": {}
        },
        description="Metadata about the response"
    )