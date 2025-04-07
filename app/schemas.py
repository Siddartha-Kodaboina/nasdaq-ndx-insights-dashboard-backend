from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
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

# Task schemas
class TaskBase(BaseModel):
    task_type: TaskTypeEnum
    parameters: Dict[str, Any]

class TaskCreate(TaskBase):
    pass

class TaskResponse(TaskBase):
    id: int
    status: TaskStatusEnum
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Stock data schemas
class StockDataBase(BaseModel):
    ticker: str
    date: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    source: DataSourceEnum

class StockDataCreate(StockDataBase):
    task_id: int

class StockDataResponse(StockDataBase):
    id: int
    task_id: int

    class Config:
        orm_mode = True