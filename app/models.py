from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
import datetime
from app.utils.db_client import Base

class TaskStatus(enum.Enum):
    """Enum for task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(enum.Enum):
    """Enum for task types."""
    EXPLORE_STOCK = "explore_stock"
    COMPARE_STOCKS = "compare_stocks"
    STOCK_VS_INDEX = "stock_vs_index"

class DataSource(enum.Enum):
    """Enum for data sources."""
    SOURCE_A = "source_a"
    SOURCE_B = "source_b"

class Task(Base):
    """Model for tasks.
    
    Each task represents a data analysis job created by a user.
    """
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True)
    task_type = Column(Enum(TaskType), nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    parameters = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationship with StockData
    stock_data = relationship("StockData", back_populates="task")
    
    def __repr__(self):
        return f"<Task(id={self.id}, type={self.task_type}, status={self.status})>"

class StockData(Base):
    """Model for stock data.
    
    Each record represents a single data point for a stock at a specific date.
    """
    __tablename__ = "stock_data"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    ticker = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    source = Column(Enum(DataSource), nullable=False)
    
    # Relationship with Task
    task = relationship("Task", back_populates="stock_data")
    
    def __repr__(self):
        return f"<StockData(id={self.id}, ticker={self.ticker}, date={self.date})>"