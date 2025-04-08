import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Generator, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from app.models import Task, TaskStatus, TaskType, StockData, DataSource as DataSourceEnum
from app.utils.db_client import get_db
from app.utils.data_sources import get_data_source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ProcessingState(Enum):
    """Enum for task processing states."""
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    PROCESSING_DATA = "processing_data"
    SAVING_RESULTS = "saving_results"
    COMPLETED = "completed"
    FAILED = "failed"

def process_task(task_id: int) -> Dict[str, Any]:
    """Process a task.
    
    This function is called by the job queue to process a task.
    It creates a TaskProcessor instance and runs the processing pipeline.
    
    Args:
        task_id: ID of the task to process
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting task processor for task {task_id}")
    processor = TaskProcessor(task_id)
    result = processor.process()
    logger.info(f"Task {task_id} processing completed with status: {result['status']}")
    return result


class TaskProcessor:
    """Processor for data analysis tasks.
    
    This class implements a generator-based processing pipeline for tasks.
    """
    
    def __init__(self, task_id: int):
        """Initialize a task processor.
        
        Args:
            task_id: ID of the task to process
        """
        self.task_id = task_id
        self.task = None
        self.state = ProcessingState.INITIALIZING
        self.progress = 0
        self.result = None
        self.error = None
        self.start_time = datetime.now()
        self.end_time = None
        self.processing_steps = []
        logger.info(f"Initialized TaskProcessor for task {task_id}")
    
    def _update_state(self, state: ProcessingState, progress: int = None, message: str = None):
        """Update the processing state.
        
        Args:
            state: New state
            progress: Progress percentage (0-100)
            message: Optional message to log
        """
        self.state = state
        if progress is not None:
            self.progress = progress
        
        step_info = {
            "state": state.value,
            "progress": self.progress,
            "timestamp": datetime.now().isoformat(),
        }
        
        if message:
            step_info["message"] = message
            logger.info(f"Task {self.task_id}: {message} (State: {state.value}, Progress: {self.progress}%)")
        else:
            logger.info(f"Task {self.task_id}: State changed to {state.value} (Progress: {self.progress}%)")
        
        self.processing_steps.append(step_info)
    
    def _processing_pipeline(self) -> Generator[None, None, None]:
        """Generator-based processing pipeline.
        
        Yields at each step to allow for progress tracking.
        """
        # Initialize the task
        self._update_state(ProcessingState.INITIALIZING, progress=0, message="Initializing task")
        yield
        
        # Load the task from the database
        with get_db() as db:
            self.task = db.query(Task).filter(Task.id == self.task_id).first()
            if not self.task:
                raise ValueError(f"Task {self.task_id} not found")
        
        self._update_state(ProcessingState.INITIALIZING, progress=10, 
                           message=f"Task loaded: {self.task.task_type}")
        yield
        
        # Process the task based on its type
        if self.task.task_type == TaskType.EXPLORE_STOCK:
            yield from self._process_explore_stock()
        elif self.task.task_type == TaskType.COMPARE_STOCKS:
            yield from self._process_compare_stocks()
        elif self.task.task_type == TaskType.STOCK_VS_INDEX:
            yield from self._process_stock_vs_index()
        else:
            raise ValueError(f"Unknown task type: {self.task.task_type}")
        
        # Mark the task as completed
        self._update_state(ProcessingState.COMPLETED, progress=100, 
                           message="Task completed successfully")
        self.end_time = datetime.now()
        yield
    
    def _process_explore_stock(self) -> Generator[None, None, None]:
        """Process an explore stock task.
        
        Yields at each step to allow for progress tracking.
        """
        # Extract parameters
        params = self.task.parameters
        ticker = params.get("ticker")
        source = params.get("source")
        
        # Handle source as either enum object or string value
        if isinstance(source, str):
            # If it's a string, find the matching enum
            source = DataSourceEnum(source)
            
        from_date = datetime.fromisoformat(params.get("from_date"))
        to_date = datetime.fromisoformat(params.get("to_date"))
        
        self._update_state(ProcessingState.LOADING_DATA, progress=20, 
                           message=f"Loading data for {ticker} from {source}")
        yield
        
        # Load data from the data source
        try:
            with get_data_source(source) as data_source:
                # Check if ticker exists
                tickers = data_source.get_tickers()
                if ticker not in tickers:
                    raise ValueError(f"Ticker {ticker} not found in {source}")
                
                # Get data for the ticker
                data = data_source.get_data(ticker, from_date, to_date)
                if data.empty:
                    raise ValueError(f"No data found for {ticker} in the specified date range")
                
                self._update_state(ProcessingState.LOADING_DATA, progress=40, 
                                   message=f"Loaded {len(data)} rows of data")
                yield
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
        # Process the data
        self._update_state(ProcessingState.PROCESSING_DATA, progress=50, 
                           message="Calculating statistics")
        yield
        
        # Calculate basic statistics
        stats = {
            "ticker": ticker,
            "source": source,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "data_points": len(data),
            "statistics": {
                "open": {
                    "min": float(data["open"].min()),
                    "max": float(data["open"].max()),
                    "mean": float(data["open"].mean()),
                    "std": float(data["open"].std())
                },
                "high": {
                    "min": float(data["high"].min()),
                    "max": float(data["high"].max()),
                    "mean": float(data["high"].mean()),
                    "std": float(data["high"].std())
                },
                "low": {
                    "min": float(data["low"].min()),
                    "max": float(data["low"].max()),
                    "mean": float(data["low"].mean()),
                    "std": float(data["low"].std())
                },
                "close": {
                    "min": float(data["close"].min()),
                    "max": float(data["close"].max()),
                    "mean": float(data["close"].mean()),
                    "std": float(data["close"].std())
                },
                "volume": {
                    "min": float(data["volume"].min()),
                    "max": float(data["volume"].max()),
                    "mean": float(data["volume"].mean()),
                    "std": float(data["volume"].std())
                }
            }
        }
        
        # Calculate daily returns
        data["daily_return"] = data["close"].pct_change()
        data = data.dropna()
        
        stats["statistics"]["daily_return"] = {
            "min": float(data["daily_return"].min()),
            "max": float(data["daily_return"].max()),
            "mean": float(data["daily_return"].mean()),
            "std": float(data["daily_return"].std())
        }
        
        self._update_state(ProcessingState.PROCESSING_DATA, progress=70, 
                           message="Statistics calculated")
        yield
        
        # Save the results to the database
        self._update_state(ProcessingState.SAVING_RESULTS, progress=80, 
                           message="Saving results to database")
        yield
        
        with get_db() as db:
            # Save the processed data points
            for _, row in data.iterrows():
                # Ensure date is a datetime object
                date_value = row.name
                if not isinstance(date_value, datetime):
                    date_value = pd.to_datetime(date_value)
                    
                stock_data = StockData(
                    task_id=self.task_id,
                    ticker=ticker,
                    date=date_value,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    source=source
                )
                db.add(stock_data)
            
            db.commit()
        
        self._update_state(ProcessingState.SAVING_RESULTS, progress=90, 
                           message=f"Saved {len(data)} data points to database")
        yield
        
        # Store the result
        self.result = stats
        yield
    
    def _process_compare_stocks(self) -> Generator[None, None, None]:
        """Process a compare stocks task.
        
        Yields at each step to allow for progress tracking.
        """
        # Extract parameters
        params = self.task.parameters
        ticker1 = params.get("ticker1")
        ticker2 = params.get("ticker2")
        field = params.get("field", "close")
        source = params.get("source")
        
        # Handle source as either enum object or string value
        if isinstance(source, str):
            # If it's a string, find the matching enum
            source = DataSourceEnum(source)
            
        from_date = datetime.fromisoformat(params.get("from_date"))
        to_date = datetime.fromisoformat(params.get("to_date"))
        
        self._update_state(ProcessingState.LOADING_DATA, progress=20, 
                           message=f"Loading data for {ticker1} and {ticker2} from {source}")
        yield
        
        # Load data from the data source
        try:
            with get_data_source(source) as data_source:
                # Check if tickers exist
                tickers = data_source.get_tickers()
                if ticker1 not in tickers:
                    raise ValueError(f"Ticker {ticker1} not found in {source}")
                if ticker2 not in tickers:
                    raise ValueError(f"Ticker {ticker2} not found in {source}")
                
                # Get data for the tickers
                data1 = data_source.get_data(ticker1, from_date, to_date)
                data2 = data_source.get_data(ticker2, from_date, to_date)
                
                if data1.empty:
                    raise ValueError(f"No data found for {ticker1} in the specified date range")
                if data2.empty:
                    raise ValueError(f"No data found for {ticker2} in the specified date range")
                
                self._update_state(ProcessingState.LOADING_DATA, progress=40, 
                                   message=f"Loaded {len(data1)} rows for {ticker1} and {len(data2)} rows for {ticker2}")
                yield
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
        # Process the data
        self._update_state(ProcessingState.PROCESSING_DATA, progress=50, 
                           message="Comparing stocks")
        yield
        
        # Prepare the comparison data
        comparison = {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "field": field,
            "source": source,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "ticker1_data_points": len(data1),
            "ticker2_data_points": len(data2),
            "comparison": {
                "ticker1_mean": float(data1[field].mean()),
                "ticker2_mean": float(data2[field].mean()),
                "ticker1_std": float(data1[field].std()),
                "ticker2_std": float(data2[field].std()),
                "ticker1_min": float(data1[field].min()),
                "ticker2_min": float(data2[field].min()),
                "ticker1_max": float(data1[field].max()),
                "ticker2_max": float(data2[field].max()),
            }
        }
        
        # Calculate correlation
        # Merge the data on date
        merged_data = pd.merge(
            data1[[field]].rename(columns={field: f"{ticker1}_{field}"}),
            data2[[field]].rename(columns={field: f"{ticker2}_{field}"}),
            left_index=True,
            right_index=True,
            how="inner"
        )
        
        if not merged_data.empty:
            correlation = merged_data[f"{ticker1}_{field}"].corr(merged_data[f"{ticker2}_{field}"])
            comparison["comparison"]["correlation"] = float(correlation)
        else:
            comparison["comparison"]["correlation"] = None
        
        self._update_state(ProcessingState.PROCESSING_DATA, progress=70, 
                           message="Comparison completed")
        yield
        
        # Save the results to the database
        self._update_state(ProcessingState.SAVING_RESULTS, progress=80, 
                           message="Saving results to database")
        yield
        
        with get_db() as db:
            # Save the processed data points for both tickers
            for ticker, data in [(ticker1, data1), (ticker2, data2)]:
                for _, row in data.iterrows():
                    # Ensure date is a datetime object
                    date_value = row.name
                    if not isinstance(date_value, datetime):
                        date_value = pd.to_datetime(date_value)
                        
                    stock_data = StockData(
                        task_id=self.task_id,
                        ticker=ticker,
                        date=date_value,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        source=source
                    )
                    db.add(stock_data)
            
            db.commit()
        
        self._update_state(ProcessingState.SAVING_RESULTS, progress=90, 
                           message=f"Saved {len(data1) + len(data2)} data points to database")
        yield
        
        # Store the result
        self.result = comparison
        yield
    
    def _process_stock_vs_index(self) -> Generator[None, None, None]:
        """Process a stock vs index task.
        
        Yields at each step to allow for progress tracking.
        """
        # Extract parameters
        params = self.task.parameters
        ticker = params.get("ticker")
        index_ticker = params.get("index_ticker")
        source = params.get("source")
        
        # Handle source as either enum object or string value
        if isinstance(source, str):
            # If it's a string, find the matching enum
            source = DataSourceEnum(source)
            
        from_date = datetime.fromisoformat(params.get("from_date"))
        to_date = datetime.fromisoformat(params.get("to_date"))
        
        self._update_state(ProcessingState.LOADING_DATA, progress=20, 
                           message=f"Loading data for {ticker} and {index_ticker} from {source}")
        yield
        
        # Load data from the data source
        try:
            with get_data_source(source) as data_source:
                # Check if tickers exist
                tickers = data_source.get_tickers()
                if ticker not in tickers:
                    raise ValueError(f"Ticker {ticker} not found in {source}")
                if index_ticker not in tickers:
                    raise ValueError(f"Index ticker {index_ticker} not found in {source}")
                
                # Get data for the tickers
                stock_data = data_source.get_data(ticker, from_date, to_date)
                index_data = data_source.get_data(index_ticker, from_date, to_date)
                
                if stock_data.empty:
                    raise ValueError(f"No data found for {ticker} in the specified date range")
                if index_data.empty:
                    raise ValueError(f"No data found for {index_ticker} in the specified date range")
                
                self._update_state(ProcessingState.LOADING_DATA, progress=40, 
                                   message=f"Loaded {len(stock_data)} rows for {ticker} and {len(index_data)} rows for {index_ticker}")
                yield
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
        # Process the data
        self._update_state(ProcessingState.PROCESSING_DATA, progress=50, 
                           message="Comparing stock to index")
        yield
        
        # Calculate daily returns
        stock_data["daily_return"] = stock_data["close"].pct_change()
        index_data["daily_return"] = index_data["close"].pct_change()
        
        # Drop NaN values
        stock_data = stock_data.dropna()
        index_data = index_data.dropna()
        
        # Merge the data on date
        merged_data = pd.merge(
            stock_data[["close", "daily_return"]].rename(columns={"close": f"{ticker}_close", "daily_return": f"{ticker}_return"}),
            index_data[["close", "daily_return"]].rename(columns={"close": f"{index_ticker}_close", "daily_return": f"{index_ticker}_return"}),
            left_index=True,
            right_index=True,
            how="inner"
        )
        
        # Calculate beta (stock's volatility relative to the market)
        if not merged_data.empty:
            covariance = merged_data[f"{ticker}_return"].cov(merged_data[f"{index_ticker}_return"])
            index_variance = merged_data[f"{index_ticker}_return"].var()
            beta = covariance / index_variance if index_variance != 0 else None
        else:
            beta = None
        
        # Prepare the comparison data
        comparison = {
            "ticker": ticker,
            "index_ticker": index_ticker,
            "source": source,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "stock_data_points": len(stock_data),
            "index_data_points": len(index_data),
            "comparison": {
                "stock_mean_return": float(stock_data["daily_return"].mean()),
                "index_mean_return": float(index_data["daily_return"].mean()),
                "stock_std_return": float(stock_data["daily_return"].std()),
                "index_std_return": float(index_data["daily_return"].std()),
                "beta": float(beta) if beta is not None else None,
            }
        }
        
        # Calculate correlation
        if not merged_data.empty:
            correlation = merged_data[f"{ticker}_return"].corr(merged_data[f"{index_ticker}_return"])
            comparison["comparison"]["correlation"] = float(correlation)
        else:
            comparison["comparison"]["correlation"] = None
        
        self._update_state(ProcessingState.PROCESSING_DATA, progress=70, 
                           message="Comparison completed")
        yield
        
        # Save the results to the database
        self._update_state(ProcessingState.SAVING_RESULTS, progress=80, 
                           message="Saving results to database")
        yield
        
        with get_db() as db:
            # Save the processed data points for both tickers
            for ticker_name, data in [(ticker, stock_data), (index_ticker, index_data)]:
                for _, row in data.iterrows():
                    # Ensure date is a datetime object
                    date_value = row.name
                    if not isinstance(date_value, datetime):
                        date_value = pd.to_datetime(date_value)
                        
                    db_stock_data = StockData(
                        task_id=self.task_id,
                        ticker=ticker_name,
                        date=date_value,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        source=source
                    )
                    db.add(db_stock_data)
            
            db.commit()
        
        self._update_state(ProcessingState.SAVING_RESULTS, progress=90, 
                           message=f"Saved {len(stock_data) + len(index_data)} data points to database")
        yield
        
        # Store the result
        self.result = comparison
        yield
    
    def _convert_enums_to_strings(self, obj):
        """Convert any enum values in the object to their string representation.
        
        Args:
            obj: The object to convert
            
        Returns:
            The converted object
        """
        if isinstance(obj, dict):
            return {k: self._convert_enums_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums_to_strings(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj
    
    def process(self) -> Dict[str, Any]:
        """Process the task.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Run the processing pipeline
            for _ in self._processing_pipeline():
                pass  # The pipeline yields at each step to allow for progress tracking
            
            # Convert any enum values to strings
            serializable_result = self._convert_enums_to_strings(self.result)
            
            # Return the final result
            return {
                "status": "success",
                "task_id": self.task_id,
                "result": serializable_result,
                "processing_time": (self.end_time - self.start_time).total_seconds(),
                "steps": self.processing_steps
            }
        
        except Exception as e:
            logger.exception(f"Error processing task {self.task_id}: {str(e)}")
            self._update_state(ProcessingState.FAILED, message=f"Error: {str(e)}")
            self.error = str(e)
            self.end_time = datetime.now()
            
            return {
                "status": "error",
                "task_id": self.task_id,
                "error": str(e),
                "processing_time": (self.end_time - self.start_time).total_seconds(),
                "steps": self.processing_steps
            }