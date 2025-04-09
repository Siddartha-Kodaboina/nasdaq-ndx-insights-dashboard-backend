"""
Async task processor for the stock analysis application.

This module provides asynchronous task processing utilities for data analysis tasks.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import pandas as pd
import numpy as np

from app.models import Task, StockData, TaskStatus, TaskType, DataSource as DataSourceEnum
from app.utils.async_db_client import AsyncDBSession
from app.utils.data_sources import get_data_source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class ProcessingState(str, Enum):
    """Enum for task processing states."""
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    PROCESSING_DATA = "processing_data"
    SAVING_RESULTS = "saving_results"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

async def process_task_async(task_id: int, timeout: int = 300) -> Dict[str, Any]:
    """Process a task asynchronously.
    
    This function is called by the job queue to process a task.
    It creates an AsyncTaskProcessor instance and runs the processing pipeline.
    
    Args:
        task_id: ID of the task to process
        timeout: Maximum time in seconds to allow the task to run (default: 5 minutes)
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting async task processor for task {task_id}")
    processor = AsyncTaskProcessor(task_id)
    
    # Create a task with timeout
    try:
        # Create a task for processing with timeout
        task = asyncio.create_task(processor.process())
        result = await asyncio.wait_for(task, timeout=timeout)
        logger.info(f"Task {task_id} processing completed with status: {result['status']}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Task {task_id} timed out after {timeout} seconds")
        await processor.handle_timeout()
        return {
            "status": "failed",
            "task_id": task_id,
            "error": f"Task processing timed out after {timeout} seconds",
            "processing_time": timeout,
            "steps": processor.processing_steps
        }
    except asyncio.CancelledError:
        logger.warning(f"Task {task_id} was cancelled")
        await processor.handle_cancellation()
        return {
            "status": "cancelled",
            "task_id": task_id,
            "error": "Task was cancelled",
            "processing_time": (datetime.now() - processor.start_time).total_seconds(),
            "steps": processor.processing_steps
        }
    except Exception as e:
        logger.exception(f"Error processing task {task_id}: {str(e)}")
        return {
            "status": "failed",
            "task_id": task_id,
            "error": str(e),
            "processing_time": (datetime.now() - processor.start_time).total_seconds() if processor.start_time else 0,
            "steps": processor.processing_steps
        }

class AsyncTaskProcessor:
    """Async processor for data analysis tasks.
    
    This class implements an async processing pipeline for tasks.
    """
    
    def __init__(self, task_id: int):
        """Initialize an async task processor.
        
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
        self._task = None  # The asyncio task object
        logger.info(f"Initialized AsyncTaskProcessor for task {task_id}")
    
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
    
    async def handle_timeout(self):
        """Handle task timeout.
        
        Updates the task state to indicate timeout and updates the database.
        """
        self.error = "Task processing timed out"
        self.end_time = datetime.now()
        self._update_state(ProcessingState.TIMEOUT, message="Task processing timed out")
        
        # Update the task status in the database
        async with AsyncDBSession() as db:
            # Get the task
            stmt = select(Task).where(Task.id == self.task_id)
            result = await db.execute(stmt)
            task = result.scalar_one_or_none()
            
            if task:
                task.status = TaskStatus.FAILED
                task.error = "Task processing timed out"
                await db.commit()
    
    async def handle_cancellation(self):
        """Handle task cancellation.
        
        Updates the task state to indicate cancellation and updates the database.
        """
        self.error = "Task was cancelled"
        self.end_time = datetime.now()
        self._update_state(ProcessingState.CANCELLED, message="Task was cancelled")
        
        # Update the task status in the database
        async with AsyncDBSession() as db:
            # Get the task
            stmt = select(Task).where(Task.id == self.task_id)
            result = await db.execute(stmt)
            task = result.scalar_one_or_none()
            
            if task:
                task.status = TaskStatus.CANCELLED
                task.error = "Task was cancelled"
                await db.commit()
    
    async def _processing_pipeline(self):
        """Async processing pipeline.
        
        Implements the main task processing logic.
        """
        self._update_state(ProcessingState.INITIALIZING, progress=0, 
                          message="Initializing task processor")
        
        # Load the task from the database
        async with AsyncDBSession() as db:
            # Get the task
            stmt = select(Task).where(Task.id == self.task_id)
            result = await db.execute(stmt)
            self.task = result.scalar_one_or_none()
            
            if not self.task:
                raise ValueError(f"Task {self.task_id} not found")
                
            # Update the task status to in progress
            self.task.status = TaskStatus.IN_PROGRESS
            self.task.progress = 0
            await db.commit()
        
        self._update_state(ProcessingState.INITIALIZING, progress=10, 
                          message=f"Task loaded: {self.task.task_type}")
        
        # Process the task based on its type
        if self.task.task_type == TaskType.EXPLORE_STOCK:
            await self._process_explore_stock()
        elif self.task.task_type == TaskType.COMPARE_STOCKS:
            await self._process_compare_stocks()
        elif self.task.task_type == TaskType.STOCK_VS_INDEX:
            await self._process_stock_vs_index()
        else:
            raise ValueError(f"Unknown task type: {self.task.task_type}")
        
        # Mark the task as completed
        try:
            self._update_state(ProcessingState.COMPLETED, progress=100, 
                            message="Task completed successfully")
            self.end_time = datetime.now()
            logger.info(f"Task {self.task_id} processing completed. Updating database status.")
            
            # Update the task status in the database
            async with AsyncDBSession() as db:
                # Get the task again to avoid stale data
                stmt = select(Task).where(Task.id == self.task_id)
                result = await db.execute(stmt)
                task = result.scalar_one_or_none()
                
                if task:
                    logger.info(f"Updating task {self.task_id} status from {task.status} to {TaskStatus.COMPLETED}")
                    task.status = TaskStatus.COMPLETED
                    task.progress = 100.0  # Ensure progress is set to 100%
                    task.end_time = self.end_time
                    await db.commit()
                    logger.info(f"Task {self.task_id} status updated successfully in database")
                else:
                    logger.error(f"Task {self.task_id} not found when trying to mark as completed")
        except Exception as e:
            logger.exception(f"Error marking task {self.task_id} as completed: {str(e)}")
    
    async def _process_explore_stock(self):
        """Process an explore stock task asynchronously.
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
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
        # Process the data
        self._update_state(ProcessingState.PROCESSING_DATA, progress=50, 
                          message="Calculating statistics")
        
        # Calculate statistics
        stats = {
            "ticker": ticker,
            "source": source,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "data_points": len(data),
            "statistics": {}
        }
        
        # Calculate statistics for each field
        for field in ["open", "high", "low", "close", "volume"]:
            stats["statistics"][field] = {
                "min": float(data[field].min()),
                "max": float(data[field].max()),
                "mean": float(data[field].mean()),
                "std": float(data[field].std())
            }
        
        # Calculate daily returns
        data["daily_return"] = data["close"].pct_change()
        
        # Add daily return statistics
        stats["statistics"]["daily_return"] = {
            "min": float(data["daily_return"].min()),
            "max": float(data["daily_return"].max()),
            "mean": float(data["daily_return"].mean()),
            "std": float(data["daily_return"].std())
        }
        
        self._update_state(ProcessingState.PROCESSING_DATA, progress=70, 
                          message="Statistics calculated")
        
        # Save the results to the database
        self._update_state(ProcessingState.SAVING_RESULTS, progress=80, 
                          message="Saving results to database")
        
        async with AsyncDBSession() as db:
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
            
            await db.commit()
        
        self._update_state(ProcessingState.SAVING_RESULTS, progress=90, 
                          message=f"Saved {len(data)} data points to database")
    
        # Store the result
        self.result = stats
        
        # Ensure we mark the task as completed
        logger.info(f"Task {self.task_id} processing completed successfully")
    
    async def _process_compare_stocks(self):
        """Process a compare stocks task asynchronously.
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
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
        # Process the data
        self._update_state(ProcessingState.PROCESSING_DATA, progress=50, 
                          message="Comparing stocks")
        
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
        
        # Save the results to the database
        self._update_state(ProcessingState.SAVING_RESULTS, progress=80, 
                          message="Saving results to database")
        
        async with AsyncDBSession() as db:
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
            
            await db.commit()
        
        self._update_state(ProcessingState.SAVING_RESULTS, progress=90, 
                          message=f"Saved {len(data1) + len(data2)} data points to database")
        
        # Store the result
        self.result = comparison
    
    async def _process_stock_vs_index(self):
        """Process a stock vs index task asynchronously.
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
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
        
        # Process the data
        self._update_state(ProcessingState.PROCESSING_DATA, progress=50, 
                          message="Comparing stock to index")
        
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
        
        # Save the results to the database
        self._update_state(ProcessingState.SAVING_RESULTS, progress=80, 
                          message="Saving results to database")
        
        async with AsyncDBSession() as db:
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
            
            await db.commit()
        
        self._update_state(ProcessingState.SAVING_RESULTS, progress=90, 
                          message=f"Saved {len(stock_data) + len(index_data)} data points to database")
        
        # Store the result
        self.result = comparison
    
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
    
    async def process(self):
        """Process the task asynchronously.
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Run the processing pipeline
            await self._processing_pipeline()
        
            # Explicitly ensure the task is marked as completed in the database
            async with AsyncDBSession() as db:
                # Get the task
                stmt = select(Task).where(Task.id == self.task_id)
                result = await db.execute(stmt)
                task = result.scalar_one_or_none()
                
                if task:
                    logger.info(f"Final update: Setting task {self.task_id} status to COMPLETED")
                    task.status = TaskStatus.COMPLETED
                    task.progress = 100.0
                    task.end_time = self.end_time
                    await db.commit()
                    logger.info(f"Task {self.task_id} marked as COMPLETED in the database")
        
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
            self.error = str(e)
            self.end_time = datetime.now()
            self._update_state(ProcessingState.FAILED, message=f"Error: {str(e)}")
            
            # Update the task status in the database
            async with AsyncDBSession() as db:
                # Get the task
                stmt = select(Task).where(Task.id == self.task_id)
                result = await db.execute(stmt)
                task = result.scalar_one_or_none()
                
                if task:
                    task.status = TaskStatus.FAILED
                    task.end_time = self.end_time
                    task.error = str(e)
                    await db.commit()
            
            return {
                "status": "failed",
                "task_id": self.task_id,
                "error": str(e),
                "processing_time": (self.end_time - self.start_time).total_seconds(),
                "steps": self.processing_steps
            }
