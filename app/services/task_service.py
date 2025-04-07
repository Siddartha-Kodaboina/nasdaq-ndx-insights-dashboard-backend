from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from app.models import Task, TaskStatus, TaskType, StockData
from app.schemas import TaskCreate, TaskResponse
from app.queue import job_queue
from app.utils.db_client import get_db

logger = logging.getLogger(__name__)

class TaskService:
    """Service for managing tasks."""
    
    @staticmethod
    def create_task(task_data: Dict[str, Any], db: Session) -> Task:
        """Create a new task and enqueue it for processing.
        
        Args:
            task_data: Task data
            db: Database session
            
        Returns:
            Created task
        """
        # Create a new task in the database
        task = Task(
            task_type=task_data["task_type"],
            status=TaskStatus.PENDING,
            parameters=task_data["parameters"]
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        
        logger.info(f"Created task {task.id} of type {task.task_type}")
        
        # Enqueue the task for processing
        from app.workers import process_task
        job_queue.enqueue(task.id, process_task, task.id)
        
        return task
    
    @staticmethod
    def get_task(task_id: int, db: Session) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: ID of the task to get
            db: Database session
            
        Returns:
            Task if found, None otherwise
        """
        return db.query(Task).filter(Task.id == task_id).first()
    
    @staticmethod
    def list_tasks(status: Optional[str] = None, task_type: Optional[str] = None, db: Session = None) -> List[Task]:
        """List tasks with optional filtering.
        
        Args:
            status: Filter by status
            task_type: Filter by task type
            db: Database session
            
        Returns:
            List of tasks
        """
        query = db.query(Task)
        
        if status:
            query = query.filter(Task.status == status)
        
        if task_type:
            query = query.filter(Task.task_type == task_type)
        
        return query.order_by(Task.created_at.desc()).all()
    
    @staticmethod
    def get_task_data(task_id: int, ticker: Optional[str] = None, db: Session = None) -> List[StockData]:
        """Get data for a task.
        
        Args:
            task_id: ID of the task
            ticker: Filter by ticker
            db: Database session
            
        Returns:
            List of stock data records
        """
        query = db.query(StockData).filter(StockData.task_id == task_id)
        
        if ticker:
            query = query.filter(StockData.ticker == ticker)
        
        return query.order_by(StockData.date).all()