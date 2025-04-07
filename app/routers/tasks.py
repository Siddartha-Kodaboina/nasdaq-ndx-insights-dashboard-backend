from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from app.utils.db_client import get_db
from app.models import Task, TaskStatus, TaskType
from app.schemas import TaskCreate, TaskResponse
from app.utils.validators import validate_request_body
from app.services.task_service import TaskService

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
)

@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
@validate_request_body(TaskCreate)
async def create_task(task_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create a new data analysis task."""
    task = TaskService.create_task(task_data, db)
    return task

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: Session = Depends(get_db)):
    """Get a specific task by ID."""
    task = TaskService.get_task(task_id, db)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    return task

@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all tasks with optional filtering."""
    tasks = TaskService.list_tasks(status, task_type, db)
    return tasks