from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.utils.db_client import get_db
from app.schemas import StockDataResponse

router = APIRouter(
    prefix="/data",
    tags=["data"],
)

@router.get("/task/{task_id}", response_model=List[StockDataResponse])
async def get_task_data(
    task_id: int,
    ticker: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get processed data for a specific task."""
    # Implementation will come later
    return [{"message": f"Get data for task {task_id} endpoint (to be implemented)"}]