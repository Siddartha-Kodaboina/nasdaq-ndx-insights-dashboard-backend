from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.utils.db_client import get_db
from app.schemas import StockDataResponse
from app.models import DataSource as DataSourceEnum
from app.utils.validators import validate_ticker

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

@router.get("/stock/{ticker}")
@validate_ticker
async def get_stock_data(
    ticker: str,
    source: DataSourceEnum,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """Get stock data directly from a data source."""
    # Implementation will come later
    return {"message": f"Get data for {ticker} from {source.value} (to be implemented)"}