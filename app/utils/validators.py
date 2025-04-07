from functools import wraps
from typing import Callable, Type, Any, Dict, List
from fastapi import HTTPException, status
from pydantic import BaseModel, ValidationError
import logging

logger = logging.getLogger(__name__)

def validate_request_body(model: Type[BaseModel]):
    """Decorator to validate request body against a Pydantic model.
    
    Args:
        model: Pydantic model to validate against
        
    Returns:
        Decorated function
        
    Example:
        @validate_request_body(TaskCreate)
        def create_task(data: dict):
            # data is guaranteed to be valid
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the first dict argument
            data = None
            for arg in args:
                if isinstance(arg, dict):
                    data = arg
                    break
            
            if data is None:
                for key, value in kwargs.items():
                    if isinstance(value, dict):
                        data = value
                        break
            
            if data is None:
                logger.error("No dict argument found to validate")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
            
            try:
                # Validate data against model
                validated_data = model(**data)
                
                # Replace the original dict with the validated model
                if data in args:
                    args_list = list(args)
                    args_list[args.index(data)] = validated_data
                    args = tuple(args_list)
                else:
                    for key, value in kwargs.items():
                        if value is data:
                            kwargs[key] = validated_data
                            break
                
                return await func(*args, **kwargs)
            
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=e.errors()
                )
            
            except Exception as e:
                logger.error(f"Unexpected error during validation: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        return wrapper
    
    return decorator

def validate_ticker(func: Callable):
    """Decorator to validate ticker symbols.
    
    This decorator checks if the ticker exists in the data source.
    
    Returns:
        Decorated function
        
    Example:
        @validate_ticker
        def get_stock_data(ticker: str, source: DataSourceEnum):
            # ticker is guaranteed to exist in the source
            pass
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        from app.utils.data_sources import get_data_source
        from app.models import DataSource as DataSourceEnum
        
        # Extract ticker and source from arguments
        ticker = None
        source = None
        
        for key, value in kwargs.items():
            if key == 'ticker':
                ticker = value
            elif key == 'source':
                source = value
        
        if ticker is None or source is None:
            logger.error("Missing ticker or source argument")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing ticker or source parameter"
            )
        
        try:
            # Check if ticker exists in source
            with get_data_source(source) as data_source:
                tickers = data_source.get_tickers()
                
                if ticker not in tickers:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Ticker {ticker} not found in {source.value}"
                    )
            
            return await func(*args, **kwargs)
        
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        
        except Exception as e:
            logger.error(f"Error validating ticker: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    return wrapper