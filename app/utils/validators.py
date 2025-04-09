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
    logger.info(f"Setting up validate_request_body decorator with model: {model.__name__}")
    
    def decorator(func: Callable):
        logger.info(f"Decorating function: {func.__name__}")
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger.info(f"Validating request body for {func.__name__}")
            logger.info(f"Args: {args}, Kwargs keys: {list(kwargs.keys())}")
            
            # Find the first dict argument
            data = None
            for i, arg in enumerate(args):
                logger.info(f"Checking arg {i}: {type(arg)}")
                if isinstance(arg, dict):
                    data = arg
                    logger.info(f"Found dict arg at position {i}")
                    break
            
            if data is None:
                for key, value in kwargs.items():
                    logger.info(f"Checking kwarg {key}: {type(value)}")
                    if isinstance(value, dict):
                        data = value
                        logger.info(f"Found dict kwarg with key {key}")
                        break
            
            if data is None:
                logger.error("No dict argument found to validate")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
            
            logger.info(f"Found data to validate: {data}")
            
            try:
                # Validate data against model
                logger.info(f"Validating data against {model.__name__}")
                validated_data = model(**data)
                logger.info(f"Validation successful: {validated_data}")
                
                # Convert the validated model back to a dictionary
                # This is needed because in Pydantic v2, models are not subscriptable
                validated_dict = validated_data.model_dump()
                logger.info(f"Converted to dict: {validated_dict}")
                
                # Replace the original dict with the validated dict
                if data in args:
                    logger.info("Replacing dict in args")
                    args_list = list(args)
                    args_list[args.index(data)] = validated_dict
                    args = tuple(args_list)
                else:
                    for key, value in kwargs.items():
                        if value is data:
                            logger.info(f"Replacing dict in kwargs with key {key}")
                            kwargs[key] = validated_dict
                            break
                
                logger.info(f"Calling {func.__name__} with validated data")
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