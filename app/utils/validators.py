from functools import wraps
from typing import Callable, Type, Any, Dict, List, Optional, Union, Tuple
from fastapi import HTTPException, status, Request, Query, Path, Depends
from pydantic import BaseModel, ValidationError, Field, validator
import logging
from datetime import datetime, date, timedelta
import inspect
import re

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

def validate_date_range(func: Callable):
    """Decorator to validate date range parameters.
    
    This decorator checks if the date range is valid (from_date <= to_date)
    and that dates are not in the future.
    
    Returns:
        Decorated function
        
    Example:
        @validate_date_range
        def get_stock_data(from_date: datetime, to_date: datetime):
            # dates are guaranteed to be valid
            pass
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract date parameters from kwargs
        from_date = kwargs.get('from_date')
        to_date = kwargs.get('to_date')
        
        # If both dates are provided, validate them
        if from_date is not None and to_date is not None:
            # Convert string dates to datetime if needed
            if isinstance(from_date, str):
                try:
                    from_date = datetime.fromisoformat(from_date)
                    kwargs['from_date'] = from_date
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid from_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                    )
            
            if isinstance(to_date, str):
                try:
                    to_date = datetime.fromisoformat(to_date)
                    kwargs['to_date'] = to_date
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid to_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                    )
            
            # Check if from_date is before to_date
            if from_date > to_date:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="from_date must be before or equal to to_date"
                )
            
            # Check if dates are in the future
            now = datetime.now()
            if from_date > now:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="from_date cannot be in the future"
                )
            
            if to_date > now:
                logger.warning(f"to_date {to_date} is in the future, setting to current time")
                kwargs['to_date'] = now
        
        return await func(*args, **kwargs)
    
    return wrapper

def validate_numeric_range(param_name: str, min_value: Optional[float] = None, max_value: Optional[float] = None):
    """Decorator factory to validate numeric parameters within a range.
    
    Args:
        param_name: The name of the parameter to validate
        min_value: The minimum allowed value (inclusive)
        max_value: The maximum allowed value (inclusive)
    
    Returns:
        Decorator function
        
    Example:
        @validate_numeric_range('limit', min_value=1, max_value=1000)
        def get_data(limit: int = 100):
            # limit is guaranteed to be between 1 and 1000
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract parameter value
            value = kwargs.get(param_name)
            
            # Only validate if the parameter is provided
            if value is not None:
                try:
                    # Convert to float for comparison
                    numeric_value = float(value)
                    
                    # Check minimum value
                    if min_value is not None and numeric_value < min_value:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"{param_name} must be at least {min_value}"
                        )
                    
                    # Check maximum value
                    if max_value is not None and numeric_value > max_value:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"{param_name} must be at most {max_value}"
                        )
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"{param_name} must be a valid number"
                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def validate_query_parameters(func: Callable):
    """Decorator to validate query parameters against function signature.
    
    This decorator checks if the query parameters match the function's
    parameter types and annotations.
    
    Returns:
        Decorated function
        
    Example:
        @validate_query_parameters
        def get_data(limit: int = 100, offset: int = 0):
            # limit and offset are guaranteed to be integers
            pass
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        parameters = sig.parameters
        
        # Validate each parameter
        for name, param in parameters.items():
            if name in kwargs:
                value = kwargs[name]
                
                # Skip validation for None values with default parameters
                if value is None and param.default is not inspect.Parameter.empty:
                    continue
                
                # Get parameter type annotation
                param_type = param.annotation
                if param_type is inspect.Parameter.empty:
                    # No type annotation, skip validation
                    continue
                
                try:
                    # Handle Union types (e.g., Optional[int] = Union[int, None])
                    if hasattr(param_type, '__origin__') and param_type.__origin__ is Union:
                        # Get the non-None type from Optional
                        actual_types = [t for t in param_type.__args__ if t is not type(None)]
                        if len(actual_types) == 1:
                            param_type = actual_types[0]
                    
                    # Convert and validate the parameter value
                    if param_type is int:
                        kwargs[name] = int(value)
                    elif param_type is float:
                        kwargs[name] = float(value)
                    elif param_type is bool:
                        if isinstance(value, str):
                            kwargs[name] = value.lower() in ('true', 't', 'yes', 'y', '1')
                    elif param_type is datetime:
                        if isinstance(value, str):
                            kwargs[name] = datetime.fromisoformat(value)
                    elif param_type is date:
                        if isinstance(value, str):
                            kwargs[name] = date.fromisoformat(value)
                except (ValueError, TypeError) as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid value for parameter '{name}': {str(e)}"
                    )
        
        return await func(*args, **kwargs)
    
    return wrapper

def validate_pagination(max_limit: int = 1000, default_limit: int = 100):
    """Decorator factory to validate pagination parameters.
    
    Args:
        max_limit: The maximum allowed limit value
        default_limit: The default limit value if not provided
    
    Returns:
        Decorator function
        
    Example:
        @validate_pagination(max_limit=500, default_limit=50)
        def get_data(limit: int = None, offset: int = 0):
            # limit is guaranteed to be between 1 and 500, defaulting to 50
            # offset is guaranteed to be >= 0
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Handle limit parameter
            limit = kwargs.get('limit')
            if limit is None:
                kwargs['limit'] = default_limit
            else:
                try:
                    limit = int(limit)
                    if limit < 1:
                        kwargs['limit'] = default_limit
                    elif limit > max_limit:
                        kwargs['limit'] = max_limit
                    else:
                        kwargs['limit'] = limit
                except (ValueError, TypeError):
                    kwargs['limit'] = default_limit
            
            # Handle offset parameter
            offset = kwargs.get('offset')
            if offset is None:
                kwargs['offset'] = 0
            else:
                try:
                    offset = int(offset)
                    if offset < 0:
                        kwargs['offset'] = 0
                    else:
                        kwargs['offset'] = offset
                except (ValueError, TypeError):
                    kwargs['offset'] = 0
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator