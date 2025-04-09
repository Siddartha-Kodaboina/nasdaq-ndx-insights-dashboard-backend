"""
Global error handling middleware for the FastAPI application.

This middleware provides consistent error handling across the application,
ensuring that all errors are properly logged and formatted in the response.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from typing import Union, Dict, Any, Callable
import logging
import traceback
import json
import os
from starlette.types import ASGIApp, Receive, Scope, Send

# Configure logger
logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware:
    """
    Middleware for handling exceptions globally across the application.
    
    This middleware catches all exceptions, logs them appropriately,
    and returns a consistent JSON response to the client.
    """
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware with the FastAPI application."""
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Process the request and handle any exceptions."""
        if scope["type"] != "http":
            # Pass through non-HTTP requests (like WebSockets)
            await self.app(scope, receive, send)
            return
            
        # Create a request object
        request = Request(scope)
        
        # Define a send wrapper to catch exceptions
        async def _send_wrapper(message):
            await send(message)
        
        try:
            # Process the request normally
            await self.app(scope, receive, send)
        except Exception as exc:
            # Handle the exception and generate a response
            response = await self._handle_exception(request, exc)
            # Send the response
            await response(scope, receive, send)
    
    async def _handle_exception(self, request: Request, exc: Exception):
        """Handle different types of exceptions and return appropriate responses."""
        if isinstance(exc, RequestValidationError):
            # Handle FastAPI validation errors
            return await self.handle_validation_error(request, exc)
        elif isinstance(exc, ValidationError):
            # Handle Pydantic validation errors
            return await self.handle_validation_error(request, exc)
        elif isinstance(exc, SQLAlchemyError):
            # Handle database errors
            return await self.handle_database_error(request, exc)
        else:
            # Handle all other exceptions
            return await self.handle_generic_error(request, exc)
    
    async def handle_validation_error(self, request: Request, exc: Union[RequestValidationError, ValidationError]):
        """Handle validation errors from FastAPI or Pydantic."""
        # Extract error details
        if isinstance(exc, RequestValidationError):
            errors = exc.errors()
        else:
            errors = exc.errors()
        
        # Log the error
        logger.error(f"Validation error for {request.method} {request.url.path}: {errors}")
        
        # Return a formatted response
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": errors,
                "path": request.url.path,
                "method": request.method
            }
        )
    
    async def handle_database_error(self, request: Request, exc: SQLAlchemyError):
        """Handle database errors from SQLAlchemy."""
        # Log the error with traceback
        logger.error(f"Database error for {request.method} {request.url.path}: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Return a formatted response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Database Error",
                "detail": "A database error occurred while processing your request",
                "path": request.url.path,
                "method": request.method
            }
        )
    
    async def handle_generic_error(self, request: Request, exc: Exception):
        """Handle all other types of exceptions."""
        # Log the error with traceback
        logger.error(f"Unhandled exception for {request.method} {request.url.path}: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Determine if this is a development environment
        # In production, we wouldn't want to expose the full error details
        is_dev = os.environ.get("ENVIRONMENT", "development").lower() == "development"
        
        # Create the error response
        error_response = {
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred while processing your request",
            "path": request.url.path,
            "method": request.method
        }
        
        # Add more details in development mode
        if is_dev:
            error_response["exception"] = str(exc)
            error_response["traceback"] = traceback.format_exc().split("\n")
        
        # Return a formatted response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )

# Exception handlers for FastAPI
def create_exception_handlers() -> Dict[Any, Callable]:
    """
    Create a dictionary of exception handlers for FastAPI.
    
    These handlers will be registered with the FastAPI app to handle
    specific exception types with custom responses.
    
    Returns:
        Dict[Any, Callable]: A dictionary mapping exception types to handler functions
    """
    handlers = {}
    
    # Handler for RequestValidationError
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors from FastAPI."""
        logger.error(f"Validation error for {request.method} {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
                "path": request.url.path,
                "method": request.method
            }
        )
    
    # Handler for SQLAlchemyError
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
        """Handle database errors from SQLAlchemy."""
        logger.error(f"Database error for {request.method} {request.url.path}: {str(exc)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Database Error",
                "detail": "A database error occurred while processing your request",
                "path": request.url.path,
                "method": request.method
            }
        )
    
    # Register the handlers
    handlers[RequestValidationError] = validation_exception_handler
    handlers[SQLAlchemyError] = sqlalchemy_exception_handler
    
    return handlers
