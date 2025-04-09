from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from app.utils.db_client import create_tables
from app.queue.job_queue import job_queue
from app.middleware.error_handler import ErrorHandlerMiddleware, create_exception_handlers
from app.middleware.rate_limiter import RateLimitMiddleware
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create routers
from app.routers import tasks, data

# Initialize FastAPI app
app = FastAPI(
    title="Stock Analysis API",
    description="API for stock data analysis and visualization",
    version="0.1.0",
)

# Add exception handlers
exception_handlers = create_exception_handlers()
for exc, handler in exception_handlers.items():
    app.add_exception_handler(exc, handler)

# Register custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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

# Register custom exception handler for database errors
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database error for {request.method} {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Database Error",
            "detail": "A database error occurred while processing your request",
            "path": request.url.path,
            "method": request.method
        }
    )

# Configure middleware
# Note: Middleware is executed in reverse order (last added, first executed)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Add error handling middleware
app.add_middleware(ErrorHandlerMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tasks.router)
app.include_router(data.router)

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()
    print("Database tables created on startup")
    
    # Start the job queue with 3 worker threads
    job_queue.start(num_workers=3)
    print("Job queue started with 3 workers")

# Stop the job queue on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    job_queue.stop()
    print("Job queue stopped")

@app.get("/")
async def root():
    return {"message": "Welcome to Stock Analysis API", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/queue/status")
async def queue_status():
    """Get the status of the job queue."""
    return {
        "queue_size": job_queue.get_queue_size(),
        "active_workers": job_queue.get_active_workers(),
        "jobs": len(job_queue.get_all_jobs())
    }