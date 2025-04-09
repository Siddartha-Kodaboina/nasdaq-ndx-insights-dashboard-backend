from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.db_client import create_tables
from app.queue.job_queue import job_queue
import os

# Create routers
from app.routers import tasks, data

# Initialize FastAPI app
app = FastAPI(
    title="Stock Analysis API",
    description="API for stock data analysis and visualization",
    version="0.1.0",
)

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