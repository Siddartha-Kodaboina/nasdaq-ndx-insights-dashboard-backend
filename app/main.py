from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.db_client import create_tables
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

@app.get("/")
async def root():
    return {"message": "Welcome to Stock Analysis API", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}