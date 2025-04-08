"""
Async database client for the stock analysis application.

This module provides async database connection utilities using SQLAlchemy's
async features.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Ensure the db directory exists
current_dir = os.path.dirname(os.path.abspath(__file__))
db_directory = os.path.join(current_dir, "..", "..", "db")
os.makedirs(db_directory, exist_ok=True)

# Create SQLite database in the db directory
# Note: For SQLite, we use aiosqlite as the driver
DATABASE_URL = f"sqlite+aiosqlite:///{db_directory}/stock_analysis.db"

# Create async engine
async_engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    future=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# Base class for declarative models
Base = declarative_base()

@asynccontextmanager
async def get_async_db():
    """Async context manager for database sessions.
    
    This uses Python's asynccontextmanager decorator to create an async context
    that automatically handles session creation and cleanup.
    
    Yields:
        AsyncSession: The database session
    """
    async_session = AsyncSessionLocal()
    try:
        yield async_session
        await async_session.commit()
    except Exception:
        await async_session.rollback()
        raise
    finally:
        await async_session.close()

async def create_async_tables():
    """Create all tables in the database asynchronously."""
    from app.models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
