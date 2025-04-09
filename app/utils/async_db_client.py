"""
Async database client for the stock analysis application.

This module provides async database connection utilities using SQLAlchemy's
async features.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
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

async def get_async_db():
    """Get an async database session for dependency injection.
    
    This function is designed to be used with FastAPI's dependency injection system.
    It creates a new database session for each request and automatically closes it
    when the request is complete.
    
    Returns:
        AsyncSession: The database session
    """
    logger.info("Creating new async database session for dependency injection")
    async_session = AsyncSessionLocal()
    logger.info(f"Created session object: {type(async_session)}, ID: {id(async_session)}")
    
    try:
        yield async_session
        logger.info("Request completed, committing session")
        await async_session.commit()
    except Exception as e:
        logger.error(f"Error in session: {str(e)}, rolling back")
        await async_session.rollback()
        raise
    finally:
        logger.info("Closing session")
        await async_session.close()


class AsyncDBSession:
    """Async context manager for database sessions.
    
    This class provides an async context manager that can be used with the
    `async with` statement to get a database session.
    
    Example:
        ```python
        async with AsyncDBSession() as db:
            # Use the database session
            result = await db.execute(query)
        ```
    """
    
    async def __aenter__(self) -> AsyncSession:
        """Enter the async context manager.
        
        Returns:
            AsyncSession: The database session
        """
        logger.info("Creating new async database session with context manager")
        self.session = AsyncSessionLocal()
        logger.info(f"Created session object: {type(self.session)}, ID: {id(self.session)}")
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        try:
            if exc_type is not None:
                # An exception occurred, rollback the session
                logger.error(f"Error in session: {exc_val}, rolling back")
                await self.session.rollback()
            else:
                # No exception, commit the session
                logger.info("Context manager exiting, committing session")
                await self.session.commit()
        finally:
            # Always close the session
            logger.info("Closing session")
            await self.session.close()

async def create_async_tables():
    """Create all tables in the database asynchronously."""
    from app.models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
