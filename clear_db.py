#!/usr/bin/env python
"""
Script to clear all data from the database tables.
This is useful for testing purposes.
"""

import logging
from app.utils.db_client import SessionLocal, engine, Base
from app.models import Task, StockData
from sqlalchemy import text
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def clear_database():
    """Clear all data from the database tables."""
    with get_db() as db:
        # Delete all records from the stock_data table
        logger.info("Deleting all records from stock_data table...")
        db.query(StockData).delete()
        
        # Delete all records from the tasks table
        logger.info("Deleting all records from tasks table...")
        db.query(Task).delete()
        
        # Try to reset the autoincrement counters if the table exists
        try:
            logger.info("Attempting to reset autoincrement counters...")
            db.execute(text("DELETE FROM sqlite_sequence WHERE name='stock_data'"))
            db.execute(text("DELETE FROM sqlite_sequence WHERE name='tasks'"))
            logger.info("Autoincrement counters reset successfully")
        except Exception as e:
            logger.warning(f"Could not reset autoincrement counters: {e}")
            logger.info("This is normal if the database doesn't use sqlite_sequence for autoincrement")
        
        # Commit the changes
        db.commit()
    
    logger.info("Database cleared successfully!")

if __name__ == "__main__":
    clear_database()
