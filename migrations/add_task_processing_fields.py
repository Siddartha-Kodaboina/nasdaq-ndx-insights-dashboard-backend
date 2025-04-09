"""
Migration script to add task processing fields to the Task model.

This script adds the following fields to the Task model:
- progress: Float column to track task progress (0-100%)
- error_message: String column to store error messages if a task fails
- estimated_completion_time: DateTime column for the estimated completion time
"""

import sys
import os
from datetime import datetime

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import Column, Float, String, DateTime, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import the database URL directly
from app.utils.db_client import DATABASE_URL
from app.models import Task

def run_migration():
    """Run the migration to add task processing fields."""
    print("Starting migration: add_task_processing_fields")
    
    # Create engine and session
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    connection = engine.connect()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Add the progress column if it doesn't exist
        print("Adding 'progress' column to tasks table...")
        connection.execute(text("ALTER TABLE tasks ADD COLUMN progress FLOAT DEFAULT 0.0"))
        
        # Add the error_message column if it doesn't exist
        print("Adding 'error_message' column to tasks table...")
        connection.execute(text("ALTER TABLE tasks ADD COLUMN error_message TEXT"))
        
        # Add the estimated_completion_time column if it doesn't exist
        print("Adding 'estimated_completion_time' column to tasks table...")
        connection.execute(text("ALTER TABLE tasks ADD COLUMN estimated_completion_time TIMESTAMP"))
        
        # Commit the changes
        session.commit()
        print("Migration completed successfully!")
    
    except Exception as e:
        # Rollback in case of error
        session.rollback()
        print(f"Migration failed: {str(e)}")
        raise
    
    finally:
        # Close the session and connection
        session.close()
        connection.close()

if __name__ == "__main__":
    run_migration()
