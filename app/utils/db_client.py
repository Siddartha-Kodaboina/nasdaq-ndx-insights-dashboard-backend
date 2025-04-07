from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import os


# Ensure the db directory exists
current_dir = os.path.dirname(os.path.abspath(__file__))
db_directory = os.path.join(current_dir, "..", "..", "db")
os.makedirs(db_directory, exist_ok=True)

# Create SQLite database in the db directory
DATABASE_URL = f"sqlite:///{db_directory}/stock_analysis.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

@contextmanager
def get_db():
    """Context manager for database sessions.
    
    This uses Python's contextmanager decorator to create a context
    that automatically handles session creation and cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables in the database."""
    from app.models import Base
    Base.metadata.create_all(bind=engine)