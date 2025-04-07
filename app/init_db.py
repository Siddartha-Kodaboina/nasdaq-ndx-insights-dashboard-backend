from app.utils.db_client import create_tables
import os

def init_database():
  """Initilize the database and create all tables"""
  # db exists check
  current_dir = os.path.dirname(os.path.abspath(__file__))
  db_directory = os.path.join(current_dir, "..", "db")
  print(f"DB Directory: {db_directory}")
  
  os.makedirs(db_directory, exist_ok=True)
  create_tables()
  
  print("Database initialized successfully.")
  
if __name__ == "__main__":
  init_database()
  