from app.utils.db_client import get_db
from app.models import StockData, Task
import pandas as pd

def main():
    with get_db() as db:
        # Check task data first
        tasks = db.query(Task).limit(3).all()
        print("Sample Tasks:")
        for t in tasks:
            print(f"ID: {t.id}, Type: {t.task_type}, Status: {t.status}")
            print(f"Created: {t.created_at}, Updated: {t.updated_at}")
            print(f"Parameters: {t.parameters}")
            print('-' * 80)
        
        # Check stock data with proper date formatting
        data = db.query(StockData).limit(5).all()
        print("\nSample StockData rows with formatted dates:")
        if not data:
            print("No data found in the StockData table.")
        for d in data:
            # Format date properly
            date_str = d.date.strftime('%Y-%m-%d %H:%M:%S') if d.date else "None"
            print(f'ID: {d.id}, Task: {d.task_id}, Ticker: {d.ticker}, Date: {date_str}')
            print(f'  Open: {d.open}, High: {d.high}, Low: {d.low}, Close: {d.close}')
            print(f'  Volume: {d.volume}, Source: {d.source}')
            print('-' * 80)

if __name__ == "__main__":
    main()
