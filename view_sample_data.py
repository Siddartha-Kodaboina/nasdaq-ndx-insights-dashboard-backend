from app.utils.db_client import get_db
from app.models import StockData

def main():
    print('Sample StockData rows:')
    with get_db() as db:
        data = db.query(StockData).limit(5).all()
        if not data:
            print("No data found in the StockData table.")
        for d in data:
            print(f'ID: {d.id}, Task: {d.task_id}, Ticker: {d.ticker}, Date: {d.date}')
            print(f'  Open: {d.open}, High: {d.high}, Low: {d.low}, Close: {d.close}')
            print(f'  Volume: {d.volume}, Source: {d.source}')
            print('-' * 80)

if __name__ == "__main__":
    main()
