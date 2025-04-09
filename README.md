# Stock Analysis Backend

A FastAPI-based backend application for stock data analysis, featuring asynchronous task processing and data transformation capabilities.

## Features

### API Endpoints
- **Task Management**: Create, list, and check status of data analysis tasks
- **Stock Data**: Retrieve and analyze stock data with various parameters

### Asynchronous Processing
- **Job Queue**: Thread-safe job queue for background task processing
- **Async Workers**: Process data analysis tasks asynchronously
- **Task Status Tracking**: Real-time status updates for long-running tasks

### Data Transformation
- **OHLC Transformations**: Convert, normalize, and calculate returns on stock data
- **Time Series Resampling**: Weekly, monthly, and yearly data resampling
- **Missing Data Handling**: Fill missing dates in time series data

## Architecture

### Core Components
- **FastAPI Application**: RESTful API with dependency injection
- **SQLAlchemy ORM**: Async database interactions with SQLite
- **Async Database Client**: Custom async context manager for database sessions
- **Job Queue**: Thread-safe background task processing
- **Async Task Processor**: Pipeline for data analysis tasks

### Database Models
- **Task**: Represents a data analysis job with parameters and status
- **StockData**: Stores OHLCV data for stocks

## Setup and Installation

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Stack_analysis/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn app.main:app --reload --log-level debug
```

## API Documentation

Once the application is running, access the API documentation at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Testing

Run the test script to verify API functionality:
```bash
python test_task_endpoints.py
```

## Recent Updates

### Async Database Session Management
- Added `AsyncDBSession` context manager for proper async database session handling
- Fixed issues with task processing in async workers
- Ensured consistent database access patterns across the application

### Task Processing Pipeline
- Improved error handling during task processing
- Fixed task status updates in the job queue
- Ensured proper data population in the stock_data table

## License

MIT
