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

Run the test scripts to verify API and data processing functionality:
```bash
# Test basic API endpoints
python test_task_endpoints.py

# Test all frequency aggregations
python test_all_frequencies.py

# Test data retrieval endpoints
python test_data_retrieval.py

# Test comprehensive backend functionality
python test_backend_comprehensive.py

# Test error handling and validation
python test_error_handling.py
```

## Recent Updates

### Version 1.5.3 - Robust Error Handling and Validation System
- Implemented global error handling middleware for consistent error responses
- Added rate limiting middleware to prevent API abuse
- Created comprehensive validation decorators for request data
- Enhanced API endpoints with proper validation and rate limiting
- Added unit tests for validation and rate limiting components
- Created detailed documentation for the error handling system

### Version 1.5.2 - Enhanced Frequency Aggregation and Error Handling
- Fixed ME (Month-End) frequency aggregation to properly handle period calculations
- Enhanced error handling in the async task service
- Added detailed logging throughout the aggregation process
- Updated AggregatedStockData schema with additional return fields
- Added comprehensive test scripts for all frequency aggregations
- Improved data transformation capabilities with better error handling

### Async Database Session Management
- Added `AsyncDBSession` context manager for proper async database session handling
- Fixed issues with task processing in async workers
- Ensured consistent database access patterns across the application

### Task Processing Pipeline
- Improved error handling during task processing
- Fixed task status updates in the job queue
- Ensured proper data population in the stock_data table

### Error Handling and Validation
- Global error handling middleware with consistent JSON responses
- Validation decorators for various data types and constraints
- Rate limiting with token bucket algorithm
- Comprehensive documentation in `docs/error_handling.md`

## License

MIT
