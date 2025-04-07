from app.utils.data_sources.base import DataSource
from app.utils.data_sources.factory import DataSourceFactory
from app.utils.data_sources.utils import get_data_source, get_all_data_sources
from app.utils.data_sources.csv_source import CSVDataSource
from app.models import DataSource as DataSourceEnum
import os

# Get the absolute path to the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Register data sources
DataSourceFactory.register(
    DataSourceEnum.SOURCE_A, 
    lambda: CSVDataSource(os.path.join(DATA_DIR, "source_a.csv"), DataSourceEnum.SOURCE_A)
)

DataSourceFactory.register(
    DataSourceEnum.SOURCE_B, 
    lambda: CSVDataSource(os.path.join(DATA_DIR, "source_b.csv"), DataSourceEnum.SOURCE_B)
)

__all__ = ['DataSource', 'DataSourceFactory', 'get_data_source', 'get_all_data_sources', 'CSVDataSource']