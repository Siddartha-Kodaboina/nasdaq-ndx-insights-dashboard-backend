from app.utils.data_sources import DataSource, DataSourceFactory, get_data_source
from app.models import DataSource as DataSourceEnum

print("DataSource abstract base class:", DataSource)
print("DataSourceFactory:", DataSourceFactory)
print("get_data_source context manager:", get_data_source)

print("Test completed successfully!")