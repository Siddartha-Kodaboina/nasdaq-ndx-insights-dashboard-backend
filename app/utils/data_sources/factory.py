from typing import Dict, Type
from app.models import DataSource as DataSourceEnum
from app.utils.data_sources.base import DataSource

class DataSourceFactory:
    """Factory class for creating data source instances.
    
    This class implements the Factory pattern to create the appropriate
    data source based on the requested source type.
    """
    
    _sources: Dict[DataSourceEnum, Type[DataSource]] = {}
    
    @classmethod
    def register(cls, source_type: DataSourceEnum, source_class: Type[DataSource]):
        """Register a data source class for a specific source type.
        
        Args:
            source_type: The enum value for the source type
            source_class: The class to instantiate for this source type
        """
        cls._sources[source_type] = source_class
    
    @classmethod
    def get_source(cls, source_type: DataSourceEnum) -> DataSource:
        """Get an instance of the appropriate data source.
        
        Args:
            source_type: The enum value for the source type
            
        Returns:
            An instance of the appropriate data source
            
        Raises:
            ValueError: If the source type is not registered
        """
        if source_type not in cls._sources:
            raise ValueError(f"Data source {source_type} not registered")
        
        return cls._sources[source_type]()
    
    @classmethod
    def get_all_sources(cls) -> Dict[DataSourceEnum, DataSource]:
        """Get instances of all registered data sources.
        
        Returns:
            A dictionary mapping source types to data source instances
        """
        return {source_type: source_class() for source_type, source_class in cls._sources.items()}