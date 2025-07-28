#%% # Import necessary libraries
import pandas as pd
import shapely.geometry as geom
import fiona
import os

#%% # Function to check if the input is a shapefile
def _shapefile_checker(shapefile_path: str) -> None:
    """
    Check if the input is a shapefile.

    Args:
        shapefile_path (str): The path to the shapefile.

    Raises:
        ValueError: If the input is not a shapefile.
        FileNotFoundError: If the shapefile does not exist.
    """
    if not shapefile_path.endswith('.shp'):
        raise ValueError("Input must be a shapefile.")
    
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

#%% # Function to get min/max coordinates from a polygon
def get_polygon_extremes(polygon: geom.Polygon) -> pd.DataFrame:
    """Get min/max coordinates from a polygon.
    
    Args:
        polygon (geom.Polygon): A shapely Polygon or MultiPolygon object.
        
    Returns:
        pd.DataFrame: A DataFrame with min and max longitude and latitude.
    """
    # if hasattr(polygon, "exterior"):
    #     xs, ys = polygon.exterior.xy
    # else:
    #     raise ValueError("Input must be a shapely Polygon or MultiPolygon (or have an exterior attribute).")
    
    # df = pd.DataFrame(
    #         [[min(xs), min(ys)], [max(xs), max(ys)]],
    #         columns=['lon', 'lat'], 
    #         index=['min', 'max']
    #     )

    poly_bbox =polygon.bounds
    
    df = pd.DataFrame(
            [poly_bbox[0:2], poly_bbox[2:4]],
            columns=['lon', 'lat'], 
            index=['min', 'max']
        )
    return df

#%% # Function to get field names and data types from a shapefile
def get_shapefile_fields(shapefile_path: str) -> tuple[list, list]:
    """
    Get field names and data types from a shapefile.

    Args:
        shapefile_path (str): The path to the shapefile.

    Returns:
        list: A list of field names.
        list: A list of data types.
    """
    _shapefile_checker(shapefile_path)
    
    with fiona.open(shapefile_path, 'r') as src:
        schema = src.schema
        fields = list(schema['properties'].keys())
        data_types = list(schema['properties'].values())
        return fields, data_types

#%% # Function to get field values from a shapefile
def get_shapefile_field_values(shapefile_path: str, field_name: str, sort: bool = False) -> list:
    """
    Get field values from a shapefile as strings.
    
    Args:
        shapefile_path (str): The path to the shapefile.
        field_name (str): The name of the field to get values from.
        
    Returns:
        list: A list of field values as strings.
    """
    _shapefile_checker(shapefile_path)
    
    with fiona.open(shapefile_path, 'r') as src:
        if field_name not in src.schema['properties']:
            raise ValueError(f"Field '{field_name}' not found in shapefile.")
        
        if sort:
            field_values = sorted(set([str(feature['properties'][field_name]) for feature in src])) # sorted always returns a list
        else:
            field_values = [str(feature['properties'][field_name]) for feature in src]
        return field_values