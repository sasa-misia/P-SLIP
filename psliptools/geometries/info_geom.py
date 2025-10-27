# %% === Import necessary libraries
import os
import fiona
import pandas as pd
import shapely.geometry as geom

# %% === Function to check if the input is a shapefile
def _geo_file_checker(
        file_path: str
    ) -> None:
    """
    Check if the input is a shapefile.

    Args:
        file_path (str): The path to the shapefile (or other vectorial file).

    Raises:
        ValueError: If the input is not a shapefile.
        FileNotFoundError: If the shapefile does not exist.
    """
    if not any([file_path.endswith(x) for x in ['shp', 'gpkg', 'geojson']]):
        raise ValueError("Input must be a shapefile.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vectorial file not found: {file_path}")

# %% === Function to get min/max coordinates from a polygon
def get_polygon_extremes(
        polygon: geom.Polygon
    ) -> pd.DataFrame:
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

# %% === Function to get field names and data types from a shapefile
def get_geo_file_attributes(
        file_path: str
    ) -> tuple[list, list]:
    """
    Get attribute names and data types from a shapefile (or other vectorial file).

    Args:
        file_path (str): The path to the shapefile (or other vectorial file).

    Returns:
        tuple(list, list): A tuple containing a list of attribute names and a list of data types.
    """
    _geo_file_checker(file_path)
    
    with fiona.open(file_path, 'r') as src:
        schema = src.schema
        attributes = list(schema['properties'].keys())
        data_types = list(schema['properties'].values())

        return attributes, data_types

# %% === Function to get field values from a shapefile
def get_geo_file_field_values(
        file_path: str, 
        attribute: str, 
        sort: bool = False
    ) -> list[str]:
    """
    Get attribute values from a shapefile (or other vectorial file) as strings.
    
    Args:
        file_path (str): The path to the shapefile (or other vectorial file).
        attribute (str): The name of the attribute to get values from.
        
    Returns:
        list: A list of attribute values as strings.
    """
    _geo_file_checker(file_path)
    
    with fiona.open(file_path, 'r') as src:
        if attribute not in src.schema['properties']:
            raise ValueError(f"Attribute [{attribute}] not found in vectorial file.")
        
        if sort:
            attribute_values = sorted(set([str(feature['properties'][attribute]) for feature in src])) # sorted always returns a list
        else:
            attribute_values = [str(feature['properties'][attribute]) for feature in src]

        return attribute_values
    
# %%
