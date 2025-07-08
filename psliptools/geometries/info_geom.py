import pandas as pd
import shapely.geometry as geom

def get_polygon_extremes(polygon: geom.Polygon) -> pd.DataFrame:
    """Get min/max coordinates from a polygon.
    
    Args:
        polygon (geom.Polygon): A shapely Polygon or MultiPolygon object.
        
    Returns:
        pd.DataFrame: A DataFrame with min and max longitude and latitude.
    """
    if hasattr(polygon, "exterior"):
        xs, ys = polygon.exterior.xy
    else:
        raise ValueError("Input must be a shapely Polygon or MultiPolygon (or have an exterior attribute).")
    
    df = pd.DataFrame(
            [[min(xs), max(xs)], [min(ys), max(ys)]], 
            columns=['lon', 'lat'], 
            index=['min', 'max']
        )
    return df