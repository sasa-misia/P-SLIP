import shapely.geometry as geom
import warnings

def get_rectangle_parameters(n_rectangles: int) -> list:
    """
    Get rectangle parameters from user input.

    Args:
        n_rectangles (int): Number of rectangles to define.

    Returns:
        list: List of tuples, each containing (lon_min, lon_max, lat_min, lat_max) for each rectangle.
    """
    if isinstance(n_rectangles, float):
        n_rectangles = int(n_rectangles)
        warnings.warn("Number of rectangles was provided as a float. Converted to integer.")

    if not isinstance(n_rectangles, int):
        raise TypeError("Number of rectangles must be an integer.")
    
    if n_rectangles < 1:
        raise ValueError("Number of rectangles must be at least 1.")
    
    rectangle_params = []
    for i in range(n_rectangles):
        print(f"Rectangle {i + 1} parameters:")
        lon_min = float(input("  Lon min [°]: "))
        lon_max = float(input("  Lon max [°]: "))
        lat_min = float(input("  Lat min [°]: "))
        lat_max = float(input("  Lat max [°]: "))
        
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError("Invalid rectangle coordinates: ensure lon_min < lon_max and lat_min < lat_max.")
        
        rectangle_params.append((lon_min, lon_max, lat_min, lat_max))
    return rectangle_params

def create_rectangle_polygons(rectangle_coordinates: list) -> list:
    """
    Create rectangular polygons from a list of (lon_min, lon_max, lat_min, lat_max).

    Args:
        rectangle_coordinates (list): List of tuples/lists, each with (lon_min, lon_max, lat_min, lat_max) as float.

    Returns:
        list: List of shapely.geometry.Polygon objects, each representing a rectangle.

    Raises:
        ValueError: If input coordinates are not valid.
    """
    polygons = []
    for params in rectangle_coordinates:
        lon_min, lon_max, lat_min, lat_max = params
        poly = geom.Polygon([
            (lon_min, lat_min), (lon_max, lat_min),
            (lon_max, lat_max), (lon_min, lat_max)
        ])
        polygons.append(poly)
    return polygons