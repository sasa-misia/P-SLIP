import shapely.geometry as geom

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