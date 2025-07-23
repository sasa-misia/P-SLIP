import shapely.geometry as geom
import warnings

def get_rectangle_parameters(n_rectangles: int) -> list:
    """
    Get rectangle parameters from user input.

    Args:
        n_rectangles (int): Number of rectangles to define.

    Returns:
        list: List of tuples, each containing (x_min, y_min, x_max, y_max) for each rectangle.
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
        x_min = float(input("  x min [°]: "))
        x_max = float(input("  x max [°]: "))
        y_min = float(input("  y min [°]: "))
        y_max = float(input("  y max [°]: "))

        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Invalid rectangle coordinates: ensure x_min < x_max and y_min < y_max.")

        rectangle_params.append((x_min, y_min, x_max, y_max))
    return rectangle_params

def create_rectangle_polygons(rectangle_coordinates: list) -> list:
    """
    Create rectangular polygons from a list of (x_min, y_min, x_max, y_max).

    Args:
        rectangle_coordinates (list): List of tuples/lists, each with (x_min, y_min, x_max, y_max) as float.

    Returns:
        list: List of shapely.geometry.Polygon objects, each representing a rectangle.

    Raises:
        ValueError: If input coordinates are not valid.
    """
    polygons = []
    for params in rectangle_coordinates:
        x_min, y_min, x_max, y_max = params
        poly = geom.Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max)
        ])
        polygons.append(poly)
    return polygons