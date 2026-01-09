# %% === Import necessary modules
import warnings
import numpy as np
import pandas as pd
import shapely.geometry as geom

# %% === Function to get rectangle parameters
def get_rectangle_parameters(
        n_rectangles: int
    ) -> list:
    """
    Get rectangle parameters from user input.

    Args:
        n_rectangles (int): Number of rectangles to define.

    Returns:
        list: List of tuples, each containing (x_min, y_min, x_max, y_max) for each rectangle.
    """
    if isinstance(n_rectangles, float):
        n_rectangles = int(n_rectangles)
        warnings.warn("Number of rectangles was provided as a float. Converted to integer.", stacklevel=2)

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

# %% === Function to create rectangle based on given coordinates
def create_rectangle_polygons(
        rectangle_coordinates: list[tuple[float, float, float, float]]
    ) -> list[geom.Polygon]:
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

# %% === Function to create polygons from points
def create_polygons_from_points(
        x: int | float | list[float] | np.ndarray | pd.Series,
        y: int | float | list[float] | np.ndarray | pd.Series, 
        buffer: float | int=10, # same unit as x and y!
        shape: str='circle'
    ) -> list[geom.Polygon]:
    """
    Create polygons from a list of points.

    Args:
        x (list): List of x coordinates.
        y (list): List of y coordinates.
        buffer (float, optional): Buffer size to apply in the units of x and y (default is 10).
        shape (str, optional): Shape of the polygon ('circle' or 'square') (default is 'circle').

    Returns:
        list: List of shapely.geometry.Polygon objects, each representing a polygon.
    """
    if not isinstance(x, np.ndarray):
        x = np.atleast_1d(x)
    if not isinstance(y, np.ndarray):
        y = np.atleast_1d(y)
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions.")
    if x.size != y.size:
        raise ValueError("x and y must have the same number of elements.")
    if x.ndim > 1:
        raise ValueError("x and y must be 1D arrays.")
    
    if not isinstance(buffer, (int, float)):
        raise TypeError("Buffer must be a float or int.")
    
    if not isinstance(shape, str) :
        raise TypeError("Shape must be a string.")
    if shape not in ['circle', 'square']:
        raise ValueError("Shape must be 'circle' or 'square'.")
    
    if shape == 'circle':
        polygons = [geom.Point(xy).buffer(buffer) for xy in zip(x, y)]
    elif shape == 'square':
        polygons = [geom.Polygon([
            (xy[0] - buffer, xy[1] - buffer), 
            (xy[0] + buffer, xy[1] - buffer), 
            (xy[0] + buffer, xy[1] + buffer), 
            (xy[0] - buffer, xy[1] + buffer)
        ]) for xy in zip(x, y)]
    else:
        raise ValueError("Shape type not recognized.")

    return polygons

# %%
