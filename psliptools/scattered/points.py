# %% === Import necessary modules
import os
import warnings
import pandas as pd
import numpy as np
import scipy.spatial

from ..rasters.coordinates import _check_and_convert_coords, are_coords_geographic

# %% === Metod to obtain nearest point from a list of points
def get_closest_point_id(
        x: float | list | np.ndarray | pd.Series,
        y: float| list | np.ndarray | pd.Series,
        x_ref: list | np.ndarray | pd.Series,
        y_ref: list | np.ndarray | pd.Series
    ):
    """
    Get the index of the point in a list of points that is closest to a given coordinate.

    Args:
        x (float | np.ndarray): The longitude(s) or projected x(s) of the coordinate(s), which is used to find the closest point to x_list and y_list.
        y (float | np.ndarray): The latitude(s) or projected y(s) of the coordinate(s), which is used to find the closest point to x_list and y_list.
        x_ref (list | np.ndarray | pd.Series): The longitude or projected x list of points based on which the distance is calculated.
        y_ref (list | np.ndarray | pd.Series): The latitude or projected y list of points based on which the distance is calculated.

    Returns:
        tuple(np.ndarray, np.ndarray): A tuple containing the index of the point that is closest to the coordinate and the distance to the point.
    """
    # Convert input coordinates to numpy arrays
    x, y = _check_and_convert_coords(x, y)
    x_ref, y_ref = _check_and_convert_coords(x_ref, y_ref)
    
    # Check coordinate systems consistency
    if not (
        (are_coords_geographic(x, y) and are_coords_geographic(x_ref, y_ref)) or \
        (not are_coords_geographic(x, y) and not are_coords_geographic(x_ref, y_ref))
        ):
        raise ValueError("The provided coordinates are not in the same coordinate system. Please convert them to the same coordinate system before using them.")
    
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError('x and y must be 1d arrays!')
    if x.size != y.size:
        raise ValueError('x and y must have the same size!')
    
    if x_ref.ndim > 1 or y_ref.ndim > 1:
        raise ValueError('x_ref and y_ref must be 1d arrays!')
    if x_ref.size != y_ref.size:
        raise ValueError('x_ref and y_ref must have the same size!')
    
    # === Fast KDTree method
    ref_points = np.column_stack((x_ref, y_ref))
    query_points = np.column_stack((x, y))
    kdtree = scipy.spatial.cKDTree(ref_points)
    point_dist, point_idx = kdtree.query(query_points, k=1)
    
    return point_idx, point_dist
# %%
