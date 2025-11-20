# %% === Import necessary modules
import pandas as pd
import numpy as np
import scipy.spatial
import scipy.interpolate

from ..rasters.coordinates import _check_and_convert_coords, are_coords_geographic

# %% === Function to obtain nearest point from a list of points
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

# %% Function to interpolate scatter data
def interpolate_scatter_to_scatter(
        x_in: np.ndarray | pd.Series | list[float | int], 
        y_in: np.ndarray | pd.Series | list[float | int], 
        data_in: np.ndarray | pd.Series | list[float | int],
        x_out: np.ndarray | pd.Series | list[float | int],
        y_out: np.ndarray | pd.Series | list[float | int],
        interpolation_method: str='nearest',
        fill_value: float | int=np.nan
    ) -> np.ndarray:
    """
    Interpolate values from unstructured scatter data to unstructured scatter data.

    Args:
        x_in (np.ndarray | pd.Series | list[float | int]): The x coordinates of the input data.
        y_in (np.ndarray | pd.Series | list[float | int]): The y coordinates of the input data.
        data_in (np.ndarray | pd.Series | list[float | int]): The values to interpolate.
        x_out (np.ndarray | pd.Series | list[float | int]): The x coordinates of the output data.
        y_out (np.ndarray | pd.Series | list[float | int]): The y coordinates of the output data.
        interpolation_method (str, optional): The interpolation method ('nearest', 'linear', 'cubic', 'rbf', 'idw') (defaults to 'nearest'). Method 'idw' is paticularly useful for rainfall data.
        fill_value (float | int, optional): The fill value. Defaults to np.nan.

    Returns:
        np.ndarray: The interpolated value/s.
    """
    VALID_INTERPOLATION_METHODS = ['nearest', 'linear', 'cubic', 'rbf', 'idw']
    if not isinstance(interpolation_method, str) or interpolation_method not in VALID_INTERPOLATION_METHODS:
        raise TypeError('interpolation_method must be a string')
    if not isinstance(fill_value, (int, float)):
        raise TypeError('fill_value must be a number')

    x_in = np.atleast_1d(x_in)
    y_in = np.atleast_1d(y_in)
    data_in = np.atleast_1d(data_in)
    x_out = np.atleast_1d(x_out)
    y_out = np.atleast_1d(y_out)

    # Ensure all inputs are 1‑D arrays
    for var in [x_in, y_in, data_in, x_out, y_out]:
        if var.ndim != 1:
            raise ValueError("All input arrays must be 1‑D")

    if x_in.size != y_in.size or x_in.size != data_in.size:
        raise ValueError('x_in, y_in and data_in must have the same size!')
    
    if x_out.size != y_out.size:
        raise ValueError('x_out and y_out must have the same size!')
    
    data_out = np.full(x_out.shape, fill_value, dtype=float)

    # Check for minimum points for linear and cubic methods
    if x_in.size < 3 and interpolation_method in ['linear', 'cubic']:
        # Cannot interpolate with fewer than 3 points, return fill_value
        pass  # data_out is already filled with fill_value

    elif interpolation_method == 'nearest':
        for i in range(x_out.size):
            idx = np.argmin(np.abs(x_in - x_out[i]) + np.abs(y_in - y_out[i]))
            data_out[i] = data_in[idx]
    elif interpolation_method == 'linear':
        # Use scipy's griddata for linear interpolation
        points_in = np.column_stack((x_in, y_in))
        points_out = np.column_stack((x_out, y_out))
        data_out = scipy.interpolate.griddata(
            points_in, data_in, points_out, 
            method='linear', fill_value=fill_value
        )
    elif interpolation_method == 'cubic':
        # Use scipy's griddata for cubic interpolation
        points_in = np.column_stack((x_in, y_in))
        points_out = np.column_stack((x_out, y_out))
        data_out = scipy.interpolate.griddata(
            points_in, data_in, points_out, 
            method='cubic', fill_value=fill_value
        )
    elif interpolation_method == 'rbf':
        # Use scipy's RBF interpolation
        rbf = scipy.interpolate.Rbf(x_in, y_in, data_in, function='multiquadric')
        data_out = rbf(x_out, y_out)
    elif interpolation_method == 'idw':
        # Inverse Distance Weighting with power 2
        points_in = np.column_stack((x_in, y_in))
        points_out = np.column_stack((x_out, y_out))
        distances = scipy.spatial.distance.cdist(points_out, points_in, 'euclidean')
        p = 2  # Power for IDW
        for i in range(x_out.size):
            dist = distances[i]
            weights = 1 / (dist ** p)
            weights[dist == 0] = 1  # Handle exact matches
            data_out[i] = np.sum(weights * data_in) / np.sum(weights) if np.sum(weights) > 0 else fill_value
    else:
        raise ValueError(f"Resample method [{interpolation_method}] is not supported. Please choose one of the following: {VALID_INTERPOLATION_METHODS}")
    
    return data_out

# %%
