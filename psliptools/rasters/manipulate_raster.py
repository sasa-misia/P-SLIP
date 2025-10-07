# %% === Import necessary modules
import numpy as np
import rasterio
import rasterio.warp
import warnings
import shapely
import scipy.interpolate
import scipy.spatial
from .coordinates import (
    get_projected_crs_from_bbox,
    convert_bbox, 
    create_grid_from_bbox, 
    get_pixels_inside_polygon,
    get_xy_grids_from_profile,
    get_bbox_from_profile
)

# %% === Function to replace values in a raster
def replace_values(
        raster: np.ndarray, 
        old_value: np.ndarray, 
        new_value: np.ndarray
    ) -> np.ndarray:
    """
    Replace values in a raster.

    Args:
        raster (np.ndarray): The raster to modify.
        old_value (np.ndarray): The value to replace.
        new_value (np.ndarray): The value to replace the old value with.

    Returns:
        np.ndarray: The modified raster.
    """
    # Convert old_value and new_value to numpy array
    if not isinstance(old_value, np.ndarray):
        old_value = np.array(old_value)
    if not isinstance(new_value, np.ndarray):
        new_value = np.array(new_value)

    # Check that old_value and new_value have the same shape
    if old_value.shape != new_value.shape:
        raise ValueError('old_value and new_value must have the same shape')

    # Check that old_value and new_value are scalar or vector
    if old_value.ndim > 1 or new_value.ndim > 1:
        raise ValueError('old_value and new_value must be scalar or vector')

    out_raster = raster.copy()  # Create a copy of the raster to avoid modifying the original
    for old, new in zip(old_value, new_value):
        out_raster[out_raster == old] = new
    return out_raster

# %% === Function to resample raster
def resample_raster(
        in_raster: np.ndarray, 
        in_profile: dict,
        resample_method: str='nearest',
        new_size: np.ndarray=[10, 10],
        poly_mask: shapely.geometry.Polygon | shapely.geometry.MultiPolygon=None
    ) -> np.ndarray:
    """
    Resample a raster to a new size.

    Args:
        in_raster (np.ndarray): The input raster.
        in_profile (dict): A dictionary containing the raster profile.
        in_grid_x (np.ndarray): Array of x coordinates.
        in_grid_y (np.ndarray): Array of y coordinates.
        resample_method (str, optional): The resampling method. Defaults to 'nearest'.
        new_size (np.ndarray, optional): The new size of the pixels in the raster (in meters). Defaults to [10, 10].
    Returns:
        tuple[np.ndarray, dict, np.ndarray, np.ndarray]: Tuple containing the resampled raster, the raster profile, the x and y coordinates of the resampled raster.
    """
    bbox = get_bbox_from_profile(in_profile)
    if in_profile['crs'].is_geographic:
        utm_crs = get_projected_crs_from_bbox(bbox)
        bbox_utm = convert_bbox(in_profile['crs'].to_epsg(), utm_crs.to_epsg(), bbox)
    else:
        utm_crs = in_profile['crs']
        bbox_utm = bbox
    
    if not isinstance(new_size, np.ndarray):
        new_size = np.array(new_size)
    if new_size.size == 1:
        new_size = np.array([new_size, new_size])
    if new_size.size != 2:
        raise ValueError('new_size must have 2 values (pixels_x, pixels_y)')
    
    old_size = np.array([
        round(abs(bbox_utm[2] - bbox_utm[0]) / in_profile['width']), # in_profile['width'] is the number of columns (horizontal pixels) in the raster
        round(abs(bbox_utm[3] - bbox_utm[1]) / in_profile['height']) # in_profile['height'] is the number of rows (vertical pixels) in the raster
    ])

    if new_size[0] < old_size[0] or new_size[1] < old_size[1]:
        warnings.warn(
            f'new_size ({new_size[0]} x {new_size[1]}) is smaller than the old size of the raster '
            f'({old_size[0]} x {old_size[1]}). The resampled raster will be more detailed!', stacklevel=2
        )
    
    out_pixel_res_width = round(abs(bbox_utm[2] - bbox_utm[0]) / new_size[0])
    out_pixel_res_height = round(abs(bbox_utm[3] - bbox_utm[1]) / new_size[1])

    possible_resample_methods = ['nearest', 'bilinear', 'cubic', 'cubic_spline', 'average', 'rms', 'mode', 'lanczos', 'max', 'min', 'med', 'q1', 'q3', 'sum']
    try:
        resample_obj = getattr(rasterio.enums.Resampling, resample_method)
    except AttributeError:
        raise ValueError(f"resample_method must be one of the following: {possible_resample_methods}")
    
    out_grid_x, out_grid_y, out_profile = create_grid_from_bbox(bbox, [out_pixel_res_width, out_pixel_res_height], in_profile)
    out_raster = np.zeros((out_profile['count'], out_profile['height'], out_profile['width']), dtype=in_raster.dtype)
    rasterio.warp.reproject(
        in_raster, 
        src_crs=in_profile['crs'],
        src_transform=in_profile['transform'],
        dst_crs=out_profile['crs'],
        dst_transform=out_profile['transform'],
        dst_resolution=(out_pixel_res_width,out_pixel_res_height),
        src_nodata=in_profile['nodata'],
        dst_nodata=out_profile['nodata'],
        resampling=resample_obj,
        destination=out_raster
    )

    mask_matrix = np.ones((out_profile['height'], out_profile['width']), dtype=bool)
    if poly_mask is not None:
        mask_matrix = get_pixels_inside_polygon(geo_polygon=poly_mask, raster_profile=out_profile)
    return out_raster, out_profile, out_grid_x, out_grid_y, mask_matrix

# %% === Function to interpolate scatter data to raster
def interpolate_scatter_to_raster(
        in_data: np.ndarray, 
        in_coord_x: np.ndarray, 
        in_coord_y: np.ndarray, 
        resample_method: str='nearest',
        out_profile: dict=None,
        out_grid_x: np.ndarray=None,
        out_grid_y: np.ndarray=None,
        fill_value: float=np.nan,
        max_distance: float=None
    ) -> np.ndarray:
    """
    Interpolate values from one raster to another.

    Args:
        in_data (np.ndarray): The input data to interpolate.
        in_coord_x (np.ndarray): The x coordinates of the input data.
        in_coord_y (np.ndarray): The y coordinates of the input data.
        resample_method (str, optional): The resampling method. Defaults to 'nearest'.
        out_profile (dict, optional): A dictionary containing the raster profile. If provided, out_grid_x and out_grid_y will be derived from it.
        out_grid_x (np.ndarray, optional): The x coordinates of the output grid. If None, it will be derived from out_profile.
        out_grid_y (np.ndarray, optional): The y coordinates of the output grid. If None, it will be derived from out_profile.

    Returns:
        np.ndarray: The interpolated raster.
    """
    if not isinstance(in_data, np.ndarray) or not isinstance(in_coord_x, np.ndarray) or not isinstance(in_coord_y, np.ndarray):
        raise TypeError('in_data, in_coord_x, and in_coord_y must be numpy arrays')

    if in_data.ndim != 1 or in_coord_x.ndim != 1 or in_coord_y.ndim != 1:
        raise ValueError('in_data, in_coord_x, and in_coord_y must be 1D arrays')
    if in_data.size != in_coord_x.size or in_data.size != in_coord_y.size:
        raise ValueError('in_data, in_coord_x, and in_coord_y must have the same size')
    if out_profile:
        if out_grid_x or out_grid_y:
            raise ValueError('If out_profile is provided, out_grid_x and out_grid_y must not be provided')
        out_grid_x, out_grid_y = get_xy_grids_from_profile(out_profile)
    else:
        if not isinstance(out_grid_x, np.ndarray) or not isinstance(out_grid_y, np.ndarray):
            raise TypeError('out_grid_x and out_grid_y must be numpy arrays')
        if out_grid_x.ndim != 2 or out_grid_y.ndim != 2:
            raise ValueError('out_grid_x and out_grid_y must be 2D arrays')
        if out_grid_x.shape != out_grid_y.shape:
            raise ValueError('out_grid_x and out_grid_y must have the same shape')
    
    if np.isnan(in_data).any() or np.isinf(in_data).any():
        raise ValueError('in_data contains NaN or Inf')
    if np.isnan(in_coord_x).any() or np.isinf(in_coord_x).any():
        raise ValueError('in_coord_x contains NaN or Inf')
    if np.isnan(in_coord_y).any() or np.isinf(in_coord_y).any():
        raise ValueError('in_coord_y contains NaN or Inf')
    
    out_raster_data = np.zeros(out_grid_x.shape, dtype=in_data.dtype)

    points = np.column_stack((in_coord_x, in_coord_y))
    grid_points = (out_grid_x, out_grid_y)

    allowed_methods = ['nearest', 'linear', 'cubic']
    if resample_method not in allowed_methods:
        raise ValueError(f"resample_method must be one of {allowed_methods}")
    
    out_raster_data = scipy.interpolate.griddata(
        points, 
        in_data, 
        grid_points, 
        method=resample_method, 
        fill_value=fill_value,
    )

    if max_distance:
        # === Fast method using KDTree for nearest neighbor search
        max_distance = float(max_distance)
        if max_distance <= 0:
            raise ValueError('max_distance must be greater than 0')
        # KDTree for fast nearest neighbor search
        kdtree = scipy.spatial.cKDTree(points)
        grid_coords = np.stack([out_grid_x.ravel(), out_grid_y.ravel()], axis=1)
        min_dists, _ = kdtree.query(grid_coords, k=1)
        min_dists = min_dists.reshape(out_grid_x.shape)
        out_raster_data[min_dists > max_distance] = fill_value
    return out_raster_data

# %%