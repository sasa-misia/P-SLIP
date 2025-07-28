#%% # Import necessary modules
import numpy as np
import rasterio
import rasterio.warp
import warnings
from .coordinates import get_projected_crs_from_bbox, create_bbox, convert_bbox, create_grid_from_bbox

#%% # Function to replace values in a raster
def replace_values(raster: np.ndarray, old_value: np.ndarray, new_value: np.ndarray) -> np.ndarray:
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

#%% # Function to resample raster
def resample_raster(in_raster: np.ndarray, in_profile: dict, in_grid_x: np.ndarray, in_grid_y: np.ndarray, resample_method: str='nearest', new_size: np.ndarray=[10, 10]) -> np.ndarray:
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
    bbox = create_bbox(in_grid_x, in_grid_y)
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
            f'({old_size[0]} x {old_size[1]}). The resampled raster will be more detailed!'
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
    return out_raster, out_profile, out_grid_x, out_grid_y

#%%
