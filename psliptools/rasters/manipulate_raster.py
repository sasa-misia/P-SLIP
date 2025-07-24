#%% # Import necessary modules
import numpy as np
import pyproj
import rasterio
import rasterio.warp
import warnings
from .info_raster import get_xy_grids_from_profile, get_projected_crs_from_bbox, is_geographic_coords

#%% # Function to create bounding box from coordinates
def create_bbox(
        coords_x: np.ndarray, 
        coords_y: np.ndarray
    ) -> np.ndarray:
    """
    Create a bounding box from coordinates.

    Args:
        coords_x (np.ndarray): Array of x coordinates.
        coords_y (np.ndarray): Array of y coordinates.

    Returns:
        np.ndarray: Array of bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    dx_pixel = abs(np.average(coords_x[:, 1] - coords_x[:, 0]))
    dy_pixel = abs(np.average(coords_y[1, :] - coords_y[0, :]))
    out_bbox = np.array([
        coords_x.min() - dx_pixel/2, # left 
        coords_y.min() - dy_pixel/2, # bottom
        coords_x.max() + dx_pixel/2, # right
        coords_y.max() + dy_pixel/2  # top
    ])
    return out_bbox

#%% # Function to create grid from bounding box
def create_grid_from_bbox(
        bbox: np.ndarray, 
        resolution: np.ndarray, 
        profile: dict = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a grid from a bounding box.

    Args:
        bbox (np.ndarray): Array of bounding box coordinates [xmin, ymin, xmax, ymax].
        resolution (np.ndarray): Grid resolution (pixels_x, pixels_y) (ex. [1800, 1800]).

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: Tuple containing the x and y coordinates of the grid and the raster profile (optional).
    """
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)
    if not bbox.size == 4:
        raise ValueError('Bounding box must have 4 values (xmin, ymin, xmax, ymax)')
    
    if not isinstance(resolution, np.ndarray):
        resolution = np.array(resolution)
    if resolution.size == 1:
        resolution = np.array([resolution, resolution])
    if resolution.size != 2:
        raise ValueError('Resolution must have 2 values (pixels_x, pixels_y)')
    if not (resolution[0] == int(resolution[0]) and resolution[1] == int(resolution[1])):
        raise ValueError('Resolution must be integer values (pixels_x, pixels_y)')
    
    if bbox[0] > bbox[2]:
        raise ValueError('Bounding box xmin must be less than xmax')
    if bbox[1] > bbox[3]:
        raise ValueError('Bounding box ymin must be less than ymax')

    if profile:
        out_profile = profile.copy()
        out_profile['width'] = int(resolution[0])
        out_profile['height'] = int(resolution[1])
        out_profile['blockxsize'] = int(resolution[0])
        out_profile['transform'] = rasterio.transform.from_bounds(*bbox, out_profile['width'], out_profile['height'])
        grid_x, grid_y = get_xy_grids_from_profile(out_profile)
    else:
        out_profile = {
            'width': int(resolution[0]),
            'height': int(resolution[1]),
            'blockxsize': int(resolution[0]),
            'blockysize': 1,
            'transform': rasterio.transform.from_bounds(*bbox, resolution[0], resolution[1]),
            'crs': None,
            'nodata': None,
            'dtype': None,
            'count': None,
        }
        dx = (bbox[2] - bbox[0])/resolution[0]
        dy = (bbox[3] - bbox[1])/resolution[1]
        x = dx/2 + np.arange(bbox[0], bbox[2], dx)
        y = np.arange(bbox[3], bbox[1], -dy) - dy/2
        grid_x, grid_y = np.meshgrid(x, y) # Not very precise, because with geographic coordinates, the grid is not square and it is slightly rotated

    return grid_x, grid_y, out_profile

#%% # Function to convert coordinates to desired coordinate reference system
def convert_coords(
        crs_in: int, 
        crs_out: int, 
        in_coords_x: np.ndarray, 
        in_coords_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert coordinates from one coordinate reference system to another.

    Args:
        crs_in (int): The EPSG code of the input coordinate reference system.
        crs_out (int): The EPSG code of the output coordinate reference system.
        in_coords_x (np.ndarray): Array of x coordinates.
        in_coords_y (np.ndarray): Array of y coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the converted x and y coordinates.
    """
    transformer = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
    out_coords_x, out_coords_y = transformer.transform(in_coords_x, in_coords_y)
    # ===== Less efficient
    # crs_in = rasterio.crs.CRS.from_epsg(crs_in)
    # crs_out = rasterio.crs.CRS.from_epsg(crs_out)
    # in_coords_x_flat = in_coords_x.flatten()
    # in_coords_y_flat = in_coords_y.flatten()
    # out_coords_x_flat, out_coords_y_flat = rasterio.warp.transform(crs_in, crs_out, in_coords_x_flat, in_coords_y_flat)
    # out_coords_x = np.array(out_coords_x_flat).reshape(in_coords_x.shape)
    # out_coords_y = np.array(out_coords_y_flat).reshape(in_coords_y.shape)
    # =====
    return out_coords_x, out_coords_y

#%% # Function to convert coordinates to geographic
def convert_coords_to_geo(
        crs_in: int,  
        in_coords_x: np.ndarray, 
        in_coords_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert coordinates from one coordinate reference system to another.

    Args:
        crs_in (int): The EPSG code of the input coordinate reference system.
        in_coords_x (np.ndarray): Array of x coordinates.
        in_coords_y (np.ndarray): Array of y coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the converted longitude and latitude coordinates.
    """
    out_coords_x, out_coords_y = convert_coords(crs_in, 4326, in_coords_x, in_coords_y)
    return out_coords_x, out_coords_y

#%% # Function to convert bbox
def convert_bbox(
        crs_in: int, 
        crs_out: int, 
        bbox: np.ndarray
    ) -> np.ndarray:
    """
    Convert a bounding box from one coordinate reference system to another.

    Args:
        crs_in (int): The EPSG code of the input coordinate reference system.
        crs_out (int): The EPSG code of the output coordinate reference system.
        bbox (np.ndarray): Array of bounding box coordinates [xmin, ymin, xmax, ymax].

    Returns:
        np.ndarray: Array of converted bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    in_coords_x = np.array([bbox[0], bbox[2]])
    in_coords_y = np.array([bbox[1], bbox[3]])
    out_coords_x, out_coords_y = convert_coords(crs_in, crs_out, in_coords_x, in_coords_y)
    return np.array([out_coords_x.min(), out_coords_y.min(), out_coords_x.max(), out_coords_y.max()])

#%% # Function to create a transformer from grids
def transformer_from_grids(
        grid_x: np.ndarray, 
        grid_y: np.ndarray
    ) -> rasterio.transform.Affine:
    a_trans = np.average(grid_x[:, 1] - grid_x[:, 0])
    b_trans = np.average(grid_x[1:, 0] - grid_x[:-1, 0])
    c_trans = grid_x[0, 0] - a_trans/2
    d_trans = np.average(grid_y[0, 1:] - grid_y[0, :-1])
    e_trans = np.average(grid_y[1, :] - grid_y[0, :])
    f_trans = grid_y[0, 0] + e_trans/2
    out_transformer = rasterio.transform.Affine(a_trans, b_trans, c_trans, d_trans, e_trans, f_trans)
    return out_transformer

#%% # Function to convert grids and profile to EPSG:4326 (WGS84)
def convert_grids_and_profile_to_geo(
        crs_in: int, 
        in_grid_x: np.ndarray, 
        in_grid_y: np.ndarray, 
        profile: dict
    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert grids and profile to EPSG:4326 (WGS84).

    Args:
        crs_in (int): The EPSG code of the input coordinate reference system.
        in_grid_x (np.ndarray): Grid (matrix) of x coordinates.
        in_grid_y (np.ndarray): Grid (matrix) of y coordinates.
        profile (dict): A dictionary containing the raster profile.

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: Tuple containing the converted x and y grids, and the raster profile.
    """
    out_lons, out_lats = convert_coords_to_geo(
        crs_in=crs_in,
        in_coords_x=in_grid_x,
        in_coords_y=in_grid_y
    )
    #  ===== Not correct in case of b and d not zero
    # new_bbox = create_bbox(ref_grid_x, ref_grid_y)
    # profile['transform'] = rasterio.transform.from_bounds(
    #     *new_bbox,
    #     profile['width'],
    #     profile['height']
    # )
    # =====
    out_profile = profile.copy() # Copy the profile to avoid modifying the original
    out_profile['transform'] = transformer_from_grids(out_lons, out_lats)
    out_profile['crs'] = rasterio.crs.CRS.from_epsg(4326)
    return out_lons, out_lats, out_profile

#%% # Function to convert grids and profile to projected crs
def convert_grids_and_profile_to_prj(
        in_grid_lon: np.ndarray, 
        in_grid_lat: np.ndarray, 
        profile: dict=None,
        crs_out: int=None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert grids and profile to the desired or nearest projected coordinate reference system.

    Args:
        in_grid_lon (np.ndarray): Array of longitude coordinates.
        in_grid_lat (np.ndarray): Array of latitude coordinates.
        profile (dict): A dictionary containing the raster profile.
        crs_out (int, optional): The EPSG code of the projected output coordinate reference system (default: None).

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: Tuple containing the converted projected x and y grids, and the converted raster profile.
    """
    if not is_geographic_coords(in_grid_lon, in_grid_lat):
        raise ValueError('in_grid_lon and in_grid_lat must be geographic coordinates!')
    
    if crs_out is None:
        bbox_geo = create_bbox(in_grid_lon, in_grid_lat)
        crs_out_obj = get_projected_crs_from_bbox(bbox_geo)
    elif isinstance(crs_out, int):
        crs_out_obj = rasterio.crs.CRS.from_epsg(crs_out)
        if not crs_out_obj.is_projected:
            raise ValueError(f'The EPSG code {crs_out} is not a projected coordinate reference system!')
    else:
        raise ValueError('crs_out must be an integer or None!')
    
    out_coords_x, out_coords_y = convert_coords(
        crs_in=4326,
        crs_out=crs_out_obj.to_epsg(),
        in_coords_x=in_grid_lon,
        in_coords_y=in_grid_lat
    )

    if profile:
        out_profile = profile.copy() # Copy the profile to avoid modifying the original
    else:
        out_profile = {
            'width': int(out_coords_x.shape[1]),
            'height': int(out_coords_x.shape[0]),
            'blockxsize': int(out_coords_x.shape[1]),
            'blockysize': 1,
            'transform': None,
            'crs': None,
            'nodata': None,
            'dtype': None,
            'count': None,
        }
    
    out_profile['transform'] = transformer_from_grids(out_coords_x, out_coords_y)
    out_profile['crs'] = crs_out_obj
    return out_coords_x, out_coords_y, out_profile

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
