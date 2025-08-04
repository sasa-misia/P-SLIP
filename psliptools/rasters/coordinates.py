# %% === Import necessary modules
import rasterio
import rasterio.features
import os
import warnings
import numpy as np
import pyproj
import shapely
import scipy.spatial

# %% === Function to get raster crs
def _get_crs(raster_path: str, set_crs: int=None) -> rasterio.crs.CRS:
    """
    Get the crs of a raster file.

    Args:
        raster_path (str): The path to the raster file.
        set_crs (int, optional): The EPSG code of the coordinate reference system. Defaults to None.

    Returns:
        rasterio.crs.CRS: The crs of the raster file.
    """
    raster_basedir = os.path.dirname(raster_path)
    raster_basename_no_ext = os.path.splitext(os.path.basename(raster_path))[0]
    raster_crs_file = os.path.join(raster_basedir, f"{raster_basename_no_ext}.prj")
    if os.path.exists(raster_crs_file):
        with open(raster_crs_file, 'r') as f:
            raster_crs_file_content = f.read()
        f.close()
        crs_obj = rasterio.crs.CRS.from_string(raster_crs_file_content)
    else:
        warnings.warn(f"CRS not found for raster file: {raster_path}! The set CRS (EPSG {set_crs}) will be used.")
        if set_crs is None:
            raise ValueError(f"Unable to read crs of raster file: {raster_path}. Please specify a code as the set_crs argument.")
        crs_obj = rasterio.crs.CRS.from_epsg(set_crs)
    return crs_obj

# %% === Function to get raster information
def get_georaster_info(raster_path: str, set_crs: int=None, set_bbox: list | np.ndarray=None) -> dict:
    """
    Get information about a GeoTIFF raster file.

    Args:
        raster_path (str): The path to the GeoTIFF raster file.
        set_crs (int, optional): The EPSG code of the coordinate reference system. Defaults to None.
        set_bbox (list, optional): The bounding box coordinates [xmin, ymin, xmax, ymax]. Defaults to None.

    Returns:
        dict: A dictionary containing the raster profile.
    """
    with rasterio.open(raster_path, 'r') as src: # read-only to improve performance
        src_profile = src.profile
    src.close()
    
    if src_profile is None:
        raise ValueError(f"Unable to read raster file: {raster_path}")
    
    if src_profile.get('crs', None) is None:
        src_profile['crs'] = _get_crs(raster_path, set_crs)
    
    if src_profile.get('transform', None) in [None, rasterio.transform.Affine(1, 0, 0, 0, 1, 0)]:
        if set_bbox is None:
            raise ValueError(f"Unable to read transform of raster file: {raster_path}. Please specify a bounding box as the set_bbox argument.")
        
        if len(set_bbox) != 4:
            raise ValueError(f"Invalid bounding box: {set_bbox}. Please specify a valid bounding box as the set_bbox argument (minx, miny, maxx, maxy).")
        
        if not(all(isinstance(item, (int, float)) for item in set_bbox)):
            raise ValueError(f"Invalid bounding box: {set_bbox}. Please specify each element as an integer or float.")
        
        src_profile['transform'] = rasterio.transform.from_bounds(*set_bbox, src_profile['width'], src_profile['height'])
    return src_profile

# %% === Function to create xy grids from profile
def get_xy_grids_from_profile(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Create x and y grids from a raster profile.

    Args:
        profile (dict): A dictionary containing the raster profile.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the x and y grids.
    """
    ref_grid_x, ref_grid_y = np.zeros((2, profile['height'], profile['width'])) # single command row and unpacked array
    for col in range(profile['width']):
        ref_grid_x[:, col], ref_grid_y[:, col] = rasterio.transform.xy(
            profile['transform'], 
            np.arange(profile['height']),
            col
        )
    return ref_grid_x, ref_grid_y

# %% === Function to get the bounding box from a profile
def get_bbox_from_profile(
        profile: dict
    ) -> np.ndarray[(1, 4), float]:
    """
    Get the bounding box from a raster profile.

    Args:
        profile (dict): A dictionary containing the raster profile.

    Returns:
        np.ndarray: Array of bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    if 'transform' not in profile:
        raise ValueError("The profile does not contain a 'transform' key.")
    if not isinstance(profile['transform'], rasterio.transform.Affine):
        raise ValueError("The 'transform' key in the profile must be of type rasterio.transform.Affine.")
    if 'height' not in profile or 'width' not in profile:
        raise ValueError("The profile must contain 'height' and 'width' keys.")
    
    bbox = rasterio.transform.array_bounds(
        profile['height'], 
        profile['width'], 
        profile['transform']
    )
    return np.array(bbox)

# %% === Function to get the projected epsg code from a bounding box
def get_projected_epsg_code_from_bbox(geo_bbox: list | np.ndarray) -> int:
    """
    Get the projected crs from a bounding box.

    Args:
        bbox (list): The bounding box coordinates, in longitude and latitude as a list of 4 elements (min_lon, min_lat, max_lon, max_lat).

    Returns:
        int: The EPSG code of the projected coordinate reference system.
    """
    if len(geo_bbox) != 4:
        raise ValueError(f"Invalid bounding box: {geo_bbox}. Please specify a valid bounding box as a list of 4 elements (min_lon, min_lat, max_lon, max_lat).")
    
    if not(-180 <= geo_bbox[0] <= 180 and -90 <= geo_bbox[1] <= 90 and -180 <= geo_bbox[2] <= 180 and -90 <= geo_bbox[3] <= 90):
        raise ValueError(f"Invalid bounding box: {geo_bbox}. Please specify a valid bounding box in lat and lon as a list of 4 elements (min_lon, min_lat, max_lon, max_lat).")
    
    if not(isinstance(geo_bbox, np.ndarray)):
        geo_bbox = np.array(geo_bbox)
           
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=geo_bbox[0],
            south_lat_degree=geo_bbox[1],
            east_lon_degree=geo_bbox[2],
            north_lat_degree=geo_bbox[3]
        ),
    )

    if not utm_crs_list:
        raise ValueError(f"Unable to find a suitable UTM CRS for the bounding box: {geo_bbox}")
    
    if len(utm_crs_list) > 1:
        warnings.warn(f"Multiple UTM CRS found for the bounding box: {geo_bbox}. Using the first one: {utm_crs_list[0].code}")

    utm_epsg_code = utm_crs_list[0].code
    if isinstance(utm_epsg_code, str):
        utm_epsg_code = int(utm_epsg_code)
    if not isinstance(utm_epsg_code, int):
        print(type(utm_epsg_code))  # This will raise an error if utm_epsg_code is not an integer
        raise ValueError(f"Invalid UTM EPSG code: {utm_epsg_code}. Expected an integer value.")
    return utm_epsg_code

# %% === Function to get the projected crs from a bounding box
def get_projected_crs_from_bbox(geo_bbox: list | np.ndarray) -> rasterio.crs.CRS:
    """ 
    Get the projected coordinate reference system (CRS) from a bounding box.

    Args:
        geo_bbox (list): The bounding box coordinates, in longitude and latitude as a list of 4 elements (min_lon, min_lat, max_lon, max_lat).

    Returns:
        rasterio.crs.CRS: The projected CRS object.
    """
    epsg_code = get_projected_epsg_code_from_bbox(geo_bbox)

    utm_crs = rasterio.crs.CRS.from_epsg(epsg_code)
    if not utm_crs.is_projected:
        raise ValueError('The auto-generated UTM CRS is not projected!')
    return utm_crs

# %% === Check if arrays contain geographic coordinates
def are_coords_geographic(lon: np.ndarray, lat: np.ndarray) -> bool:
    """
    Check if arrays contain geographic coordinates.

    Args:
        lon (np.ndarray): Array of longitude coordinates.
        lat (np.ndarray): Array of latitude coordinates.

    Returns:
        bool: True if the arrays contain geographic coordinates, False otherwise.
    """
    if not isinstance(lon, np.ndarray) or not isinstance(lat, np.ndarray):
        lon = np.array(lon)
        lat = np.array(lat)
    return np.all(lon >= -180) and np.all(lon <= 180) and np.all(lat >= -90) and np.all(lat <= 90)

# %% === Function to create bounding box from coordinates
def create_bbox_from_grids(
        coords_x: np.ndarray, 
        coords_y: np.ndarray
    ) -> np.ndarray:
    """
    Create a bounding box from coordinates.

    Args:
        coords_x (np.ndarray): 2D array of x coordinates.
        coords_y (np.ndarray): 2D array of y coordinates.

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

# %% === Function to create grid from bounding box
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

# %% === Function to convert coordinates to desired coordinate reference system
def convert_coords(
        crs_in: int, 
        crs_out: int, 
        in_coords_x: np.ndarray, 
        in_coords_y: np.ndarray,
        force_ndarray: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert coordinates from one coordinate reference system to another.

    Args:
        crs_in (int): The EPSG code of the input coordinate reference system.
        crs_out (int): The EPSG code of the output coordinate reference system.
        in_coords_x (np.ndarray): Array of x coordinates.
        in_coords_y (np.ndarray): Array of y coordinates.
        force_ndarray (bool, optional): Force the input (and output) coordinates to be numpy arrays. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the converted x and y coordinates.
    """
    if force_ndarray:
        if not isinstance(in_coords_x, np.ndarray) or not isinstance(in_coords_y, np.ndarray):
            in_coords_x = np.array(in_coords_x)
            in_coords_y = np.array(in_coords_y)
        if in_coords_x.size != in_coords_y.size:
            raise ValueError('Coordinates must have the same size')
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

# %% === Function to convert coordinates inside lists
def convert_coords_from_list(
        crs_in: int,
        crs_out: int,
        x_coords: list,
        y_coords: list
    ) -> tuple[list, list]:
    """
    Convert coordinates inside lists from one coordinate reference system to another.

    Args:
        crs_in (int): The EPSG code of the input coordinate reference system.
        crs_out (int): The EPSG code of the output coordinate reference system.
        x_coords (list): List of x coordinates, where each element of the list contains an array of x coordinates.
        y_coords (list): List of y coordinates, where each element of the list contains an array of y coordinates.

    Returns:
        tuple[list, list]: Tuple containing the converted x and y coordinates in lists of numpy arrays.
    """
    out_coords_x, out_coords_y = [], []
    for x_coords, y_coords in zip(x_coords, y_coords):
        x_temp, y_temp = convert_coords(crs_in, crs_out, x_coords, y_coords, force_ndarray=True)
        out_coords_x.append(x_temp)
        out_coords_y.append(y_temp)
    return out_coords_x, out_coords_y

# %% === Function to convert coordinates to geographic
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

# %% === Function to convert bbox
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

# %% === Function to create a transformer from grids
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

# %% === Function to convert grids and profile to EPSG:4326 (WGS84)
def convert_grids_and_profile_to_geo(
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
    out_profile = profile.copy() # Copy the profile to avoid modifying the original
    crs_in = out_profile['crs'].to_epsg()
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
    out_profile['transform'] = transformer_from_grids(out_lons, out_lats)
    out_profile['crs'] = rasterio.crs.CRS.from_epsg(4326)
    return out_lons, out_lats, out_profile

# %% === Function to convert grids and profile to projected crs
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
    if not are_coords_geographic(in_grid_lon, in_grid_lat):
        raise ValueError('in_grid_lon and in_grid_lat must be geographic coordinates!')
    
    if crs_out is None:
        bbox_geo = create_bbox_from_grids(in_grid_lon, in_grid_lat)
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

# %% === Function to obtain pixels that are inside a polygon
def get_pixels_inside_polygon(
        geo_polygon: shapely.geometry.Polygon | shapely.geometry.MultiPolygon, 
        raster_profile: dict
    ) -> np.ndarray:
    """
    Get the pixels that are inside a polygon.

    Args:
        geo_polygon (shapely.geometry.Polygon | shapely.geometry.MultiPolygon): The polygon to check against.
        raster_profile (dict): The raster profile dictionary.

    Returns:
        np.ndarray: An array of indices of the pixels that are inside the polygon.
    """
    if not are_coords_geographic(geo_polygon.bounds[2], geo_polygon.bounds[3]):
        raise ValueError("The provided polygon is not in geographic coordinates. Please convert it to geographic coordinates before using it.")
    
    ref_grid_x, ref_grid_y = get_xy_grids_from_profile(raster_profile)
    _, _, raster_profile_geo = convert_grids_and_profile_to_geo(
        in_grid_x=ref_grid_x,
        in_grid_y=ref_grid_y,
        profile=raster_profile
    )
    mask = rasterio.features.geometry_mask([geo_polygon], out_shape=[raster_profile_geo['height'], raster_profile['width']], transform=raster_profile_geo['transform'], invert=True, all_touched=True)
    return mask

# %% === Function to check if raster is within a polygon and return a mask
def raster_within_polygon(
        geo_polygon: shapely.geometry.Polygon | shapely.geometry.MultiPolygon, 
        raster_profile: dict
    ) -> tuple[bool, np.ndarray]:
    """
    Check if any part of the raster is within the given polygon.

    Args:
        geo_polygon (shapely.geometry.Polygon | shapely.geometry.MultiPolygon): The polygon to check against.
        raster_profile (dict): The raster profile dictionary.

    Returns:
        tuple[bool, np.ndarray]: A tuple where the first element is True if the raster is within the polygon, and the second element is a mask of the pixels that are within the polygon.
    """
    mask = get_pixels_inside_polygon(geo_polygon, raster_profile)
    return np.any(mask), mask

# %% === Function to obtain the 1d index of the pixel that is closest to a given coordinate
def get_closest_pixel_idx(
        x: np.ndarray,
        y: np.ndarray,
        x_grid: np.ndarray=None,
        y_grid: np.ndarray=None,
        raster_profile: dict=None,
        fill_outside: bool = True,
        fill_value: float = np.nan
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the 1d index of the pixel that is closest to a given coordinate.

    Args:
        lon (float): The longitude of the coordinate.
        lat (float): The latitude of the coordinate.
        x_grid (np.ndarray, optional): The x grid of the raster. If not provided, it will be obtained from the raster profile.
        y_grid (np.ndarray, optional): The y grid of the raster. If not provided, it will be obtained from the raster profile.
        raster_profile (dict, optional): The raster profile dictionary. If not provided, x_grid and y_grid must be provided.

    Returns:
        int: The 1d index of the pixel that is closest to the coordinate.
    """
    if (x_grid is None and y_grid is None) and raster_profile is None:
        raise ValueError('x_grid + y_grid or raster_profile must be provided!')
    
    if raster_profile:
        ref_grid_x, ref_grid_y = get_xy_grids_from_profile(raster_profile)
    else:
        ref_grid_x = x_grid
        ref_grid_y = y_grid
    
    if not are_coords_geographic(x, y):
        raise ValueError("The provided coordinate is not in geographic coordinates. Please convert it to geographic coordinates before using it.")
    
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('x and y must be 1d arrays!')
    if x.size != y.size:
        raise ValueError('x and y must have the same size!')
    
    # === Fast KDTree method
    grid_points = np.column_stack((ref_grid_x.flatten(), ref_grid_y.flatten()))
    query_points = np.column_stack((x, y))
    kdtree = scipy.spatial.cKDTree(grid_points)
    dst_1d, idx_1d = kdtree.query(query_points, k=1)

    # Fill value for points outside raster boundary (optional)
    if fill_outside:
        pixel_size = np.sqrt((ref_grid_x[1,1] - ref_grid_x[0,0])**2 + (ref_grid_y[1,1] - ref_grid_y[0,0])**2)
        idx_1d = np.where(dst_1d > pixel_size, fill_value, idx_1d)
    return idx_1d, dst_1d

# %%
