#%% # Import necessary modules
import numpy as np
import pyproj
import rasterio

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
    return np.array([coords_x.min(), coords_y.min(), coords_x.max(), coords_y.max()]) # xmin, ymin, xmax, ymax

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
    
    x = np.linspace(bbox[0], bbox[2], resolution)
    y = np.linspace(bbox[1], bbox[3], resolution)

    if profile:
        profile['width'] = resolution[0]
        profile['height'] = resolution[1]
        profile['blockxsize'] = resolution[0]
        profile['transform'] = rasterio.transform.from_bounds(*bbox, profile['width'], profile['height'])
    return np.meshgrid(x, y), profile

#%% # Function to convert grids
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

    for old, new in zip(old_value, new_value):
        raster[raster == old] = new
    return raster

#%% # Function to resample raster
def resample_raster(raster: np.ndarray, profile: dict, grid_x: np.ndarray, grid_y: np.ndarray, new_size: np.ndarray=[10, 10]) -> np.ndarray:
    """
    Resample a raster to a new size.

    Args:
        raster (np.ndarray): The raster to resample.
        grid_x (np.ndarray): The x coordinates of the raster.
        grid_y (np.ndarray): The y coordinates of the raster.
        new_size (np.ndarray, optional): The new size of the raster, in meters. Defaults to [10, 10].

    Returns:
        np.ndarray: The resampled raster.
    """
    bbox = create_bbox(grid_x, grid_y)
    if profile['crs'].is_geographic:
        utm_crs_list = pyproj.database.query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=pyproj.aoi.AreaOfInterest(
                west_lon_degree=bbox[0],
                south_lat_degree=bbox[1],
                east_lon_degree=bbox[2],
                north_lat_degree=bbox[3]
            ),
        )
        utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
        if not utm_crs.is_projected:
            raise ValueError('The auto-generated UTM CRS is not projected!')
        bbox_utm = convert_bbox(profile['crs'].to_epsg(), utm_crs.to_epsg(), bbox)
    else:
        utm_crs = profile['crs']
        bbox_utm = bbox
    
    pixel_res_x = round(abs(bbox_utm[0] - bbox_utm[2]) / new_size[0])
    pixel_res_y = round(abs(bbox_utm[1] - bbox_utm[3]) / new_size[1])

    res_grid_x, res_grid_y, res_profile = create_grid_from_bbox(bbox, [pixel_res_x, pixel_res_y], profile)
    # TO CHECK!!!
    res_raster = np.zeros((grid_x.shape[0], grid_y.shape[1]))
    for i in range(grid_x.shape[0]):
        for j in range(grid_y.shape[1]):
            res_raster[i, j] = rasterio.warp.reproject(
                raster, 
                src_crs=profile['crs'],
                src_transform=profile['transform'],
                dst_crs=profile['crs'],
                dst_transform=profile['transform'],
                dst_resolution=(pixel_res_x, pixel_res_y),
                src_nodata=profile['nodata'],
                dst_nodata=profile['nodata'],
                resampling=rasterio.warp.Resampling.bilinear,
                dst_array=res_raster,
                dst_index=(i, j)
            )
    
    return res_raster, res_profile, res_grid_x, res_grid_y