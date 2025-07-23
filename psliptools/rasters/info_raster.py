#%% # Import necessary modules
import rasterio
import os
import warnings
import numpy as np
import pyproj

#%% # Function to get raster crs
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

#%% # Function to get raster information
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

#%% # Function to create xy grids from profile
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

#%% # Function to get the projected epsg code from a bounding box
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

#%% # Function to get the projected crs from a bounding box
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

# %%
