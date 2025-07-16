#%% # Import necessary modules
import rasterio
import os
import warnings

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
def get_georaster_info(raster_path: str, set_crs: int=None, set_bbox: list=None) -> dict:
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

# %%
