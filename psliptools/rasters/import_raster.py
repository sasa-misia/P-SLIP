#%% # Import necessary modules
import warnings
import rasterio
import shapely
import numpy as np
from .coordinates import get_georaster_info, get_xy_grids_from_profile, raster_within_polygon, convert_grids_and_profile_to_geo

#%% # Function to load GeoTIFF raster
def load_georaster(
        filepath: str, 
        set_crs: int=None, 
        set_bbox: list=None, 
        set_nodata: int=None, 
        set_dtype: str=None,
        convert_to_geo: bool=False,
        poly_mask_load_geo: shapely.geometry.Polygon | shapely.geometry.MultiPolygon=None
    ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """
    Load a GeoTIFF raster as a rasterio.DatasetReader object.

    Args:
        filepath (str): The path to the GeoTIFF raster file.
        set_crs (int, optional): The EPSG code of the coordinate reference system. Defaults to None.
        set_bbox (list, optional): The bounding box coordinates [xmin, ymin, xmax, ymax]. Defaults to None.
        set_nodata (int, optional): The nodata value. Defaults to None.
        set_dtype (str, optional): The data type of the raster (e.g., 'uint8', 'float32', 'int16', etc.). Defaults to None.
        convert_to_geo (bool, optional): Whether to convert the raster to geographic coordinates. Defaults to True.

    Returns:
        tuple[np.ndarray, dict, np.ndarray, np.ndarray]: Tuple containing the raster data, raster profile, x and y grids.
    """
    raster_profile = get_georaster_info(raster_path=filepath, set_crs=set_crs, set_bbox=set_bbox)
    ref_grid_x, ref_grid_y = get_xy_grids_from_profile(raster_profile)

    if poly_mask_load_geo:
        is_within_polygon, _ = raster_within_polygon(
            geo_polygon=poly_mask_load_geo, 
            raster_profile=raster_profile,
        )
        if not is_within_polygon:
            warnings.warn(f"The raster {filepath} is not within the provided polygon. It will not be loaded and processed.")
            return None, None, None, None
    
    with rasterio.open(filepath, 'r') as src:
        raster_data = src.read()
    src.close()

    if convert_to_geo:
        ref_grid_x, ref_grid_y, raster_profile = convert_grids_and_profile_to_geo(
            in_grid_x=ref_grid_x,
            in_grid_y=ref_grid_y,
            profile=raster_profile
        )

    if raster_profile.get('nodata', None) is None:
        if set_nodata and isinstance(set_nodata, (int, float)):
            raster_profile['nodata'] = set_nodata
        else:
            if raster_data.min() < 0:
                raster_profile['nodata'] = raster_data.min()
                warnings.warn(f"Unable to read nodata value of raster file: {filepath}. The set nodata value ({raster_profile['nodata']}) will be used.")
            else:
                warnings.warn(f"Unable to read nodata value of raster file: {filepath}. Consider to specify a value as the set_nodata argument.")

    if raster_profile.get('dtype', None) is None:
        if set_dtype is not None and isinstance(set_dtype, str):
            allowed_dtypes = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            if set_dtype not in allowed_dtypes:
                raise ValueError(f"Invalid dtype: {set_dtype}. It must be one of the following: {', '.join(allowed_dtypes)}")
            raster_profile['dtype'] = set_dtype
            raster_data = raster_data.astype(set_dtype)
        else:
            warnings.warn(f"Unable to read dtype value of raster file: {filepath}. Consider to specify a value as the set_dtype argument.")
    return raster_data, raster_profile, ref_grid_x, ref_grid_y

# %%
