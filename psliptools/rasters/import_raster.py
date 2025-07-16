#%% # Import necessary modules
import rasterio
import warnings
import numpy as np
from rasters.info_raster import get_georaster_info
from rasters.manipulate_raster import convert_coords, create_bbox

#%% # Function to load GeoTIFF raster
def load_georaster(
        filepath: str, 
        set_crs: int=None, 
        set_bbox: list=None, 
        set_nodata: int=None, 
        set_dtype: str=None,
        convert_to_geo: bool=True
    ) -> tuple[np.ndarray, dict]:
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
        tuple[np.ndarray, dict]: A tuple containing the raster data and the raster profile.
    """
    raster_profile = get_georaster_info(raster_path=filepath, set_crs=set_crs, set_bbox=set_bbox)
    
    with rasterio.open(filepath, 'r') as src:
        raster_data = src.read()
        ref_grid_x = np.zeros((src.height, src.width))
        ref_grid_y = np.zeros((src.height, src.width))
        for col in range(src.width):
            ref_grid_x[:, col], ref_grid_y[:, col] = rasterio.transform.xy(
                raster_profile['transform'], 
                np.arange(src.height),
                col
            )
    src.close()

    if convert_to_geo:
        epsg_geo = 4326
        ref_grid_x, ref_grid_y = convert_coords(
            crs_in=raster_profile['crs'].to_epsg(),
            crs_out=epsg_geo,
            in_coords_x=ref_grid_x,
            in_coords_y=ref_grid_y
        )
        prior_bbox = create_bbox(ref_grid_x, ref_grid_y)
        raster_profile['crs'] = rasterio.crs.CRS.from_epsg(epsg_geo)
        raster_profile['transform'] = rasterio.transform.from_bounds(
            *prior_bbox,
            raster_profile['width'],
            raster_profile['height']
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