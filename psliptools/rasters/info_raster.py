#%% # Import necessary modules
import rasterio
import os
import warnings

#%% # Function to get raster information
def get_georaster_info(raster_path, default_crs=4326):
    with rasterio.open(raster_path, 'r') as src: # read-only to improve performance
        src_profile = src.profile
    src.close()
    
    if src_profile is None:
        raise ValueError(f"Unable to read raster file: {raster_path}")
    
    if src_profile.get('crs', None) is None:
        raster_basedir = os.path.dirname(raster_path)
        raster_basename_no_ext = os.path.splitext(os.path.basename(raster_path))[0]
        raster_crs_file = os.path.join(raster_basedir, f"{raster_basename_no_ext}.prj")
        if os.path.exists(raster_crs_file):
            with open(raster_crs_file, 'r') as f:
                raster_crs_file_content = f.read()
            f.close()
            src_profile['crs'] = rasterio.crs.CRS.from_string(raster_crs_file_content)
        else:
            warnings.warn(f"CRS not found for raster file: {raster_path}! The default CRS (EPSG {default_crs}) will be used.")
            src_profile['crs'] = rasterio.crs.CRS.from_epsg(default_crs)
    
    if src_profile.get('transform', None) in [None, rasterio.transform.Affine(1, 0, 0, 0, 1, 0)]:
        raise ValueError(f"Unable to read raster file: {raster_path}")

    return src_profile

# %%
