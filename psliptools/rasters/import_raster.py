import rasterio

def load_georaster(filename, nodata=None, dtype=None, **kwargs) -> rasterio.DatasetReader:
    """Load a GeoTIFF raster as a rasterio.DatasetReader object."""
    return rasterio.open(filename, nodata=nodata, dtype=dtype, **kwargs)