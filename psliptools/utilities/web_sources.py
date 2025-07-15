#%% # Import necessary libraries
import owslib
import os
from owslib.wms import WebMapService
import requests

#%% # Function to download raster from WMS
def download_wms_raster(
        url: str, 
        out_path: str, 
        out_epsg: str="4326", 
        bbox: list=None, 
        width: int=4096, 
        height: int=4096
    ) -> None:
    # Creazione dell'oggetto WMS
    wms = WebMapService(url)

    if len(wms.contents.keys()) > 1:
        file_paths = [os.path.join(
            os.path.dirname(out_path), 
            os.path.splitext(os.path.basename(out_path))[0] + "_" + key + ".tif"
        ) for key in wms.contents.keys()]
    elif len(wms.contents.keys()) == 1:
        file_paths = out_path
    else:
        raise ValueError("No layers found in WMS")

    # Richiesta dei dati raster in formato GeoTIFF
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "FORMAT": "format=image/geotiff",
        "SRS": "EPSG:" + out_epsg,  # WGS 84 standard in lat/lon
        "BBOX": (-180, -90, 180, 90),
        "WIDTH": width,
        "HEIGHT": height,
        "TRANSPARENT": "TRUE"
    }

    if bbox:
        params["BBOX"] = bbox

    for key, path in zip(wms.contents.keys(), file_paths):
        params["LAYERS"] = key
        web_image = wms.getmap(
            layers=key, 
            srs=params["SRS"], 
            bbox=params["BBOX"], 
            size=(params["WIDTH"], 
                  params["HEIGHT"]), 
            format=params["FORMAT"],
            transparent=params["TRANSPARENT"]
        )
        with open(path, "wb") as f:
            f.write(web_image.read())

#%%