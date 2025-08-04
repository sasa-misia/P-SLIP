#%% === Import necessary modules
import os
import pandas as pd
import numpy as np
import sys
import argparse
import logging
import shapely

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    LOG_CONFIG
)

from psliptools import (
    select_files_in_folder_prompt,
    select_dir_prompt,
    load_georaster, 
    convert_coords_to_geo, 
    resample_raster, 
    plot_elevation_3d,
    mask_raster_with_1d_idx
)

from env_init import get_or_create_analysis_environment

#%% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

#%% === DEM and Analysis Base Grid (ABG) methods
# Read and import DTM files in a dataframe
def import_dtm_files(
        env: AnalysisEnvironment,
        file_type: str, 
        files_path: list[str], 
        resample_size: tuple[int, int]=None, 
        resample_method: str='average',
        poly_mask: shapely.geometry.Polygon | shapely.geometry.MultiPolygon=None,
        apply_mask_to_raster: bool=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Import DTM files in a dataframe."""
    dtm_data = [] # Initializing an empty list
    abg_data = [] # Initializing an empty list
    cust_id = []
    for idx, dtm_path in enumerate(files_path):
        raster_data, raster_profile, raster_x, raster_y, mask_matrix = load_georaster(
            filepath=dtm_path,
            set_dtype='float32', 
            convert_to_geo=False, 
            poly_mask=poly_mask
        )

        if raster_data is None: # No pixels inside poly_load_mask_geo
            continue

        _, curr_cust_id = env.add_input_file(file_path=dtm_path, file_type=file_type, file_subtype=f'dtm{idx+1}')
        if resample_size:
            raster_data, raster_profile, raster_x, raster_y, mask_matrix = resample_raster(
                in_raster=raster_data, 
                in_profile=raster_profile,
                resample_method=resample_method,
                new_size=resample_size,
                poly_mask=poly_mask
            )

        raster_lon, raster_lat = convert_coords_to_geo(
            crs_in=raster_profile['crs'].to_epsg(), 
            in_coords_x=raster_x, 
            in_coords_y=raster_y
        )

        mask_idx_1d = np.where(mask_matrix.flatten(order='C')) # C order is the default and means row-major (row by row)

        if apply_mask_to_raster:
            raster_data = mask_raster_with_1d_idx(raster_data, mask_idx_1d, profile=raster_profile)

        dtm_data.append({
            'path': dtm_path,
            'raster_data': raster_data,
            'raster_profile': raster_profile,
        })

        abg_data.append({
            'raster_lon': raster_lon,
            'raster_lat': raster_lat,
            'mask_idx_1d': mask_idx_1d
        })

        cust_id.append(curr_cust_id)
    
    dtm_df = pd.DataFrame(dtm_data) # Convert the list of dictionaries to a DataFrame
    abg_df = pd.DataFrame(abg_data) # Convert the list of dictionaries to a DataFrame

    # # === Less efficient way
    # # Pre-initialize the DataFrame
    # dtm_df = pd.DataFrame(columns=['path', 'raster_data', 'raster_profile'], index=range(len(files_path)))
    # abg_df = pd.DataFrame(columns=['raster_lon', 'raster_lat'], index=range(len(files_path)))

    # for i, dtm_path in enumerate(files_path):
    #     raster_data, raster_profile, raster_x, raster_y = load_georaster(filepath=dtm_path, set_dtype='float32', convert_to_geo=False)
    #     raster_lon, raster_lat = convert_coords_to_geo(
    #         crs_in=raster_profile['crs'].to_epsg(), 
    #         in_coords_x=raster_x, 
    #         in_coords_y=raster_y
    #     )
    #     dtm_df.iloc[i] = [dtm_path, raster_data, raster_profile]
    #     abg_df.iloc[i] = [raster_lon, raster_lat]
    # # === End of less efficient way
    return dtm_df, abg_df, cust_id

#%% === Main function to import DEM and define base grid
def main(base_dir: str=None, gui_mode: bool=False, resample_size: int=None, resample_method: str='average', apply_mask_to_raster: bool=False):
    """Main function to define the base grid."""
    src_type = 'dtm'
    
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    study_area_polygon = env.load_variable(variable_filename='study_area_vars.pkl')['study_area_cln_poly']

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        dtm_fold = select_dir_prompt(default_dir=env.folders['inputs'][src_type]['path'], content_type=src_type)
        dtm_paths = select_files_in_folder_prompt(base_dir=dtm_fold, src_ext=['tif', '.tiff'])
    
    dtm_df, abg_df, cust_id = import_dtm_files(
        env, 
        src_type, 
        dtm_paths, 
        resample_size=resample_size, 
        resample_method=resample_method, 
        poly_mask=study_area_polygon,
        apply_mask_to_raster=apply_mask_to_raster
    )

    dtm_abg_vars = {'dtm': dtm_df, 'abg': abg_df}

    env.config['inputs'][src_type][0]['settings'] = {
        'resample_size': resample_size, # None or (x, y)
        'resample_method': resample_method,
        'apply_mask_to_raster': apply_mask_to_raster
    }
    env.config['inputs'][src_type][0]['custom_id'] = cust_id
    env.collect_input_files(file_type=[src_type], multi_extension=True)
    
    env.save_variable(variable_to_save=dtm_abg_vars, variable_filename="dtm_abg_vars.pkl")

    # Check-plot
    plot_elevation_3d(dtm_df['raster_data'], abg_df['raster_lon'], abg_df['raster_lat'], mask_idx_1d=abg_df['mask_idx_1d'], projected=True)
    return dtm_abg_vars

# %% === Command line interface
if __name__ == '__main__':
    # Command line interface
    parser = argparse.ArgumentParser(description="Define the base grid for the analysis, importing the DEM.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--resample_size', type=int, default=None, help="Resample size for the DEM.")
    parser.add_argument('--resample_method', type=str, default='average', help="Resample method for the DEM.")
    parser.add_argument('--apply_mask_to_raster', type=bool, default=False, help="Apply mask to the raster.")
    args = parser.parse_args()

    dtm_abg_vars = main(
        base_dir=args.base_dir, 
        gui_mode=False, 
        resample_size=args.resample_size, 
        resample_method=args.resample_method,
        apply_mask_to_raster=args.apply_mask_to_raster
    )