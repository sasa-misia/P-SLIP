#%% === Import necessary modules
import os
import pandas as pd
import sys
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    get_analysis_environment,
    add_input_file,
    save_variable,
    load_variable,
    LOG_CONFIG
)

from psliptools import file_selector, load_georaster, convert_coords_to_geo, resample_raster, plot_elevation_3d

#%% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

#%% === DEM and Analysis Base Grid (ABG) methods
# Read and import DTM files in a dataframe
def import_dtm_files(
        env, 
        file_type, 
        files_path: list[str], 
        resample_size: tuple[int, int]=None, 
        resample_method: str='average'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtm_data = [] # Initializing an empty list
    abg_data = [] # Initializing an empty list
    cust_id = []
    for idx, dtm_path in enumerate(files_path):
        _, curr_cust_id = add_input_file(env, file_path=dtm_path, file_type=file_type, file_subtype=f'dtm{idx+1}')
        raster_data, raster_profile, raster_x, raster_y = load_georaster(filepath=dtm_path, set_dtype='float32', convert_to_geo=False)
        if resample_size:
            raster_data, raster_profile, raster_x, raster_y = resample_raster(
                in_raster=raster_data, 
                in_profile=raster_profile,
                in_grid_x=raster_x,
                in_grid_y=raster_y,
                resample_method=resample_method,
                new_size=resample_size
            )
        raster_lon, raster_lat = convert_coords_to_geo(
            crs_in=raster_profile['crs'].to_epsg(), 
            in_coords_x=raster_x, 
            in_coords_y=raster_y
        )

        dtm_data.append({
            'path': dtm_path,
            'raster_data': raster_data,
            'raster_profile': raster_profile,
        })

        abg_data.append({
            'raster_lon': raster_lon,
            'raster_lat': raster_lat
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
def main(base_dir: str=None, gui_mode: bool=False, resample_size: int=None, resample_method: str='average'):
    """Main function to define the base grid."""
    src_type = 'dtm'

    # --- User choices section ---
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        if base_dir is None:
            base_dir = input(f"Enter the base directory for the analysis (or press Enter to use the current directory {os.getcwd()}): ").strip(' "')
            if not base_dir:
                base_dir = os.getcwd()
    
    # Get the analysis environment
    env, _ = get_analysis_environment(base_dir=base_dir)

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        dtm_fold = input(f"Enter the folder name where the DTM files are stored (default: {env.inp_dir['dtm']['path']}): ").strip(' "')
        if not dtm_fold:
            dtm_fold = env.inp_dir['dtm']['path']

        dtm_paths = file_selector(base_dir=dtm_fold, src_ext=['tif', '.tiff'])
    
    dtm_df, abg_df, cust_id = import_dtm_files(env, src_type, dtm_paths, resample_size=resample_size, resample_method=resample_method)

    dtm_abg_vars = {'dtm': dtm_df, 'abg': abg_df}

    env.config['inputs'][src_type]['1']['settings'] = {
        'resample_size': resample_size, # None or (x, y)
        'resample_method': resample_method
    }
    env.config['inputs'][src_type]['1']['custom_id'] = [cust_id]
    env.collect_input_files(file_type=[src_type], multi_extension=True)
    
    save_variable(analysis_env=env, variable_to_save=dtm_abg_vars, filename="dtm_abg_vars.pkl")

    # Check-plot
    plot_elevation_3d(dtm_df['raster_data'], abg_df['raster_lon'], abg_df['raster_lat'], projected=True)
    return dtm_abg_vars

if __name__ == '__main__':
    # Command line interface
    parser = argparse.ArgumentParser(description="Define the base grid for the analysis, importing the DEM.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--resample_size', type=int, default=None, help="Resample size for the DEM.")
    parser.add_argument('--resample_method', type=str, default='average', help="Resample method for the DEM.")
    args = parser.parse_args()

    dtm_abg_vars = main(base_dir=args.base_dir, gui_mode=False, resample_size=args.resample_size, resample_method=args.resample_method)