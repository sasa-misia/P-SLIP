#%% # Import necessary modules
import os
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import sys
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    get_analysis_environment,
    add_input_file,
    save_variable,
    load_variable,
    LOG_CONFIG
)

from psliptools import file_selector, load_georaster, convert_coords_to_geo, resample_raster

#%% Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

#%%
# Definisci le costanti e le variabili
FOLD_VAR = 'path/to/fold/var'
DTM_TYPE = 0  # o 1 o 2
ORTHOPHOTO_ANSWER = True

# Leggi i file di dati DTM
def read_dtm_files(fold_raw_dtm, dtm_type):
    dtm_files = []
    for file in os.listdir(fold_raw_dtm):
        if file.endswith('.tif') and dtm_type in file:
            dtm_files.append(os.path.join(fold_raw_dtm, file))
    return dtm_files

# Elabora i dati DTM
def process_dtm_data(dtm_files):
    dtm_data = []
    for file in dtm_files:
        ds = gdal.Open(file)
        data = ds.GetRasterBand(1).ReadAsArray()
        dtm_data.append(data)
    return dtm_data

# Crea la griglia di base
def create_base_grid(dtm_data):
    grid = grid.create_grid(dtm_data, resolution=10)
    return grid

# Calcola le variabili morfologiche
def calculate_morphological_variables(grid):
    variables = morphology.calculate_morphological_variables(grid)
    return variables

# Crea l'ortofoto
def create_orthophoto(grid, variables):
    if ORTHOPHOTO_ANSWER:
        orthophoto = grid.create_orthophoto(variables)
        return orthophoto
    else:
        return None

# Visualizza i risultati
def visualize_results(grid, orthophoto):
    plt.imshow(grid, cmap='viridis')
    if orthophoto is not None:
        plt.imshow(orthophoto, cmap='viridis', alpha=0.5)
    plt.show()

#%% # Main function to import DEM and define base grid
def main(base_dir: str=None, gui_mode: bool=False, resample_size: int=None):
    """Main function to define the base grid."""
    src_type_abg = 'base_gird'
    src_type_dem = 'morphology'

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

    dtm_data = [] # Initializing an empty list
    abg_data = [] # Initializing an empty list
    for dtm_path in dtm_paths:
        raster_data, raster_profile, raster_x, raster_y = load_georaster(filepath=dtm_path, set_dtype='float32', convert_to_geo=False)
        if resample_size:
            raster_data, raster_profile, raster_x, raster_y = resample_raster(
                in_raster=raster_data, 
                in_profile=raster_profile,
                in_grid_x=raster_x,
                in_grid_y=raster_y,
                resample_method='average',
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
    
    dtm_df = pd.DataFrame(dtm_data) # Convert the list of dictionaries to a DataFrame
    abg_df = pd.DataFrame(abg_data) # Convert the list of dictionaries to a DataFrame

    # ==== Less efficient way
    # # Pre-initialize the DataFrame
    # dtm_df = pd.DataFrame(columns=['path', 
    #                                 'raster_data', 
    #                                 'raster_profile', 
    #                                 'raster_x', 
    #                                 'raster_y'], index=range(len(dtm_paths)))

    # for i, dtm_path in enumerate(dtm_paths):
    #     raster_data, raster_profile, raster_x, raster_y = load_georaster(filepath=dtm_path, set_dtype='float32', convert_to_geo=True)
    #     dtm_df.iloc[i] = [dtm_path, raster_data, raster_profile, raster_x, raster_y]
    # =====
    
    dtm_files = read_dtm_files(FOLD_RAW_DTM, DTM_TYPE)
    dtm_data = process_dtm_data(dtm_files)
    grid = create_base_grid(dtm_data)
    variables = calculate_morphological_variables(grid)
    orthophoto = create_orthophoto(grid, variables)
    visualize_results(grid, orthophoto)

if __name__ == '__main__':
    # Command line interface
    parser = argparse.ArgumentParser(description="Define the base grid for the analysis, importing the DEM.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    args = parser.parse_args()

    base_grid_vars, morphology_vars = main(base_dir=args.base_dir, gui_mode=False)