# %% === Import necessary modules
import os
import pandas as pd
import numpy as np
import sys
import argparse
import logging
import warnings
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    LOG_CONFIG,
    REFERENCE_POINTS_FILENAME
)

# Importing necessary modules from psliptools
from psliptools.rasters import (
    get_closest_pixel_idx,
    generate_grids_from_indices,
    pick_point_from_1d_idx
)

from psliptools.utilities import (
    select_file_prompt
)

# from psliptools.geometries import (
# )

# Importing necessary modules from main_modules
from env_init import get_or_create_analysis_environment

# %% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# %% === Methods to extract reference points info

# %% === Main function
def main(
        gui_mode: bool=False, 
        base_dir: str=None
    ) -> None:
    """Main function to obtain reference points info."""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    abg_df = env.load_variable(variable_filename='dtm_vars.pkl')['abg']

    association_df = env.load_variable(variable_filename='parameter_vars.pkl')['association_df']

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Reference points file selection ===")
        ref_points_csv_path = select_file_prompt(
            base_dir=env.folders['user_control']['path'],
            usr_prompt=f"Name or full path of the reference points csv (Default: {REFERENCE_POINTS_FILENAME}): ",
            src_ext='csv',
            default_file=os.path.join(env.folders['user_control']['path'], REFERENCE_POINTS_FILENAME)
        )

    ref_points_df = pd.read_csv(ref_points_csv_path)
    if ref_points_df.empty:
        raise ValueError(f"Reference points CSV ({ref_points_csv_path}) is empty. Please check the file.")
    
    parameters_csv_path = []
    for _, row in association_df.iterrows():
        poly_type = row['type']
        poly_subtype = row['subtype']
        match_indices = [i for i, setting in enumerate(env.config['inputs'][poly_type]) 
                            if setting['settings']['source_subtype'] == poly_subtype]
        if len(match_indices) > 1:
            raise ValueError(f"Multiple matches found for poly_subtype '{poly_subtype}' in inputs configuration.")
        elif len(match_indices) == 1:
            parameters_csv_path.append(os.path.join(env.folders['user_control']['path'], 
                                                    env.config['inputs'][poly_type][match_indices[0]]['settings']['parameter_filename']))
        
    if len(set(parameters_csv_path)) == 1:
        parameters_csv_path = parameters_csv_path[0]
    
    abg_shapes = []
    for _, row in abg_df.iterrows():
        abg_shapes.append(row['raster_lon'].shape)
    
    gd_grids = generate_grids_from_indices(
        shapes=abg_shapes,
        indices=association_df['abg_idx_1d'],
        csv_paths=parameters_csv_path,
        class_names=association_df['class_name'],
        parameter_name='gd',
        out_type='float32',
        no_data=0
    )

    for i, row_ref_pnt in ref_points_df.iterrows():
        point_idxs, points_dists = np.zeros(abg_df.shape[0]), np.zeros(abg_df.shape[0])
        for n, (_, row_abg) in enumerate(abg_df.iterrows()): # _ because I don't care about the index but just the current order of rows
            curr_idx, curr_dst = \
                get_closest_pixel_idx(x=row_ref_pnt['lon'], y=row_ref_pnt['lat'], x_grid=row_abg['raster_lon'], y_grid=row_abg['raster_lat'])
            
            if curr_idx.size == 1 and curr_dst.size == 1:
                point_idxs[n], points_dists[n] = curr_idx.item(), curr_dst.item()
            else:
                raise ValueError(f"Not unique match found for point {i} (lon={row_ref_pnt['lon']}, lat={row_ref_pnt['lat']}) in DTM n. {n}")
        
        point_dtm = int(np.nanargmin(points_dists))
        ref_points_df.loc[i, 'dtm'] = point_dtm
        ref_points_df.loc[i, 'idx_1d'] = int(point_idxs[point_dtm])
        ref_points_df.loc[i, 'dist'] = points_dists[point_dtm]
        ref_points_df.loc[i, 'gd'] = pick_point_from_1d_idx(gd_grids[point_dtm], point_idxs[point_dtm], order='C')

    # TODO: Add code to extract reference points info

# %% === Command line interface