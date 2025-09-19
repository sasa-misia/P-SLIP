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
    KNOWN_OPTIONAL_STATIC_INPUT_TYPES
)

# Importing necessary modules from psliptools
from psliptools.rasters import (
    raster_within_polygon,
    get_1d_idx_from_2d_mask
)

from psliptools.utilities import (
    compare_dataframes_columns,
    reorder_list_prompt
)

from psliptools.geometries import (
    union_polygons,
    check_and_report_polygons_alignment,
    detect_intersections_between_polygons,
    resolve_polygons_intersections
)

# Importing necessary modules from main_modules
from env_init import get_or_create_analysis_environment

# %% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# %% === Methods for main parameters association
def get_raw_associated_df(
        env: AnalysisEnvironment
    ) -> pd.DataFrame:
    """Get the raw associated dataframe (classes still must be merged)"""
    raw_associated_df = pd.DataFrame()

    for curr_par_type in KNOWN_OPTIONAL_STATIC_INPUT_TYPES:
        for curr_par in env.config['inputs'][curr_par_type]:
            if "source_subtype" in curr_par['settings']:
                curr_par_subtype = curr_par['settings']['source_subtype']
            else:
                curr_par_subtype = None

            if not "parameter_filename" in curr_par['settings']:
                warnings.warn(f"No parameter association found for type:{curr_par_type}; subtype: {curr_par_subtype}. Skipping...")
                continue # Skip to the next parameter

            association_filepath = os.path.join(env.folders['user_control']['path'], curr_par['settings']['association_filename'])

            if curr_par_subtype:
                rel_filename = f"{curr_par_type}_{curr_par_subtype}"
            else:
                rel_filename = f"{curr_par_type}"

            curr_prop_df = env.load_variable(variable_filename=f"{rel_filename}_vars.pkl")['prop_df']
            association_df = pd.read_csv(association_filepath)

            is_prop_df_in_association_df = compare_dataframes_columns(
                dataframe1=curr_prop_df,
                dataframe2=association_df,
                columns_df1=['class_name', 'parameter_class'],
                columns_df2=['class_name', 'parameter_class'],
                row_order=False
            )

            if not is_prop_df_in_association_df.all():
                raise ValueError(f"Property dataframe must be updated to match the association dataframe. Please run the association script first.")
            
            curr_associated_df = curr_prop_df.loc[curr_prop_df['parameter_class'].notna(), :].loc[:, ['class_name', 'geometry', 'parameter_class']]
            curr_associated_df['type'] = curr_par_type
            curr_associated_df['subtype'] = curr_par_subtype
            raw_associated_df = pd.concat([raw_associated_df, curr_associated_df], axis=0, ignore_index=True)
    
    return raw_associated_df

def group_associated_df(
        raw_associated_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Get the merged associated dataframe (classes merged based on parameter class, type and subtype).

    It is important to maintain these three parameters because type and subtype 
    are used to give the file to pick (each property may have a different filename),
    while parameter_class gives the class to pick from that specific file.
    """
    grouped_df_w_subtype = raw_associated_df.groupby(
        ['parameter_class', 'type', 'subtype'], 
        as_index=False
    ).agg(
        {
            'class_name': list, 
            'geometry': lambda x: union_polygons(x.tolist())
        }
    )

    grouped_df_no_subtype = raw_associated_df.loc[raw_associated_df['subtype'].isna(), :].groupby(
        ['parameter_class', 'type'], 
        as_index=False
    ).agg(
        {
            'subtype': lambda x: x.iloc[0],
            'class_name': list, 
            'geometry': lambda x: union_polygons(x.tolist())
        }
    )

    associated_df = pd.concat(
        [
            grouped_df_w_subtype, 
            grouped_df_no_subtype
        ], 
        axis=0,
        ignore_index=True
    )

    return associated_df

def align_and_index_associated_df(
        associated_df: pd.DataFrame,
        abg_df: pd.DataFrame
    ) -> pd.DataFrame:
    poly_intersections_list = detect_intersections_between_polygons(associated_df['geometry'], start_indices_from_1=True)
    if any(poly_intersections_list):
        associated_df['geometry'] = resolve_polygons_intersections(
            polygons=associated_df['geometry']
        )

    abg_idx_1d = []
    for r, (_, row_poly) in enumerate(associated_df.iterrows()):
        abg_idx_1d.append([np.array([], dtype=np.uint32) for _ in range(len(abg_df))])
        curr_poly = row_poly['geometry']
        for c, (_, row_grid) in enumerate(abg_df.iterrows()):
            lon_grid = row_grid['raster_lon']
            lat_grid = row_grid['raster_lat']
            
            is_in_poly, curr_mask = raster_within_polygon(curr_poly, x_grid=lon_grid, y_grid=lat_grid)
            if is_in_poly:
                abg_idx_1d[r][c] = get_1d_idx_from_2d_mask(curr_mask, order='C')
    
    associated_df['abg_idx_1d'] = abg_idx_1d

    return associated_df

# %% === Main function
def main(
        gui_mode: bool=False, 
        base_dir: str=None
    ) -> Dict[str, Any]:
    """Main function to create soil and vegetation grids."""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    abg_df = env.load_variable(variable_filename='dtm_vars.pkl')['abg']

    raw_associated_df = get_raw_associated_df(env)

    associated_df = group_associated_df(raw_associated_df)

    polygons_alignment_report = check_and_report_polygons_alignment(associated_df['geometry'])
    if not polygons_alignment_report['aligned']:
        poly_intersections_list = detect_intersections_between_polygons(associated_df['geometry'], start_indices_from_1=True)
        classes_str = [str(x).replace('[', '').replace(']', '').replace("'", '') for x in associated_df["class_name"]]
        type_and_subtype_str = [f'{x} ({y})' for x, y in zip(associated_df["type"], associated_df["subtype"])]
        if gui_mode:
            raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
        else:
            print(f'\n=== Polygon intersection detected ===')
            for p, poly_intersections in enumerate(poly_intersections_list):
                print(f'\nCurrent polygon index: {p+1} of {len(poly_intersections_list)}')
                print(f'Type (subtype): {type_and_subtype_str[p]}')
                print(f'Classes: {classes_str[p]}')
                print(f'Intersections with polygons: {poly_intersections}')
            
            _, priority_order = reorder_list_prompt(obj_list=classes_str, obj_type=type_and_subtype_str, start_indices_from_1=False)
        
        associated_df = associated_df.iloc[priority_order]

    associated_df = align_and_index_associated_df(associated_df, abg_df)

    parameter_vars = {'association_df': associated_df, 'originally_aligned': polygons_alignment_report['aligned']}

    env.save_variable(variable_to_save=parameter_vars, variable_filename="parameter_vars.pkl")

    return parameter_vars
    
# %% === Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Index the polygons of parameters according to the base grid.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    args = parser.parse_args()

    parameters_vars = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode
    )

# %%
