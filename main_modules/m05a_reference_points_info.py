# %% === Import necessary modules
import os
import pandas as pd
import numpy as np
import sys
import argparse
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    REFERENCE_POINTS_FILENAME
)

# Importing necessary modules from psliptools
from psliptools.rasters import (
    create_bbox_from_grids,
    get_projected_epsg_code_from_bbox,
    convert_coords,
    get_closest_pixel_idx,
    generate_grids_from_indices,
    pick_point_from_1d_idx
)

from psliptools.utilities import (
    select_file_prompt
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger()
logger.info(f"=== Obtain reference points info ===")

# %% === Methods to extract reference points info
MORPHOLOGY_NAMES = ['elevation', 'slope', 'aspect', 'profile_curvature', 'planform_curvature', 'twisting_curvature']
PARAMETER_NAMES = ['GS', 'gd', 'c', 'cr', 'phi', 'kt', 'beta', 'A', 'lambda', 'n', 'E', 'ni']

def get_paths_and_shapes(
        env: AnalysisEnvironment, 
        association_df: pd.DataFrame,
        abg_df: pd.DataFrame
    ) -> tuple[list | str, list[tuple[int, int]]]:
    """Helper function to get paths and shapes"""
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

    return parameters_csv_path, abg_shapes

def get_morphology_grids(
        env: AnalysisEnvironment,
        selected_morphology: str
    ) -> dict[str, list[np.ndarray]]:
    """Helper function to get morphology grids"""
    if not all([x in MORPHOLOGY_NAMES for x in selected_morphology]):
        raise ValueError(f"Please select one or more of the following morphologies: {MORPHOLOGY_NAMES}")
    
    dtm_df = env.load_variable(variable_filename='dtm_vars.pkl')['dtm']
    angles_df = env.load_variable(variable_filename='morphology_vars.pkl')['angles']
    curvatures_df = env.load_variable(variable_filename='morphology_vars.pkl')['curvatures']

    morph_grids = {morph: [] for morph in selected_morphology}
    for morph in morph_grids.keys():
        if morph == 'elevation':
            morph_grids[morph] = dtm_df['elevation'].to_list()
        elif morph == 'slope':
            morph_grids[morph] = angles_df['slope'].to_list()
        elif morph == 'aspect':
            morph_grids[morph] = angles_df['aspect'].to_list()
        elif morph == 'profile_curvature':
            morph_grids[morph] = curvatures_df['profile'].to_list()
        elif morph == 'planform_curvature':
            morph_grids[morph] = curvatures_df['planform'].to_list()
        elif morph == 'twisting_curvature':
            morph_grids[morph] = curvatures_df['twisting'].to_list()
        else:
            raise ValueError(f"{morph} not recognized as a valid morphology name during generation of grids.")

    return morph_grids

def get_parameters_grids(
        association_df: pd.DataFrame,
        parameters: list[str],
        shapes: list[tuple[int, int]],
        parameters_csv_paths: str | list[str], 
        clases_column: str='class_id',
        out_type: str='float32',
        no_data: float | int | str=0
    ) -> dict[str, list[np.ndarray]]:
    """Helper function to get parameters grids"""
    par_grids = {par: [] for par in parameters}
    for par in par_grids.keys():
        par_grids[par] = generate_grids_from_indices(
            indices=association_df['abg_idx_1d'],
            classes=association_df['parameter_class'],
            shapes=shapes,
            csv_paths=parameters_csv_paths,
            csv_parameter_column=par,
            csv_classes_column=clases_column,
            out_type=out_type,
            no_data=no_data
        )

    return par_grids

def convert_abg_and_ref_points_to_prj(
        abg_df: pd.DataFrame,
        ref_points_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper function to convert reference points to projected"""
    # Find projected epsg
    abg_first_bbox = create_bbox_from_grids(coords_x=abg_df['longitude'][0], coords_y=abg_df['latitude'][0])
    prj_epsg = get_projected_epsg_code_from_bbox(geo_bbox=abg_first_bbox)

    # Convert abg
    abg_prj_df = abg_df.copy()
    abg_prj_df.rename(columns={'longitude': 'prj_x', 'latitude': 'prj_y'}, inplace=True)
    for i, row in abg_df.iterrows():
        curr_prj_x, curr_prj_y = convert_coords(
            crs_in=4326,
            crs_out=prj_epsg,
            in_coords_x=row['longitude'],
            in_coords_y=row['latitude']
        )
        abg_prj_df.loc[i, 'prj_x'] = curr_prj_x
        abg_prj_df.loc[i, 'prj_y'] = curr_prj_y

    # Convert reference points
    ref_points_prj_df = ref_points_df.copy()
    ref_points_prj_df.rename(columns={'lon': 'prj_x', 'lat': 'prj_y'}, inplace=True)
    ref_points_prj_x, ref_points_prj_y = convert_coords(
        crs_in=4326,
        crs_out=prj_epsg,
        in_coords_x=ref_points_df['lon'],
        in_coords_y=ref_points_df['lat']
    )
    ref_points_prj_df['prj_x'] = ref_points_prj_x
    ref_points_prj_df['prj_y'] = ref_points_prj_y

    return abg_prj_df, ref_points_prj_df

def update_reference_points_csv(
        abg_df: pd.DataFrame,
        ref_points_csv_path: str,
        morph_grids: dict[str, list[np.ndarray]]=None,
        par_grids: dict[str, list[np.ndarray]]=None,
    ) -> pd.DataFrame:
    """Helper function to update reference points csv"""
    ref_points_df = pd.read_csv(ref_points_csv_path)
    if ref_points_df.empty:
        raise ValueError(f"Reference points CSV ({ref_points_csv_path}) is empty. Please check the file.")
    
    abg_prj_df, ref_points_prj_df = convert_abg_and_ref_points_to_prj(abg_df=abg_df, ref_points_df=ref_points_df)

    for i, row_ref_prj_pnt in ref_points_prj_df.iterrows():
        point_idxs, points_dists = np.zeros(abg_df.shape[0]), np.zeros(abg_df.shape[0])
        for n, (_, row_abg_prj) in enumerate(abg_prj_df.iterrows()): # _ because I don't care about the actual index (different if reordered based on priority), but just the current order of rows
            curr_idx, curr_dst = \
                get_closest_pixel_idx(x=row_ref_prj_pnt['prj_x'], y=row_ref_prj_pnt['prj_y'], x_grid=row_abg_prj['prj_x'], y_grid=row_abg_prj['prj_y'])
            
            if curr_idx.size == 1 and curr_dst.size == 1:
                point_idxs[n], points_dists[n] = curr_idx.item(), curr_dst.item()
            else:
                raise ValueError(f"Not unique match found for point {i} (lon={ref_points_df.loc[i,'lon']}, lat={ref_points_df.loc[i,'lat']}) in DTM n. {n}")
        
        curr_sel_dtm = int(np.nanargmin(points_dists))
        if np.isnan(point_idxs[curr_sel_dtm]):
            warnings.warn(f"No match found for point {i} (lon={ref_points_df.loc[i,'lon']}, lat={ref_points_df.loc[i,'lat']}). Nearest DTM is {curr_sel_dtm}.", stacklevel=2)
            continue

        ref_points_df.loc[i, 'dtm'] = curr_sel_dtm
        ref_points_df.loc[i, 'idx_1d'] = int(point_idxs[curr_sel_dtm])
        ref_points_df.loc[i, 'dist'] = points_dists[curr_sel_dtm]

        if morph_grids is not None:
            for morph in morph_grids.keys():
                ref_points_df.loc[i, morph] = pick_point_from_1d_idx(morph_grids[morph][curr_sel_dtm], point_idxs[curr_sel_dtm], order='C')

        if par_grids is not None:
            for par in par_grids.keys():
                ref_points_df.loc[i, par] = pick_point_from_1d_idx(par_grids[par][curr_sel_dtm], point_idxs[curr_sel_dtm], order='C')
    
    ref_points_df.to_csv(ref_points_csv_path, index=False)

    return ref_points_df

# %% === Main function
def main(
        gui_mode: bool=False, 
        base_dir: str=None,
        ref_points_csv_path: str=None,
        morphology: list[str]=MORPHOLOGY_NAMES,
        parameters: list[str]=PARAMETER_NAMES,
        out_type: str='float32',
        no_parameter_data: float | int | str=0
    ) -> pd.DataFrame:
    """Main function to obtain reference points info."""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    abg_df = env.load_variable(variable_filename='dtm_vars.pkl')['abg']

    association_df = env.load_variable(variable_filename='parameter_vars.pkl')['association_df']

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Reference points file selection ===")
        if ref_points_csv_path is None:
            ref_points_csv_path = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the reference points csv (Default: {REFERENCE_POINTS_FILENAME}): ",
                src_ext='csv',
                default_file=os.path.join(env.folders['user_control']['path'], REFERENCE_POINTS_FILENAME)
            )

    logger.info("Generating reference points info about parameters...")
    parameters_csv_path, abg_shapes = get_paths_and_shapes(env, association_df, abg_df)

    if morphology is None:
        morph_grids = None
    else:
        morph_grids = get_morphology_grids(
            env=env,
            selected_morphology=morphology
        )

    if parameters is None:
        par_grids = None
    else:
        par_grids = get_parameters_grids(
            association_df=association_df,
            parameters=parameters,
            shapes=abg_shapes,
            parameters_csv_paths=parameters_csv_path,
            clases_column='class_id',
            out_type=out_type,
            no_data=no_parameter_data
        )
    
    logger.info("Updating reference points csv...")
    ref_points_df = update_reference_points_csv(
        abg_df=abg_df,
        ref_points_csv_path=ref_points_csv_path,
        morph_grids=morph_grids,
        par_grids=par_grids
    )

    return ref_points_df

# %% === Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Information about the parameters of reference points.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    parser.add_argument('--ref_points_csv_path', type=str, default=None, help="Path to the reference points csv file.")
    parser.add_argument('--morphology', type=list, default=MORPHOLOGY_NAMES, help="List of morphologies to be processed.")
    parser.add_argument('--parameters', type=list, default=PARAMETER_NAMES, help="List of parameters to be processed.")
    parser.add_argument('--out_type', type=str, default='float32', help="Output type for the parameters.")
    parser.add_argument('--no_parameter_data', type=float, default=0, help="No data value for the parameters.")
    args = parser.parse_args()

    ref_points_df = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode,
        ref_points_csv_path=args.ref_points_csv_path,
        morphology=args.morphology,
        parameters=args.parameters,
        out_type=args.out_type,
        no_parameter_data=args.no_parameter_data
    )

# %%