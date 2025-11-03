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
    REFERENCE_POINTS_FILENAME,
    SUPPORTED_FILE_TYPES
)

# Importing necessary modules from psliptools
from psliptools.rasters import (
    create_bbox_from_grids,
    get_projected_epsg_code_from_bbox,
    convert_coords,
    get_closest_1d_pixel_idx,
    generate_grids_from_indices,
    pick_point_from_1d_idx
)

from psliptools.utilities import (
    select_file_prompt
)

from psliptools.scattered import (
    get_closest_point_id
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger(__name__)
logger.info(f"=== Obtain reference points info ===")

# %% === Helper functions and global variables
MORPHOLOGY_NAMES = ['elevation', 'slope', 'aspect', 'profile_curvature', 'planform_curvature', 'twisting_curvature']
PARAMETER_NAMES = ['GS', 'gd', 'c', 'cr', 'phi', 'kt', 'beta', 'A', 'lambda', 'n', 'E', 'ni']
TIME_SENSITIVE_NAMES = ['nearest_rain_station']

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
        abg_shapes.append(row['longitude'].shape)

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
        selectd_parameters: list[str],
        shapes: list[tuple[int, int]],
        parameters_csv_paths: str | list[str], 
        clases_column: str='class_id',
        out_type: str='float32',
        no_data: float | int | str=0
    ) -> dict[str, list[np.ndarray]]:
    """Helper function to get parameters grids"""
    if not all([x in PARAMETER_NAMES for x in selectd_parameters]):
        raise ValueError(f"Please select one or more of the following parameters: {PARAMETER_NAMES}")
    
    par_grids = {par: [] for par in selectd_parameters}
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

def get_time_sensitive_dict(
        env: AnalysisEnvironment,
        sel_ts_opts: list[str]
    ) -> dict[str, pd.DataFrame]:
    """Helper function to get time sensitive options"""
    if not all([x in TIME_SENSITIVE_NAMES for x in sel_ts_opts]):
        raise ValueError(f"Please select one or more of the following time sensitive options: {TIME_SENSITIVE_NAMES}")
    
    ts_dict = {ts: [] for ts in sel_ts_opts}
    for ts in sel_ts_opts:
        if ts == 'nearest_rain_station':
            sta_df = env.load_variable(variable_filename='rain_recordings_vars.pkl')['stations']
            ts_dict[ts] = sta_df
        else:
            raise ValueError(f"{ts} not recognized as a valid time sensitive option during extracion of sations.")
    
    return ts_dict

def convert_abg_and_ref_points_to_prj(
        abg_df: pd.DataFrame,
        ref_points_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Helper function to convert reference points to projected
    
    Args:
        abg_df (pd.DataFrame): ABG dataframe
        ref_points_df (pd.DataFrame): Reference points dataframe

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, int): ABG dataframe, reference points dataframe, projected epsg code
    """
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

    return abg_prj_df, ref_points_prj_df, prj_epsg

def convert_time_sens_dict_to_prj(
    out_epsg: int,
    ts_dict: dict[str, pd.DataFrame]=None
    ) -> dict[str, pd.DataFrame]:
    """
    Helper function to convert time sensitive dict to projected
    
    Args:
        out_epsg (int): Projected epsg code
        ts_dict (dict[str, pd.DataFrame], optional): Time sensitive dictionary (each key is a time sensitive option to collect, like 'nearest_rain_station', which contains the stations_df table) (defaults to None).
    
    Returns:
        dict[str, pd.DataFrame]: Time sensitive dictionary with projected coordinates
    """
    ts_prj_dict = None
    if ts_dict:
        ts_prj_dict = ts_dict.copy()
        for key, ts_df in ts_prj_dict.items():
            ts_df = ts_df.rename(columns={'longitude': 'prj_x', 'latitude': 'prj_y'})
            ts_prj_x, ts_prj_y = convert_coords(
                crs_in=4326,
                crs_out=out_epsg,
                in_coords_x=ts_dict[key]['longitude'],
                in_coords_y=ts_dict[key]['latitude']
            )
            ts_df['prj_x'] = ts_prj_x
            ts_df['prj_y'] = ts_prj_y
            ts_prj_dict[key] = ts_df
    
    return ts_prj_dict

def get_closest_dtm_point(
        x: float,
        y: float,
        base_grid_df: pd.DataFrame,
        base_grid_x_col: str='prj_x',
        base_grid_y_col: str='prj_y'
    ) -> tuple[int, int, float]:
    """Helper function to get closest DTM point"""
    point_idxs, points_dists = np.zeros(base_grid_df.shape[0]), np.zeros(base_grid_df.shape[0])
    for n, (_, row_abg) in enumerate(base_grid_df.iterrows()): # _ because I don't care about the actual index (different if reordered based on priority), but just the current order of rows
        curr_idx, curr_dst = \
            get_closest_1d_pixel_idx(x=x, y=y, x_grid=row_abg[base_grid_x_col], y_grid=row_abg[base_grid_y_col])
        
        if curr_idx.size == 1 and curr_dst.size == 1:
            point_idxs[n], points_dists[n] = curr_idx.item(), curr_dst.item()
        else:
            raise ValueError(f"Not unique match found for point: [x={x}; y={y}] in DTM n. {n}")
    
    if np.isnan(points_dists).all():
        nearest_dtm, nearest_1d_idx, dist_to_grid_pixel = np.nan, np.nan, np.nan
    else:
        nearest_dtm = int(np.nanargmin(points_dists))
        nearest_1d_idx = int(point_idxs[nearest_dtm])
        dist_to_grid_pixel = points_dists[nearest_dtm]

    return nearest_dtm, nearest_1d_idx, dist_to_grid_pixel

def update_reference_points_csv(
        abg_df: pd.DataFrame,
        ref_points_csv_path: str,
        morph_grids: dict[str, list[np.ndarray]]=None,
        par_grids: dict[str, list[np.ndarray]]=None,
        ts_dict: dict[str, pd.DataFrame]=None
    ) -> pd.DataFrame:
    """Helper function to update reference points csv"""
    ref_points_df = pd.read_csv(ref_points_csv_path)
    if ref_points_df.empty:
        raise ValueError(f"Reference points CSV ({ref_points_csv_path}) is empty. Please check the file.")
    
    abg_prj_df, ref_points_prj_df, prj_epsg_code = convert_abg_and_ref_points_to_prj(
        abg_df=abg_df, 
        ref_points_df=ref_points_df
    )
    
    ts_prj_dict = convert_time_sens_dict_to_prj(
        out_epsg=prj_epsg_code, 
        ts_dict=ts_dict
    )

    for i, row_ref_prj_pnt in ref_points_prj_df.iterrows():
        curr_sel_dtm, curr_sel_1d_idx, curr_dist_to_grid_point = get_closest_dtm_point(
            x=row_ref_prj_pnt['prj_x'], 
            y=row_ref_prj_pnt['prj_y'], 
            base_grid_df=abg_prj_df, 
            base_grid_x_col='prj_x', 
            base_grid_y_col='prj_y'
        )
        
        if np.isnan(curr_sel_1d_idx):
            warnings.warn(f"No match found for point {i} (lon={ref_points_df.loc[i,'lon']}, lat={ref_points_df.loc[i,'lat']}). Nearest DTM is {curr_sel_dtm}.", stacklevel=2)
            continue

        ref_points_df.loc[i, 'dtm'] = curr_sel_dtm
        ref_points_df.loc[i, 'idx_1d'] = curr_sel_1d_idx
        ref_points_df.loc[i, 'dist'] = curr_dist_to_grid_point

        if morph_grids is not None:
            for morph in morph_grids.keys():
                ref_points_df.loc[i, morph] = pick_point_from_1d_idx(morph_grids[morph][curr_sel_dtm], curr_sel_1d_idx, order='C')

        if par_grids is not None:
            for par in par_grids.keys():
                ref_points_df.loc[i, par] = pick_point_from_1d_idx(par_grids[par][curr_sel_dtm], curr_sel_1d_idx, order='C')
        
        if ts_prj_dict is not None:
            for ts_par in ts_prj_dict.keys():
                nearest_sta_idx, nearest_sta_dist = get_closest_point_id(
                    x=row_ref_prj_pnt['prj_x'], 
                    y=row_ref_prj_pnt['prj_y'], 
                    x_ref=ts_prj_dict[ts_par]['prj_x'], 
                    y_ref=ts_prj_dict[ts_par]['prj_y']
                )

                ref_points_df.loc[i, ts_par] = ts_prj_dict[ts_par].loc[nearest_sta_idx.item(), 'station']
                ref_points_df.loc[i, f"dist_{ts_par}"] = nearest_sta_dist

    ref_points_df.to_csv(ref_points_csv_path, index=False)

    return ref_points_df

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        ref_points_csv_path: str=None,
        morphology: list[str]=MORPHOLOGY_NAMES,
        parameters: list[str]=PARAMETER_NAMES,
        time_sensitive: list[str]=TIME_SENSITIVE_NAMES,
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
                src_ext=SUPPORTED_FILE_TYPES['table'],
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
            selectd_parameters=parameters,
            shapes=abg_shapes,
            parameters_csv_paths=parameters_csv_path,
            clases_column='class_id',
            out_type=out_type,
            no_data=no_parameter_data
        )

    if time_sensitive is None:
        ts_dict = None
    else:
        ts_dict = get_time_sensitive_dict(
            env=env,
            sel_ts_opts=time_sensitive
        )
    
    logger.info("Updating reference points csv...")
    ref_points_df = update_reference_points_csv(
        abg_df=abg_df,
        ref_points_csv_path=ref_points_csv_path,
        morph_grids=morph_grids,
        par_grids=par_grids,
        ts_dict=ts_dict
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
    parser.add_argument('--time_sensitive', type=list, default=TIME_SENSITIVE_NAMES, help="List of time-sensitive parameters to be processed.")
    parser.add_argument('--out_type', type=str, default='float32', help="Output type for the parameters.")
    parser.add_argument('--no_parameter_data', type=float, default=0, help="No data value for the parameters.")
    args = parser.parse_args()

    ref_points_df = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode,
        ref_points_csv_path=args.ref_points_csv_path,
        morphology=args.morphology,
        parameters=args.parameters,
        time_sensitive=args.time_sensitive,
        out_type=args.out_type,
        no_parameter_data=args.no_parameter_data
    )

# %%