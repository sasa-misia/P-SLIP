# %% === Import necessary modules
import argparse
import os
import numpy as np
import pandas as pd
import datetime as dt
from collections import defaultdict

# Importing necessary modules from main_modules
from m00a_env_init import get_or_create_analysis_environment, setup_logger, log_and_error, obtain_config_idx_and_rel_filename
from m05a_reference_points_info import get_parameters_csv_paths, get_parameters_grids
logger = setup_logger(__name__)
logger.info("=== Attention pixels analysis and alert ===")

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    SUPPORTED_FILE_TYPES,
    DYNAMIC_SUBFOLDERS
)

# Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

from psliptools.utilities import (
    select_from_list_prompt,
    select_file_prompt,
    read_generic_csv,
    delta_to_string
)

# from psliptools.geometries import (
# )

from psliptools.scattered import (
    get_closest_point_id,
    interpolate_scatter_to_scatter
)

from psliptools.models import (
    run_slip_model
)

# %% === Helper functions
LANDSLIDE_PATHS_FILENAME = "landslide_paths_vars.pkl"
POSSIBLE_TRIGGER_MODES = ['rainfall-threshold', 'safety-factor', 'machine-learning']
POSSIBLE_SAFETY_FACTOR_MODELS = ['slip'] # TODO: Add more
STRAIGHT_LABEL = 'straight_'
MA_LABEL = 'mobile_average_'
DEFAULT_THRESHOLD_PERC = {
    'high-quantile': 0.98,
    'low-quantile': 0.02,
    'percentage-of-max': 0.75,
    'percentage-of-min': 1.05
}
DEFAULT_ALERT_THR_FILE = {
    'rainfall-threshold': 'rainfall_alert_thresholds.csv', 
    'safety-factor': 'safety_factor_alert_thresholds.csv', 
    'machine-learning': 'machine_learning_alert_thresholds.csv'
}
REMOVE_OUTLIERS_DURING_THRESHOLDS_DEFINITION = True

if any([tm not in DEFAULT_ALERT_THR_FILE for tm in POSSIBLE_TRIGGER_MODES]):
    log_and_error(f"Missing default alert threshold file for some trigger modes: [{', '.join([tm for tm in POSSIBLE_TRIGGER_MODES if tm not in DEFAULT_ALERT_THR_FILE])}].", ValueError, logger)

def get_top_k_paths(
        paths_df: pd.DataFrame,
        dtm: int,
        idx_2d: tuple[int, int] | list[tuple[int, int]] | np.ndarray,
        k: int = 3,
        separate_starting_points: bool = False,
        top_k_paths_similarity_tolerance = 1
    ) -> list[str]:
    """
    Return top-k path_ids (highest path_realism_score) passing through point_2d in given dtm.
    
    Args:
        paths_df (pd.DataFrame): DataFrame containing landslide paths data.
        dtm (int): DTM value.
        idx_2d (tuple[int, int] | list[tuple[int, int]] | np.ndarray): 2D indices of the points.
        k (int, optional): Number of top paths to return (default is 3).
        separate_starting_points (bool, optional): If True, return top-k paths for each found starting point (default is False).
        top_k_paths_similarity_tolerance (float, optional): Tolerance for path similarity (0 to 1). 0 means no common points allowed (except starting point), 1 means identical paths allowed (default is 1).

    Returns:
        list[str]: List of top-k path_ids.
    """
    if not isinstance(paths_df, pd.DataFrame):
        log_and_error("paths_df must be a pandas DataFrame.", ValueError, logger)
    if not isinstance(dtm, int):
        log_and_error("dtm must be an integer.", ValueError, logger)
    if not isinstance(idx_2d, (tuple, list, np.ndarray)):
        log_and_error("point_2d must be a tuple, list, or numpy array.", ValueError, logger)
    if k <= 0:
        log_and_error("k must be positive.", ValueError, logger)
    if not (0 <= top_k_paths_similarity_tolerance <= 1):
        log_and_error("top_k_paths_similarity_tolerance must be between 0 and 1.", ValueError, logger)

    idx_2d = np.atleast_1d(idx_2d)
    if idx_2d.shape[1] != 2:
        log_and_error("point_2d must be a tuple, list, or numpy array of shape n x 2.", ValueError, logger)
    
    curr_df = paths_df[paths_df['path_dtm'] == dtm].copy()

    # Precompute set of points for fast lookup
    idx_2d_set = set(map(tuple, idx_2d))

    # Vectorized check for intersection
    curr_df['is_candidate'] = curr_df['path_2D_idx'].apply(lambda x: bool(set(map(tuple, x)) & idx_2d_set))
    candidate_df = curr_df[curr_df['is_candidate']]
    candidates = candidate_df[['path_realism_score', 'path_id', 'starting_point_id', 'path_2D_idx']].values.tolist()

    def calculate_similarity(path1_idx, path2_idx):
        """Calculate similarity between two paths (excluding first point - starting point)."""
        set1 = set(map(tuple, path1_idx[1:]))  # Exclude starting point
        set2 = set(map(tuple, path2_idx[1:]))  # Exclude starting point
        
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        intersection = len(set1 & set2)
        min_length = min(len(set1), len(set2))
        
        return intersection / min_length if min_length > 0 else 0.0

    def filter_by_similarity(candidates_list, k, tolerance):
        """Filter candidates to ensure similarity is below tolerance."""
        selected = []
        
        for score, pid, spid, path_idx in candidates_list:
            # Check similarity with already selected paths
            is_valid = True
            for _, _, _, selected_path_idx in selected:
                similarity = calculate_similarity(path_idx, selected_path_idx)
                if similarity > tolerance:
                    is_valid = False
                    break
            
            if is_valid:
                selected.append((score, pid, spid, path_idx))
                if len(selected) >= k:
                    break
        
        return selected

    if separate_starting_points:
        candidates_by_sp = defaultdict(list)
        for score, pid, spid, path_idx in candidates:
            candidates_by_sp[spid].append((score, pid, spid, path_idx))
        top_paths = []
        for spid, cands in candidates_by_sp.items():
            cands.sort(reverse=True, key=lambda x: x[0])
            filtered = filter_by_similarity(cands, k, top_k_paths_similarity_tolerance)
            top_for_sp = [pid for _, pid, spid, _ in filtered]
            top_paths.extend(top_for_sp + [None] * (k - len(top_for_sp)))
    else:
        candidates.sort(reverse=True, key=lambda x: x[0])
        filtered = filter_by_similarity(candidates, k, top_k_paths_similarity_tolerance)
        top_paths = [pid for _, pid, spid, _ in filtered]
        top_paths += [None] * (k - len(top_paths))

    return top_paths

def get_attention_pixel_coordinates(
        abg_df: pd.DataFrame,
        attention_pixels_df: pd.DataFrame
    ) -> list[np.ndarray]:
    """
    Get coordinates of attention pixels.
    
    Args:
        abg_df (pd.DataFrame): DataFrame containing Analysys Base Grid (ABG) data.
        attention_pixels_df (pd.DataFrame): DataFrame containing attention pixel data.

    Returns:
        np.ndarray: List of coordinates of attention pixels (each element contains an array of coordinates nx2: n points with longitude and latitude).
    """
    attention_coords = []
    for _, row in attention_pixels_df.iterrows():
        curr_dtm = row['dtm']
        curr_2D_idx = row['2D_idx']
        curr_row_idx = curr_2D_idx[:,0]
        curr_col_idx = curr_2D_idx[:,1]
        attention_coords.append(
            np.column_stack([
                abg_df['longitude'][curr_dtm][curr_row_idx, curr_col_idx], 
                abg_df['latitude'][curr_dtm][curr_row_idx, curr_col_idx]
            ])
        )

    return attention_coords

def get_rain_info(
        env: AnalysisEnvironment,
        gui_mode: bool
    ) -> tuple[str, str, str, str]:
    """
    Get information about the rain source.
    
    Args:
        env (AnalysisEnvironment): Analysis environment.
        gui_mode (bool): Whether to run in GUI mode.

    Returns:
        tuple[str, str, str, str]: Tuple containing relative filename of file with rain data, source type, source subtype, and source mode.
    """
    if gui_mode:
        log_and_error("GUI mode not implemented yet.", NotImplementedError, logger)
    else:
        source_type = 'rain'
        source_subtype = select_from_list_prompt(
            obj_list=DYNAMIC_SUBFOLDERS,
            usr_prompt='Select the source subtype: ',
            allow_multiple=False
        )[0]
    
    env, idx_config, rel_filename = obtain_config_idx_and_rel_filename(env, source_type, source_subtype)

    source_mode = env.config['inputs'][source_type][idx_config]['settings']['source_mode']
    if not source_mode == 'station':
        log_and_error("Invalid source mode. Must be 'station'", ValueError, logger)
    
    return rel_filename, source_type, source_subtype, source_mode

def get_rain_station_ids(
        attention_coords: list[np.ndarray],
        stations_df: pd.DataFrame
    ) -> list[list[str]]:
    """
    Get the IDs of the nearest rain stations for each attention pixel.
    
    Args:
        attention_coords (list[np.ndarray]): List of coordinates of attention pixels (each element contains an array of coordinates nx2: n points with longitude and latitude).
        stations_df (pd.DataFrame): DataFrame containing rain station data.

    Returns:
        list[list[str]]: List of lists of rain station IDs (one list per dtm and one string per attention pixel).
    """
    rain_station_ids = []
    for curr_coords in attention_coords:
        curr_station_ids, _ = get_closest_point_id(
            x=curr_coords[:,0],
            y=curr_coords[:,1],
            x_ref=stations_df['longitude'],
            y_ref=stations_df['latitude']
        )
        rain_station_ids.append(curr_station_ids)

    return rain_station_ids

def get_station_pixels_association(
        attention_pixels_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
    """
    Get the attention pixels associated with each rain station.
    
    Args:
        attention_pixels_df (pd.DataFrame): DataFrame containing attention pixel data and their reference stations.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping rain station names to attention pixels (one DataFrame per rain station).
    """
    unique_rain_station_names = sorted(list(set.union(*[set(x) for x in attention_pixels_df['stations']])))

    station_pixels_association = {}
    for curr_sta in unique_rain_station_names:
        filtered_rows = []
        for _, row in attention_pixels_df.iterrows():
            if curr_sta in row['stations']:
                # Find indices where stations match curr_sta
                matching_indices = [i for i, sta in enumerate(row['stations']) if sta == curr_sta]
                # Filter 2D_idx and coordinates
                filtered_2d_idx = row['2D_idx'][matching_indices]
                filtered_coords = row['coordinates'][matching_indices]
                filtered_rows.append({
                    'dtm': row['dtm'],
                    '2D_idx': filtered_2d_idx,
                    'coordinates': filtered_coords
                })
        station_pixels_association[curr_sta] = pd.DataFrame(filtered_rows)
    
    return station_pixels_association

def evaluate_safety_factors_on_attention_pixels(
        env: AnalysisEnvironment,
        attention_pixels_df: pd.DataFrame,
        parameter_class_association_df: pd.DataFrame,
        rain_vars: dict[str, pd.DataFrame],
        model_name: str
    ) -> dict[str, pd.DataFrame]:
    """
    Evaluate safety factors on attention pixels.
    
    Args:
        env (AnalysisEnvironment): Analysis environment.
        attention_pixels_df (pd.DataFrame): DataFrame containing attention pixel data.
        parameter_class_association_df (pd.DataFrame): DataFrame containing parameter class association data.
        rain_vars (dict[str, pd.DataFrame]): Dictionary containing rain data.
        model_name (str): Name of the model (e. g., "slip")

    Returns:
        dict: a dictionary containing multiple DataFrames based on the number of rows in attention_pixels_df. Inside each key, the dataframe has the history of FS.
    """
    if model_name not in POSSIBLE_SAFETY_FACTOR_MODELS:
        log_and_error(f"Invalid model_name: [{model_name}]. Must be one of {POSSIBLE_SAFETY_FACTOR_MODELS}.", ValueError, logger)
    
    slope_df = env.load_variable(variable_filename='morphology_vars.pkl')['angles_df'].loc[:, ['slope', 'no_data']]

    # Extract base grids shapes
    base_grids_shapes = []
    for _, row in slope_df.iterrows():
        base_grids_shapes.append(row['slope'].shape)
    
    # Extract parameter csv path(s) to use
    par_csv_paths = get_parameters_csv_paths(
        env=env, 
        association_df=parameter_class_association_df
    )

    # Apply filter to slope arrays: set values < 0 to 0, and no_data to 0
    slope_df['slope'] = slope_df.apply(lambda row: np.where((row['slope'] < 0) | (row['slope'] == row['no_data']), 0, row['slope']), axis=1)

    # Calculate cosine of slope angles (convert degrees to radians), because is the beta without vegetation
    beta_slope = slope_df['slope'].apply(lambda x: np.cos(np.radians(x))).to_list()

    # Add data to attention_pixels_df
    attention_pixels_df['slope'] = None
    for ap_idx, ap_row in attention_pixels_df.iterrows():
        ap_slope = slope_df['slope'].loc[ap_row['dtm']][ap_row['2D_idx'][:,0], ap_row['2D_idx'][:,1]]
        attention_pixels_df.at[ap_idx, 'slope'] = ap_slope
    
    del slope_df
    
    if model_name == 'slip':
        required_parameters = ['GS', 'c', 'cr', 'phi', 'kt', 'beta', 'A', 'n']

        par_grids_dict = get_parameters_grids(
            association_df=parameter_class_association_df,
            selectd_parameters=required_parameters,
            shapes=base_grids_shapes,
            parameters_csv_paths=par_csv_paths,
            class_column='class_id',
            out_type='float16',
            no_data=[2.7, 5, 0, 20, 0.001, 1, 80, 0.3]
        )

        par_grids_dict['beta'] = [x * y for x, y in zip(par_grids_dict['beta'], beta_slope)]

        del beta_slope

        for par in required_parameters:
            attention_pixels_df[par] = None
            for ap_idx, ap_row in attention_pixels_df.iterrows():
                ap_par = par_grids_dict[par][ap_row['dtm']][ap_row['2D_idx'][:,0], ap_row['2D_idx'][:,1]]
                attention_pixels_df.at[ap_idx, par] = ap_par
        
        del par_grids_dict

        fs_history_dict = {}
        for ap_idx, ap_row in attention_pixels_df.iterrows():
            if ap_idx % 10 == 0:
                logger.info(f"Factor of safety calculations for attention area {ap_idx + 1} of {attention_pixels_df.shape[0]}...")

            interp_rain_history = []
            for r_idx, r_row in rain_vars['data']['cumulative_rain'].iterrows():
                curr_interp_rain = interpolate_scatter_to_scatter(
                    x_in=rain_vars['stations']['longitude'],
                    y_in=rain_vars['stations']['latitude'],
                    data_in=r_row,
                    x_out=ap_row['coordinates'][:,0],
                    y_out=ap_row['coordinates'][:,1],
                    interpolation_method='nearest', 
                    fill_value=0,
                    exclude_nans=True
                )

                interp_rain_history.append(curr_interp_rain)
            
            curr_rain_data_df = rain_vars['datetimes'].copy()
            curr_rain_data_df['rain'] = interp_rain_history
            
            curr_fs_history_df = run_slip_model(
                slope=ap_row['slope'],
                soil_specific_gravity=ap_row['GS'],
                soil_cohesion=ap_row['c'],
                root_cohesion=ap_row['cr'],
                soil_friction=ap_row['phi'],
                soil_drainage=ap_row['kt'],
                infiltration_coefficient=ap_row['beta'],
                A_slip=ap_row['A'],
                soil_porosity=ap_row['n'],
                lambda_slip=0.4,
                alpha_slip=3.4,
                instability_depth=1.2,
                soil_saturation=0.8,
                rain_history=curr_rain_data_df,
                predisposing_time_window=pd.Timedelta(days=30)
            )

            fs_history_dict[ap_idx] = curr_fs_history_df # Index is the same as attention_pixels_df index

    else:
        log_and_error(f"Model [{model_name}] not implemented yet. Please contact the developers.", NotImplementedError, logger)
    
    return fs_history_dict

def get_alert_datetimes_per_attention_area(
        alert_metric_history_dict: dict[str, pd.DataFrame],
        alert_metric_col: str,
        attention_pixels_df: pd.DataFrame,
        alert_thresholds_df: pd.DataFrame,
        trigger_mode: str,
        alert_metric: str,
        alert_metric_mode: str,
        events_time_tolerance: pd.Timedelta,
        landslide_paths_df: pd.DataFrame,
        top_k_paths_per_activation: int,
        top_k_paths_similarity_tolerance: float
    ) -> dict[str, pd.DataFrame]:
    """
    Get datetimes for alerts per attention area.

    Args:
        alert_metric_history_dict (dict[str, pd.DataFrame]): Dictionary with alert metric history per attention area.
        alert_metric_col (str): Alert metric column.
        attention_pixels_df (pd.DataFrame): Attention pixels dataframe.
        alert_thresholds_df (pd.DataFrame): Alert thresholds dataframe.
        trigger_mode (str): Trigger mode.
        alert_metric (str): Alert metric.
        alert_metric_mode (str): Alert metric mode.
        events_time_tolerance (pd.Timedelta): Events time tolerance.
        landslide_paths_df (pd.DataFrame): Landslide paths dataframe.
        top_k_paths_per_activation (int): Top k paths per activation.
        top_k_paths_similarity_tolerance (float): Top k paths similarity tolerance.

    Returns:
        dict[str, pd.DataFrame]: Dictionary with alert datetimes per attention area.
    """
    alert_datetimes_dict = {}
    for aa, data_hist_df in alert_metric_history_dict.items():
        data_stacked_vals = np.vstack(data_hist_df[alert_metric_col])
        if data_stacked_vals.ndim != 2:
            log_and_error(f"Invalid data_stacked_vals shape: {data_stacked_vals.shape}. Please contact the developers.", ValueError, logger)
        
        if trigger_mode in ['rainfall-threshold', 'machine-learning']:
            alert_mask = data_stacked_vals >= alert_thresholds_df.loc[aa, 'threshold']
        elif trigger_mode == 'safety-factor':
            alert_mask = data_stacked_vals <= alert_thresholds_df.loc[aa, 'threshold']
        else:
            log_and_error(f"Invalid trigger_mode during creation of alerts: [{trigger_mode}]. Please contact the developers.", ValueError, logger)
        
        alert_rows = alert_mask.any(axis=1)
        alert_mask_only_activated = alert_mask[alert_rows, :]

        curr_alert_datetimes_df = data_hist_df.loc[alert_rows, ['start_date', 'end_date']].copy().reset_index(drop=True)

        # Group datetimes into events based on events_time_tolerance using 'start_date'
        curr_alert_datetimes_df = curr_alert_datetimes_df.sort_values('start_date')
        event_labels = []
        current_event = 0
        last_date = None
        for cad_idx, cad_row in curr_alert_datetimes_df.iterrows():
            dt = cad_row['start_date']
            if last_date is None or (dt - last_date) > events_time_tolerance:
                current_event += 1
            event_labels.append(f'rt{current_event}')
            last_date = dt
        curr_alert_datetimes_df['event'] = event_labels

        logger.info(f"Number of events detected with tolerance [{events_time_tolerance}]: {len(curr_alert_datetimes_df['event'].unique())} (alert_area {aa} of {len(alert_metric_history_dict)})")

        curr_alert_datetimes_df['trigger_mode'] = trigger_mode
        curr_alert_datetimes_df['alert_metric'] = alert_metric
        curr_alert_datetimes_df['alert_metric_mode'] = alert_metric_mode
        curr_alert_datetimes_df['activated_pixels_dtm_file_id'] = attention_pixels_df.loc[aa, 'dtm_file_id']

        curr_alert_datetimes_df['activated_pixels_2D_idx'] = None
        curr_alert_datetimes_df['activated_pixels_coordinates'] = None
        for act_idx, act_mask in enumerate(alert_mask_only_activated):
            curr_alert_datetimes_df.at[act_idx, 'activated_pixels_2D_idx'] = attention_pixels_df.loc[aa, '2D_idx'][act_mask, :]
            curr_alert_datetimes_df.at[act_idx, 'activated_pixels_coordinates'] = attention_pixels_df.loc[aa, 'coordinates'][act_mask, :]

        # Extract top-k critical landslide paths for each alert (could be slow)
        curr_alert_datetimes_df['top_critical_landslide_path_ids'] = None
        if top_k_paths_per_activation > 0:
            curr_dtm = int(attention_pixels_df.loc[aa, 'dtm'])

            for cad_idx, cad_row in curr_alert_datetimes_df.iterrows():
                if cad_idx % 100 == 0:
                    logger.info(f"Finding top {top_k_paths_per_activation} critical landslide paths for event {cad_idx + 1} of {len(curr_alert_datetimes_df)} (alert_area n.: {aa} of {len(alert_metric_history_dict)})...")

                if cad_idx > 0 and np.array_equal(curr_alert_datetimes_df.at[cad_idx - 1, 'activated_pixels_2D_idx'], cad_row['activated_pixels_2D_idx']):
                    curr_alert_datetimes_df.at[cad_idx, 'top_critical_landslide_path_ids'] = curr_alert_datetimes_df.loc[cad_idx - 1, 'top_critical_landslide_path_ids']
                    continue # Already computed, no need to recompute

                if cad_idx > 0:
                    already_computed = curr_alert_datetimes_df.loc[0:(cad_idx - 1), 'activated_pixels_2D_idx'].apply(lambda arr: np.array_equal(arr, cad_row['activated_pixels_2D_idx']))
                    if already_computed.any():
                        old_idx = already_computed[already_computed].index[-1] # The last row that was equal
                        curr_alert_datetimes_df.at[cad_idx, 'top_critical_landslide_path_ids'] = curr_alert_datetimes_df.loc[old_idx, 'top_critical_landslide_path_ids']
                        continue # Already computed, no need to recompute

                top_k_paths = get_top_k_paths(
                    paths_df=landslide_paths_df,
                    dtm=curr_dtm,
                    idx_2d=attention_pixels_df.loc[aa, '2D_idx'],
                    k=top_k_paths_per_activation,
                    separate_starting_points=True,
                    top_k_paths_similarity_tolerance=top_k_paths_similarity_tolerance
                )

                top_k_paths = [x for x in top_k_paths if x is not None]
                curr_alert_datetimes_df.at[cad_idx, 'top_critical_landslide_path_ids'] = top_k_paths
        
        alert_datetimes_dict[aa] = curr_alert_datetimes_df
    
    return alert_datetimes_dict

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        trigger_mode: str='rainfall-threshold', # or 'safety-factor' or 'machine-learning'
        alert_thresholds: float | list[float] | np.ndarray | pd.Series=None, # values and measure units depend on trigger mode
        events_time_tolerance: dt.timedelta | pd.Timedelta=dt.timedelta(days=5),
        top_k_paths_per_activation: int=5,
        top_k_paths_similarity_tolerance: float=0.5 # 0 (no point in common, except for start) to 1 (all points in common)
    ) -> dict[str, object]:
    """Main function to analyze and alert on attention pixels."""

    # Input validation
    if trigger_mode not in POSSIBLE_TRIGGER_MODES:
        log_and_error(f"Invalid trigger_mode: [{trigger_mode}]. Must be one of {POSSIBLE_TRIGGER_MODES}.", ValueError, logger)
    if not alert_thresholds is None and not isinstance(alert_thresholds, (float, list, np.ndarray, pd.Series)):
        log_and_error(f"Invalid alert_thresholds: [{alert_thresholds}]. Must be a float, list, np.ndarray, or pd.Series.", ValueError, logger)
    if not isinstance(events_time_tolerance, (dt.timedelta, pd.Timedelta)):
        log_and_error(f"Invalid events_time_tolerance: [{events_time_tolerance}]. Must be a dt.timedelta or pd.Timedelta.", ValueError, logger)
    if not isinstance(top_k_paths_per_activation, int):
        log_and_error(f"Invalid top_k_paths_per_activation: [{top_k_paths_per_activation}]. Must be an int.", ValueError, logger)

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    landslide_paths_vars = env.load_variable(variable_filename=LANDSLIDE_PATHS_FILENAME)

    landslide_paths_df = landslide_paths_vars['paths_df']

    dtm_vars = env.load_variable(variable_filename='dtm_vars.pkl')

    dtm_file_id_df = dtm_vars['dtm']['file_id']
    abg_df = dtm_vars['abg']

    del dtm_vars # It can be heavy to keep in memory on very large areas

    # Create a dataframe with every dtm, starting_point_id, and their attention pixels
    unique_combos = landslide_paths_df[['path_dtm', 'starting_point_id']].drop_duplicates().sort_values(['path_dtm', 'starting_point_id'])
    attention_pixels_list = []
    for _, row in unique_combos.iterrows():
        curr_dtm = int(row['path_dtm'])
        curr_dtm_file_id = dtm_file_id_df[curr_dtm]
        curr_spid = row['starting_point_id']
        curr_paths = landslide_paths_df[(landslide_paths_df['path_dtm'] == curr_dtm) & (landslide_paths_df['starting_point_id'] == curr_spid)]
        all_points = np.vstack(curr_paths['path_2D_idx'].values)
        unique_pts = np.unique(all_points, axis=0)
        attention_pixels_list.append({
            'dtm': curr_dtm,
            'dtm_file_id': curr_dtm_file_id,
            'starting_point_id': curr_spid,
            '2D_idx': unique_pts
        })
    
    attention_pixels_df = pd.DataFrame(attention_pixels_list)
    attention_pixels_df.index.name = 'attention_area'
    attention_pixels_df['coordinates'] = get_attention_pixel_coordinates(abg_df, attention_pixels_df)

    del abg_df # It can be heavy to keep in memory on very large areas
    
    # =============== Alerts by rainfall-threshold ===============
    if trigger_mode == 'rainfall-threshold':
        rel_filename, _, _, _ = get_rain_info(env, gui_mode)
        
        rain_vars = env.load_variable(variable_filename=f"{rel_filename}_vars.pkl")
        straight_data_dt = rain_vars['datetimes']['start_date'].diff().mean()
        straight_data_dt_str = delta_to_string(straight_data_dt, digits=[2, 2, 2, 2])

        data_label = 'data'
        rain_metrics = list(rain_vars[data_label].keys())
        rain_modes = [f'{STRAIGHT_LABEL}{data_label}']
        if 'mobile_averages' in rain_vars:
            ma_types = [key for key, value in rain_vars['mobile_averages'].items() if isinstance(value, dict) and key != 'count' and set(value.keys()).issubset(set(rain_metrics))]
            rain_modes += [f'{MA_LABEL}{lbl}' for lbl in ma_types]
            ma_data_dt = rain_vars['mobile_averages']['window_delta_time']
            ma_data_dt_str = delta_to_string(ma_data_dt, digits=[2, 2, 2, 2])
        
        if gui_mode:
            log_and_error("GUI mode not implemented yet.", NotImplementedError, logger)
        else:
            alert_metric = select_from_list_prompt(
                obj_list=rain_metrics,
                usr_prompt='Select the alert metric to use for triggering: ',
                allow_multiple=False
            )[0]
            
            alert_metric_mode = select_from_list_prompt(
                obj_list=rain_modes,
                usr_prompt='Select the way you want to consider the alert metric: ',
                allow_multiple=False
            )[0]
        
        stations_df = rain_vars['stations']
        rain_station_ids = get_rain_station_ids(attention_pixels_df['coordinates'], stations_df)
        attention_pixels_df['stations'] = [stations_df.loc[curr_id_array, 'station'].tolist() for curr_id_array in rain_station_ids]
        station_pixels_association = get_station_pixels_association(attention_pixels_df)
        unique_rain_station_names = list(station_pixels_association.keys())
        
        if alert_metric_mode.startswith(STRAIGHT_LABEL):
            alert_metric_data = rain_vars[data_label][alert_metric].loc[:,unique_rain_station_names]
            alert_metric_dt_str = straight_data_dt_str
        elif alert_metric_mode.startswith(MA_LABEL):
            ma_type = alert_metric_mode[len(MA_LABEL):]
            alert_metric_data = rain_vars['mobile_averages'][ma_type][alert_metric].loc[:,unique_rain_station_names]
            alert_metric_dt_str = ma_data_dt_str
        else:
            log_and_error(f"Invalid alert metric mode: {alert_metric_mode}", ValueError, logger)
        
        alert_metric_col = alert_metric + '_' + alert_metric_mode + '_' + alert_metric_dt_str
        default_thr_mode = 'high-quantile'

        alert_metric_history_dict = {}
        for ap_idx, ap_row in attention_pixels_df.iterrows():
            if ap_idx % 10 == 0:
                logger.info(f"Extraction of data for attention area {ap_idx + 1} of {attention_pixels_df.shape[0]}...")

            curr_rain_history_df = rain_vars['datetimes'].loc[:, ['start_date', 'end_date']]

            stations_to_pick = ap_row['stations']

            curr_rain_history_df[alert_metric_col] = None
            for amd_idx, amd_rec in alert_metric_data.iterrows():
                data_to_vectorize = amd_rec[stations_to_pick].to_numpy()
                curr_rain_history_df.at[amd_idx, alert_metric_col] = data_to_vectorize
            
            alert_metric_history_dict[ap_idx] = curr_rain_history_df

    # =============== Alerts by safety-factor ===============
    elif trigger_mode == 'safety-factor':
        alert_metric = 'safety-factor'
        alert_metric_col = 'fs'
        alert_metric_mode = 'slip' # TODO: Add more safety-factor models
        default_thr_mode = 'low-quantile'

        if alert_metric_mode not in POSSIBLE_SAFETY_FACTOR_MODELS:
            log_and_error(f"Invalid model_name: [{alert_metric_mode}]. Must be one of {POSSIBLE_SAFETY_FACTOR_MODELS}.", ValueError, logger)
        
        parameter_vars = env.load_variable(variable_filename='parameter_vars.pkl')

        rel_filename, _, _, _ = get_rain_info(env, gui_mode)
        rain_vars = env.load_variable(variable_filename=f'{rel_filename}_vars.pkl')

        alert_metric_history_dict = evaluate_safety_factors_on_attention_pixels( # It will have same number of rows as attention_pixels_df
            env=env,
            attention_pixels_df=attention_pixels_df,
            parameter_class_association_df = parameter_vars['association_df'],
            rain_vars=rain_vars,
            model_name=alert_metric_mode
        )

    # =============== Alerts by machine-learning ===============
    elif trigger_mode == 'machine-learning':
        log_and_error(f"Trigger mode {trigger_mode} is not implemented yet.", NotImplementedError, logger)
    
    # =============== Unknown trigger mode ===============
    else:
        log_and_error(f"Trigger mode not recognized or not implemented: {trigger_mode}.", ValueError, logger)

    
    # Statistics of alert metric data
    alert_metric_statistics = []
    for aa, hist_df in alert_metric_history_dict.items():
        # Create mask for outliers using IQR method
        curr_data_stacked = np.vstack(hist_df[alert_metric_col])
        outlier_and_nan_mask = np.zeros(curr_data_stacked.shape, dtype=bool)
        if REMOVE_OUTLIERS_DURING_THRESHOLDS_DEFINITION:
            lower_quantile = np.nanquantile(curr_data_stacked, q=0.01)
            higher_quantile = np.nanquantile(curr_data_stacked, q=0.99)
            iqr = higher_quantile - lower_quantile
            lower_bound = lower_quantile - 1.5 * iqr
            upper_bound = higher_quantile + 1.5 * iqr
            outlier_and_nan_mask = (curr_data_stacked < lower_bound) | (curr_data_stacked > upper_bound)
        
        if default_thr_mode == 'percentage-of-min':
            def_alert_threshold = np.nanmin(curr_data_stacked[~outlier_and_nan_mask]) * DEFAULT_THRESHOLD_PERC[default_thr_mode]
        elif default_thr_mode == 'percentage-of-max':
            def_alert_threshold = np.nanmax(curr_data_stacked[~outlier_and_nan_mask]) * DEFAULT_THRESHOLD_PERC[default_thr_mode]
        elif default_thr_mode in ['high-quantile', 'low-quantile']:
            masked_data = curr_data_stacked[~outlier_and_nan_mask]
            def_alert_threshold = np.nanquantile(masked_data, q=DEFAULT_THRESHOLD_PERC[default_thr_mode])
        else:
            log_and_error(f"Default threshold mode not recognized: {default_thr_mode}. Contact the developer.", ValueError, logger)
        
        alert_metric_statistics.append({
            'attention_area': aa,
            'starting_point_id': attention_pixels_df.at[aa, 'starting_point_id'],
            'metric': alert_metric,
            'metric_mode': alert_metric_mode,
            'mean': np.nanmean(curr_data_stacked),
            'std': np.nanstd(curr_data_stacked),
            'max': np.nanmax(curr_data_stacked),
            'min': np.nanmin(curr_data_stacked),
            'default-threshold': def_alert_threshold,
        })

    alert_metric_statistics_df = pd.DataFrame(alert_metric_statistics).set_index('attention_area')

    # Definition of alert thresholds
    if alert_thresholds is None:
        if gui_mode:
            log_and_error("GUI mode not implemented yet.", NotImplementedError, logger)
        else:
            print(f'Helpful info about alert threshold: \n\n{alert_metric_statistics_df}')

            def_alert_df = alert_metric_statistics_df.copy()
            def_alert_df['threshold'] = def_alert_df['default-threshold']

            def_alert_df_path = os.path.join(env.folders['user_control']['path'], DEFAULT_ALERT_THR_FILE[trigger_mode])
            if os.path.isfile(def_alert_df_path):
                overwrite = input(f"File {def_alert_df_path} already exists. Overwrite with default? [y/n]: ")
                if overwrite.lower() == 'y':
                    def_alert_df.to_csv(def_alert_df_path, index=True)
                else:
                    print(f"File {def_alert_df_path} not overwritten.")
            else:
                def_alert_df.to_csv(def_alert_df_path, index=True)

            alert_thresholds_path = select_file_prompt(
                base_dir=env.folders['user_control']['path'], 
                usr_prompt=f'Select file with alert thresholds (default: {DEFAULT_ALERT_THR_FILE[trigger_mode]}): ', 
                src_ext=SUPPORTED_FILE_TYPES['table'],
                default_file=DEFAULT_ALERT_THR_FILE[trigger_mode]
            )

            alert_thresholds = read_generic_csv(alert_thresholds_path, index_col='attention_area').loc[:, 'threshold']
    else:
        if not isinstance(alert_thresholds, pd.Series):
            alert_thresholds = np.atleast_1d(alert_thresholds)
            if alert_thresholds.size != len(attention_pixels_df) and alert_thresholds.size == 1:
                alert_thresholds = np.repeat(alert_thresholds, len(attention_pixels_df))
            alert_thresholds = pd.Series(alert_thresholds, index=pd.Index(attention_pixels_df.index, name='attention_area'), name='threshold')
    
    if not isinstance(alert_thresholds, pd.Series):
        log_and_error("Alert thresholds, at this point, must be a pandas Series! Please contact the developer.", ValueError, logger)
    alert_thresholds = alert_thresholds.loc[attention_pixels_df.index]
    
    # Convert alert_thresholds to DataFrame with additional columns
    alert_thresholds_df = alert_metric_statistics_df.copy()
    alert_thresholds_df['threshold'] = alert_thresholds
    
    alert_datetimes_dict = get_alert_datetimes_per_attention_area(
        alert_metric_history_dict=alert_metric_history_dict,
        alert_metric_col=alert_metric_col,
        attention_pixels_df=attention_pixels_df,
        alert_thresholds_df=alert_thresholds_df,
        trigger_mode=trigger_mode,
        alert_metric=alert_metric,
        alert_metric_mode=alert_metric_mode,
        events_time_tolerance=events_time_tolerance,
        landslide_paths_df=landslide_paths_df,
        top_k_paths_per_activation=top_k_paths_per_activation,
        top_k_paths_similarity_tolerance=top_k_paths_similarity_tolerance
    )

    # Extract all unique top critical landslide path IDs
    all_top_paths = set()
    for aa, ad_df in alert_datetimes_dict.items():
        for paths in ad_df['top_critical_landslide_path_ids']:
            if paths is not None:
                all_top_paths.update(paths)
    all_top_paths = list(all_top_paths)

    # Extract the corresponding rows from landslide_paths_df
    critical_paths_df = landslide_paths_df[landslide_paths_df['path_id'].isin(all_top_paths)].copy()
    
    OUT_ALERT_DIR = os.path.join(env.folders['outputs']['tables']['path'], 'attention_pixels_alerts')
    if not os.path.isdir(OUT_ALERT_DIR):
        os.makedirs(OUT_ALERT_DIR)

    attention_pixels_df.to_csv(os.path.join(OUT_ALERT_DIR, 'attention_pixels.csv'), index=True)
    alert_thresholds_df.to_csv(os.path.join(OUT_ALERT_DIR, DEFAULT_ALERT_THR_FILE[trigger_mode]), index=True)
    critical_paths_df.to_csv(os.path.join(OUT_ALERT_DIR, 'critical_landslide_paths.csv'), index=False)

    OUT_ALERT_DIR_AD = os.path.join(OUT_ALERT_DIR, 'activation_datetimes')
    if not os.path.isdir(OUT_ALERT_DIR_AD):
        os.makedirs(OUT_ALERT_DIR_AD)
    OUT_ALERT_DIR_MH = os.path.join(OUT_ALERT_DIR, 'activation_metric_history')
    if not os.path.isdir(OUT_ALERT_DIR_MH):
        os.makedirs(OUT_ALERT_DIR_MH)
    for aa in attention_pixels_df.index:
        alert_datetimes_dict[aa].to_csv(os.path.join(OUT_ALERT_DIR_AD, f'activation_datetimes_aa_{aa}.csv'), index=False)
        alert_metric_history_dict[aa].to_csv(os.path.join(OUT_ALERT_DIR_MH, f'activation_metric_history_aa_{aa}.csv'), index=False)

    alert_vars = {
        'attention_pixels_df': attention_pixels_df, 
        'alert_thresholds_df': alert_thresholds_df,
        'alert_datetimes_dict': alert_datetimes_dict,
        'alert_metric_history_dict': alert_metric_history_dict
    }

    env.save_variable(variable_to_save=alert_vars, variable_filename='alert_vars.pkl')

    return alert_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and alert on attention pixels.")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    parser.add_argument("--trigger_mode", type=str, default='rainfall-threshold', help="Trigger mode (rainfall-threshold, safety-factor, machine-learning)")
    parser.add_argument("--alert_thresholds", type=str, default=None, help="Alert thresholds (comma-separated values or path to CSV file)")
    parser.add_argument("--events_time_tolerance", type=str, default='5d', help="Events time tolerance (e.g., '5d' for 5 days, '2h' for 2 hours)")
    parser.add_argument("--top_k_paths_per_activation", type=int, default=5, help="Number of top critical paths to extract per activation")
    parser.add_argument("--top_k_paths_similarity_tolerance", type=float, default=0.5, help="Similarity tolerance for top-k paths (0 to 1)")
    
    args = parser.parse_args()
    
    # Parse events_time_tolerance
    events_time_tolerance = pd.Timedelta(args.events_time_tolerance)
    
    # Parse alert_thresholds if provided
    alert_thresholds = None
    if args.alert_thresholds is not None:
        if os.path.isfile(args.alert_thresholds):
            # Load from CSV file
            alert_thresholds = read_generic_csv(args.alert_thresholds, index_col='attention_area').loc[:, 'threshold']
        else:
            # Parse as comma-separated values
            try:
                alert_thresholds = [float(x.strip()) for x in args.alert_thresholds.split(',')]
            except ValueError:
                log_and_error(f"Invalid alert_thresholds format: {args.alert_thresholds}", ValueError, logger)
    
    alert_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        trigger_mode=args.trigger_mode,
        alert_thresholds=alert_thresholds,
        events_time_tolerance=events_time_tolerance,
        top_k_paths_per_activation=args.top_k_paths_per_activation,
        top_k_paths_similarity_tolerance=args.top_k_paths_similarity_tolerance
    )