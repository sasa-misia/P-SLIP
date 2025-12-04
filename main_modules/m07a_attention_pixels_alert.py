# %% === Import necessary modules
import argparse
import numpy as np
import pandas as pd

# Importing necessary modules from main_modules
from m00a_env_init import get_or_create_analysis_environment, setup_logger, log_and_warning, log_and_error, memory_report, obtain_config_idx_and_rel_filename
logger = setup_logger(__name__)
logger.info("=== Attention pixels analysis and alert ===")

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    REFERENCE_POINTS_FILENAME,
    SUPPORTED_FILE_TYPES,
    DYNAMIC_SUBFOLDERS
)

# # Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

from psliptools.utilities import (
    select_from_list_prompt,
    select_file_prompt,
    read_generic_csv
)

# from psliptools.geometries import (
# )

from psliptools.scattered import (
    get_closest_point_id
)

# %% === Helper functions
LANDSLIDE_PATHS_FILENAME = "landslide_paths_vars.pkl"
POSSIBLE_TRIGGER_MODES = ['rainfall-threshold', 'safety-factor', 'machine-learning']
STRAIGHT_LABEL = 'straight_'
MA_LABEL = 'mobile_average_'

def get_top_k_paths(
        paths_df: pd.DataFrame,
        dtm: int,
        point_2d: tuple[int, int] | list[int] | np.ndarray,
        k: int = 3
    ) -> list[str]:
    """
    Return top-k path_ids (highest path_realism_score) passing through point_2d in given dtm.
    
    Args:
        paths_df (pd.DataFrame): DataFrame containing landslide paths data.
        dtm (int): DTM value.
        point_2d (tuple[int, int] | list[int] | np.ndarray): 2D index of the point.
        k (int, optional): Number of top paths to return (default is 3).

    Returns:
        list[str]: List of top-k path_ids.
    """
    if not isinstance(paths_df, pd.DataFrame):
        log_and_error("paths_df must be a pandas DataFrame.", ValueError, logger)
    if not isinstance(dtm, int):
        log_and_error("dtm must be an integer.", ValueError, logger)
    if not isinstance(point_2d, (tuple, list, np.ndarray)):
        log_and_error("point_2d must be a tuple, list, or numpy array.", ValueError, logger)
    if k <= 0:
        log_and_error("k must be positive.", ValueError, logger)

    point_2d = np.atleast_1d(point_2d)
    if point_2d.size != 2:
        log_and_error("point_2d must be a tuple, list, or numpy array of length 2.", ValueError, logger)
    
    curr_df = paths_df[paths_df['path_dtm'] == dtm]

    candidates = []
    for _, row in curr_df.iterrows():
        if (row['path_2D_idx'] == point_2d).all(axis=1).any():
            candidates.append((row['path_realism_score'], row['path_id']))
    candidates.sort(reverse=True, key=lambda x: x[0])

    top_paths = [pid for _, pid in candidates[:k]]

    return top_paths + [None] * (k - len(top_paths))

def get_attention_pixel_coordinates(
        env: AnalysisEnvironment,
        attention_pixels_df: pd.DataFrame
    ) -> list[np.ndarray]:
    """
    Get coordinates of attention pixels.
    
    Args:
        env (AnalysisEnvironment): Analysis environment.
        attention_pixels_df (pd.DataFrame): DataFrame containing attention pixel data.

    Returns:
        np.ndarray: List of coordinates of attention pixels (each element contains an array of coordinates nx2: n points with longitude and latitude).
    """
    abg_df = env.load_variable(variable_filename='dtm_vars.pkl')['abg']

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
    
    del abg_df

    return attention_coords

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

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        trigger_mode: str='rainfall-threshold', # or 'safety-factor' or 'machine-learning'
        alert_threshold: float=None # value and measure unit depending on the trigger mode
    ) -> dict[str, object]:
    """Main function to analyze and alert on attention pixels."""

    # Input validation
    if trigger_mode not in POSSIBLE_TRIGGER_MODES:
        log_and_error(f"Invalid trigger_mode: {trigger_mode}. Must be one of {POSSIBLE_TRIGGER_MODES}.", ValueError, logger)

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    landslide_paths_vars = env.load_variable(variable_filename=LANDSLIDE_PATHS_FILENAME)

    landslide_paths_df = landslide_paths_vars['paths_df']

    initial_dtms = np.sort(landslide_paths_df['path_dtm'].unique())
    dtm_unique_points: list[np.ndarray] = []
    for curr_dtm in initial_dtms:
        curr_dtm_paths_df = landslide_paths_df[landslide_paths_df['path_dtm'] == curr_dtm]
        
        all_point_arrays = np.vstack(curr_dtm_paths_df['path_2D_idx'].values)  # (N, 2)
        unique_pts = np.unique(all_point_arrays, axis=0)
        dtm_unique_points.append(unique_pts)
    
    attention_pixels_df = pd.DataFrame({
        'dtm': initial_dtms,
        '2D_idx': dtm_unique_points,
        'alert_date': None
    })

    if trigger_mode == 'rainfall-threshold':
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
        
        rain_vars = env.load_variable(variable_filename=f"{rel_filename}_vars.pkl")

        data_label = 'data'
        rain_metrics = list(rain_vars[data_label].keys())
        rain_modes = [f'{STRAIGHT_LABEL}{data_label}']
        if 'mobile_averages' in rain_vars:
            ma_types = [key for key, value in rain_vars['mobile_averages'].items() if isinstance(value, dict) and key != 'count' and set(value.keys()).issubset(set(rain_metrics))]
            rain_modes += [f'{MA_LABEL}{lbl}' for lbl in ma_types]
        
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
        
        if alert_metric_mode.startswith(STRAIGHT_LABEL):
            alert_metric_data = rain_vars[data_label][alert_metric]
        elif alert_metric_mode.startswith(MA_LABEL):
            ma_type = alert_metric_mode[len(MA_LABEL):]
            alert_metric_data = rain_vars['mobile_averages'][ma_type][alert_metric]
        else:
            log_and_error(f"Invalid alert metric mode: {alert_metric_mode}", ValueError, logger)
        
        alert_metric_max = alert_metric_data.max(axis=None)

        if alert_metric_max is None:
            log_and_error("Alert metric data is empty.", ValueError, logger)
        
        if alert_threshold is None:
            alert_threshold = 0.8 * alert_metric_max
            info_alert_threshold_prompt = f'Info for alert threshold: \n\t- max value of {alert_metric} is {alert_metric_max}; \n\t- default value for alert threshold is {alert_threshold};'
            if alert_metric_mode.startswith(MA_LABEL):
                info_alert_threshold_prompt += f'\n\t- m.a. delta time is {rain_vars["mobile_averages"]["window_delta_time"]};'
            
            if gui_mode:
                log_and_error("GUI mode not implemented yet.", NotImplementedError, logger)
            else:
                print(info_alert_threshold_prompt)
                alert_threshold_input = input(f"Enter the alert threshold for {alert_metric}: ")
                if alert_threshold_input != '':
                    try:
                        alert_threshold = float(alert_threshold_input)
                    except ValueError:
                        log_and_warning(f"Invalid alert threshold: [{alert_threshold_input}]. Falling back to default value [{alert_threshold}].", ValueError, logger)
        
        alert_mask = alert_metric_data >= alert_threshold

        attention_coords = get_attention_pixel_coordinates(env, attention_pixels_df)
        stations_df = rain_vars['stations']
        rain_station_ids = get_rain_station_ids(attention_coords, stations_df)
        rain_station_names = [stations_df.loc[curr_id_array, 'station'].tolist() for curr_id_array in rain_station_ids]
        # TODO: Continue with extraction of stations from alert_mask and extraction of dates and pixels (consider to apply alert threshold for each individual station extracted)
    
    elif trigger_mode == 'safety-factor':
        log_and_error(f"Trigger mode {trigger_mode} is not implemented yet.", NotImplementedError, logger)

    elif trigger_mode == 'machine-learning':
        log_and_error(f"Trigger mode {trigger_mode} is not implemented yet.", NotImplementedError, logger)
    
    else:
        log_and_error(f"Trigger mode not recognized or not implemented: {trigger_mode}.", ValueError, logger)
    
    # TODO: Implement the rest of the code
    out_vars = {}

    return out_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and alert on attention pixels.")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    
    args = parser.parse_args()
    
    out_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode
    )