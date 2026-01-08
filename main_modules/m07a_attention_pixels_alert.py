# %% === Import necessary modules
import argparse
import os
import numpy as np
import pandas as pd
import datetime as dt

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
    read_generic_csv,
    delta_to_string
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
DEFAULT_THRESHOLD_PERC = {
    'quantiles': 0.9975,
    'max-percentage': 0.75
}
DEFAULT_ALERT_THR_FILE = {
    'rainfall-threshold': 'rainfall_alert_thresholds.csv', 
    'safety-factor': 'safety_factor_alert_thresholds.csv', 
    'machine-learning': 'machine_learning_alert_thresholds.csv'
}
REMOVE_OUTLIERS_IN_THRESHOLDS = True

if any([tm not in DEFAULT_ALERT_THR_FILE for tm in POSSIBLE_TRIGGER_MODES]):
    log_and_error(f"Missing default alert threshold file for some trigger modes: [{', '.join([tm for tm in POSSIBLE_TRIGGER_MODES if tm not in DEFAULT_ALERT_THR_FILE])}].", ValueError, logger)

def get_top_k_paths(
        paths_df: pd.DataFrame,
        dtm: int,
        idx_2d: tuple[int, int] | list[tuple[int, int]] | np.ndarray,
        k: int = 3,
        separate_starting_points: bool = False
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
    if not isinstance(idx_2d, (tuple, list, np.ndarray)):
        log_and_error("point_2d must be a tuple, list, or numpy array.", ValueError, logger)
    if k <= 0:
        log_and_error("k must be positive.", ValueError, logger)

    idx_2d = np.atleast_1d(idx_2d)
    if idx_2d.shape[1] != 2:
        log_and_error("point_2d must be a tuple, list, or numpy array of shape n x 2.", ValueError, logger)
    
    curr_df = paths_df[paths_df['path_dtm'] == dtm]

    candidates = []
    for _, row in curr_df.iterrows():
        path_points = row['path_2D_idx']
        if any(np.any(np.all(path_points == point, axis=1)) for point in idx_2d):
            candidates.append((row['path_realism_score'], row['path_id'], row['starting_point_id']))

    if separate_starting_points:
        from collections import defaultdict
        candidates_by_sp = defaultdict(list)
        for score, pid, spid in candidates:
            candidates_by_sp[spid].append((score, pid))
        top_paths = []
        for spid, cands in candidates_by_sp.items():
            cands.sort(reverse=True, key=lambda x: x[0])
            top_for_sp = [pid for _, pid in cands[:k]]
            top_paths.extend(top_for_sp + [None] * (k - len(top_for_sp)))
    else:
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_paths = [pid for _, pid, _ in candidates[:k]]
        top_paths += [None] * (k - len(top_paths))

    return top_paths

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

def _format_iterable_with_tab(
        iterable: list | np.ndarray | pd.Series | dict,
        prefix: str = "\t"
    ) -> str:
    """
    Convert ``iterable`` (list, np.ndarray, pd.Series, dict) to a string
    where each element/value is on its own line and prefixed with ``prefix``.
    For a pandas Series the index is included (``index: value``) just like
    the default ``print(series)`` output.
    """
    if isinstance(iterable, dict):
        lines = [f"{prefix}{k}: {v}" for k, v in iterable.items()]
    elif isinstance(iterable, pd.Series):
        # Preserve the index in the output
        lines = [f"{prefix}{idx}: {val}" for idx, val in iterable.items()]
    else:
        # pandas Series, list, ndarray, etc.
        # ``list(iterable)`` works for Series, ndarray and list alike
        lines = [f"{prefix}{v}" for v in list(iterable)]
    
    return "\n".join(lines)

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        trigger_mode: str='rainfall-threshold', # or 'safety-factor' or 'machine-learning'
        alert_thresholds: float | list[float] | np.ndarray | pd.Series=None, # values and measure units depend on trigger mode
        default_thr_mode: str='quantiles', # or 'max-percentage'
        events_time_tolerance: dt.timedelta | pd.Timedelta=dt.timedelta(days=5),
        top_k_paths_per_activation: int=5
    ) -> dict[str, object]:
    """Main function to analyze and alert on attention pixels."""

    # Input validation
    if trigger_mode not in POSSIBLE_TRIGGER_MODES:
        log_and_error(f"Invalid trigger_mode: [{trigger_mode}]. Must be one of {POSSIBLE_TRIGGER_MODES}.", ValueError, logger)
    if not alert_thresholds is None and not isinstance(alert_thresholds, (float, list, np.ndarray, pd.Series)):
        log_and_error(f"Invalid alert_thresholds: [{alert_thresholds}]. Must be a float, list, np.ndarray, or pd.Series.", ValueError, logger)
    if not default_thr_mode in DEFAULT_THRESHOLD_PERC.keys():
        log_and_error(f"Invalid default_thresholds: [{default_thr_mode}]. Must be one of {list(DEFAULT_THRESHOLD_PERC.keys())}.", ValueError, logger)
    if not isinstance(events_time_tolerance, (dt.timedelta, pd.Timedelta)):
        log_and_error(f"Invalid events_time_tolerance: [{events_time_tolerance}]. Must be a dt.timedelta or pd.Timedelta.", ValueError, logger)
    if not isinstance(top_k_paths_per_activation, int):
        log_and_error(f"Invalid top_k_paths_per_activation: [{top_k_paths_per_activation}]. Must be an int.", ValueError, logger)

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
        '2D_idx': dtm_unique_points
    })

    attention_pixels_df['coordinates'] = get_attention_pixel_coordinates(env, attention_pixels_df)

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
        
        # Create mask for outliers using IQR method
        outlier_mask = pd.DataFrame(False, index=alert_metric_data.index, columns=alert_metric_data.columns)
        if REMOVE_OUTLIERS_IN_THRESHOLDS:
            qlw_sta = alert_metric_data.quantile(q=0.01, axis=0)
            qhg_sta = alert_metric_data.quantile(q=0.99, axis=0)
            iqr_sta = qhg_sta - qlw_sta
            lower_bound = qlw_sta - 1.5 * iqr_sta
            upper_bound = qhg_sta + 1.5 * iqr_sta
            outlier_mask = (alert_metric_data < lower_bound) | (alert_metric_data > upper_bound)
        
        alert_mtr_sta_max = alert_metric_data.max(axis=0)
        if alert_mtr_sta_max.isna().any():
            log_and_error(f"Some alert metric data is empty: {alert_mtr_sta_max[alert_mtr_sta_max.isna()]}", ValueError, logger)
        
        if default_thr_mode == 'max-percentage':
            def_alert_thresholds = DEFAULT_THRESHOLD_PERC[default_thr_mode] * alert_metric_data.where(~outlier_mask).max(axis=0)
        elif default_thr_mode == 'quantiles':
            def_alert_thresholds = alert_metric_data.where(~outlier_mask).quantile(DEFAULT_THRESHOLD_PERC[default_thr_mode], axis=0)
        else:
            log_and_error(f"Default threshold mode not implemented: {default_thr_mode}. Contact the developer.", ValueError, logger)
        
        if alert_thresholds is None:
            info_alert_threshold_prompt = (
                f'Info for alert threshold: \n'
                f'\t- max values of {alert_metric} are \n{_format_iterable_with_tab(alert_mtr_sta_max, prefix="\t\t")};\n'
                f'\t- default values for alert thresholds are \n{_format_iterable_with_tab(def_alert_thresholds, prefix="\t\t")};'
            )
            if alert_metric_mode.startswith(MA_LABEL):
                info_alert_threshold_prompt += f'\n\t- m.a. delta time is {rain_vars["mobile_averages"]["window_delta_time"]};'
            
            if gui_mode:
                log_and_error("GUI mode not implemented yet.", NotImplementedError, logger)
            else:
                print(info_alert_threshold_prompt)

                def_alert_df = pd.DataFrame({
                    'station': alert_mtr_sta_max.index.tolist(),
                    f'max_{alert_metric}_[{alert_metric_mode}]': alert_mtr_sta_max.tolist(),
                    'threshold': def_alert_thresholds.tolist()
                })

                def_alert_df_path = os.path.join(env.folders['user_control']['path'], DEFAULT_ALERT_THR_FILE[trigger_mode])
                if os.path.isfile(def_alert_df_path):
                    overwrite = input(f"File {def_alert_df_path} already exists. Overwrite with default? [y/n]: ")
                    if overwrite.lower() == 'y':
                        def_alert_df.to_csv(def_alert_df_path, index=False)
                    else:
                        print(f"File {def_alert_df_path} not overwritten.")
                else:
                    def_alert_df.to_csv(def_alert_df_path, index=False)

                alert_thresholds_path = select_file_prompt(
                    base_dir=env.folders['user_control']['path'], 
                    usr_prompt=f'Select file with alert thresholds (default: {DEFAULT_ALERT_THR_FILE[trigger_mode]}): ', 
                    src_ext=SUPPORTED_FILE_TYPES['table'],
                    default_file=DEFAULT_ALERT_THR_FILE[trigger_mode]
                )

                alert_thresholds = read_generic_csv(alert_thresholds_path, index_col='station').loc[:, 'threshold']
        else:
            if not isinstance(alert_thresholds, pd.Series):
                alert_thresholds = pd.Series(alert_thresholds, index=unique_rain_station_names)
        
        if not isinstance(alert_thresholds, pd.Series):
            log_and_error("Alert thresholds, at this point, must be a pandas Series! Please contact the developer.", ValueError, logger)
        alert_thresholds.loc[unique_rain_station_names]
        
        # Convert alert_thresholds to DataFrame with additional columns
        alert_thr_df = pd.DataFrame({
            'alert_metric': alert_metric,
            'alert_metric_mode': alert_metric_mode,
            'alert_metric_delta_time': alert_metric_dt_str,
            'max': alert_mtr_sta_max,
            'threshold': alert_thresholds
        })
        alert_thr_df.index.name = 'station'
        
        alert_mask = alert_metric_data >= alert_thr_df['threshold']

        alert_datetimes_df = rain_vars['datetimes'].loc[alert_mask.any(axis=1)].copy()
        alert_datetimes_df['trigger_mode'] = trigger_mode
        alert_datetimes_df['alert_metric'] = alert_metric
        alert_datetimes_df['alert_metric_mode'] = alert_metric_mode
        alert_datetimes_df['alert_metric_delta_time'] = alert_metric_dt_str
        
        # Group datetimes into events based on events_time_tolerance using 'start_date'
        alert_datetimes_df = alert_datetimes_df.sort_values('start_date')
        event_labels = []
        current_event = 0
        last_date = None
        for idx, row in alert_datetimes_df.iterrows():
            dt = row['start_date']
            if last_date is None or (dt - last_date) > events_time_tolerance:
                current_event += 1
            event_labels.append(f'rt{current_event}')
            last_date = dt
        alert_datetimes_df['event'] = event_labels

        logger.info(f"Number of events detected with tolerance [{events_time_tolerance}]: {len(alert_datetimes_df['event'].unique())}")

        activated_station_lists = [
            alert_mask.loc[row_mask, alert_mask.loc[row_mask]].index.tolist()
            for row_mask in alert_mask[alert_mask.any(axis=1)].index
        ]
        alert_datetimes_df['activated_stations'] = activated_station_lists
        
        alert_datetimes_df['activated_pixels'] = None
        alert_datetimes_df['geo_coordinates'] = None
        for idx, curr_row in alert_datetimes_df.iterrows():
            curr_stations = curr_row['activated_stations']
            dtm_points = {}
            for curr_sta in curr_stations:
                sta_df = station_pixels_association[curr_sta]
                for _, pix_row in sta_df.iterrows():
                    curr_dtm = pix_row['dtm']
                    curr_2d_idx = pix_row['2D_idx']
                    curr_coords = pix_row['coordinates']
                    if curr_dtm not in dtm_points:
                        dtm_points[curr_dtm] = []
                    dtm_points[curr_dtm].append((curr_2d_idx, curr_coords))
            # Aggregate and unique per DTM
            activated_pixels_list = []
            for dtm, points_list in dtm_points.items():
                all_points = np.vstack([p for p, _ in points_list])
                all_coords = np.vstack([c for _, c in points_list])
                unique_indices = np.unique(all_points, axis=0, return_index=True)[1]
                unique_points = all_points[unique_indices]
                unique_coords = all_coords[unique_indices]
                activated_pixels_list.append({'dtm': dtm, 'activated_points': unique_points, 'coordinates': unique_coords})
            alert_datetimes_df.at[idx, 'activated_pixels'] = pd.DataFrame(activated_pixels_list)
            alert_datetimes_df.at[idx, 'geo_coordinates'] = np.concatenate([x['coordinates'] for x in activated_pixels_list], axis=0)

        alert_thr_dict = {
            trigger_mode: alert_thr_df
        }

    elif trigger_mode == 'safety-factor':
        log_and_error(f"Trigger mode {trigger_mode} is not implemented yet.", NotImplementedError, logger)

    elif trigger_mode == 'machine-learning':
        log_and_error(f"Trigger mode {trigger_mode} is not implemented yet.", NotImplementedError, logger)
    
    else:
        log_and_error(f"Trigger mode not recognized or not implemented: {trigger_mode}.", ValueError, logger)

    alert_datetimes_df['top_critical_landslide_path_ids'] = None
    if top_k_paths_per_activation > 0:
        for idx, al_row in alert_datetimes_df.iterrows():
            active_pixels = al_row['activated_pixels']
            curr_top_paths = []
            for _, ap_row in active_pixels.iterrows():
                curr_active_dtm = int(ap_row['dtm'])
                curr_active_2d_idx = ap_row['activated_points']
                curr_top_paths.extend(get_top_k_paths(
                    paths_df=landslide_paths_df,
                    dtm=curr_active_dtm,
                    idx_2d=curr_active_2d_idx,
                    k=top_k_paths_per_activation,
                    separate_starting_points=True
                ))
            curr_top_paths = [x for x in curr_top_paths if x is not None]
            curr_top_paths = list(set(curr_top_paths))
            alert_datetimes_df.at[idx, 'top_critical_landslide_path_ids'] = curr_top_paths

    
    OUT_ALERT_DIR = os.path.join(env.folders['outputs']['tables']['path'], 'attention_pixels_alerts')
    if not os.path.isdir(OUT_ALERT_DIR):
        os.makedirs(OUT_ALERT_DIR)

    attention_pixels_df.to_csv(os.path.join(OUT_ALERT_DIR, 'attention_pixels.csv'), index=False)
    columns_to_export = [col for col in alert_datetimes_df.columns if col != 'activated_pixels']
    alert_datetimes_df[columns_to_export].to_csv(os.path.join(OUT_ALERT_DIR, 'activation_datetimes.csv'), index=False)
    alert_thr_df.to_csv(os.path.join(OUT_ALERT_DIR, DEFAULT_ALERT_THR_FILE[trigger_mode]), index=True)

    alert_vars = {
        'attention_pixels':attention_pixels_df, 
        'activation_datetimes': alert_datetimes_df, 
        'alert_thresholds': alert_thr_dict
    }

    env.save_variable(variable_to_save=alert_vars, variable_filename='alert_vars.pkl')

    return alert_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and alert on attention pixels.")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    parser.add_argument("--trigger_mode", type=str, default='rainfall-threshold', help="Trigger mode")
    parser.add_argument("--alert_thresholds", type=str, default=None, help="Alert thresholds")
    parser.add_argument("--default_thr_mode", type=str, default=None, help="Default threshold mode")
    parser.add_argument("--events_time_tolerance", type=dt.timedelta, default=dt.timedelta(days=5), help="Events time tolerance")
    
    args = parser.parse_args()
    
    alert_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        trigger_mode=args.trigger_mode,
        alert_thresholds=args.alert_thresholds,
        default_thr_mode=args.default_thr_mode,
        events_time_tolerance=args.events_time_tolerance
    )