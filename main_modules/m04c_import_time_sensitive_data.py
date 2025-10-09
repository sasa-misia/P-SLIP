# %% === Import necessary modules
import os
import sys
import argparse
import pandas as pd
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    KNOWN_DYNAMIC_INPUT_TYPES,
    DYNAMIC_SUBFOLDERS
)

from psliptools.utilities import (
    select_dir_prompt,
    select_files_in_folder_prompt,
    select_file_prompt,
    label_list_elements_prompt,
    get_csv_column_names,
    rename_csv_header
)

from psliptools.scattered import (
    load_time_sensitive_data_from_csv,
    load_time_sensitive_stations_from_csv,
    merge_time_sensitive_data_and_stations
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, obtain_config_idx_and_rel_filename, setup_logger
logger = setup_logger()
logger.info("=== Importing time-sensitive data ===") # This script must be putted after m03, because with satellite you need dtm and abg grids

# %% === Methods to import time sensitive data as rainfall and temperature
SOURCE_MODES = ['station', 'satellite']
AGGREGATION_METHODS = ['mean', 'sum', 'min', 'max']

def get_fill_and_aggregation_methods(
        source_type: str,
        aggregation_method: list[str] | None
    ) -> tuple[str, list[str]]:
    """Utility function to get the fill method and aggregation method based on the source type."""
    if source_type == 'rain':
        fill_method = 'zero'
        if aggregation_method is None:
            aggregation_method = ['sum']
    elif source_type == 'temperature':
        fill_method = 'linear'
        if aggregation_method is None:
            aggregation_method = ['mean']
    else:
        fill_method = None
        if aggregation_method is None:
            aggregation_method = ['sum']
    return fill_method, aggregation_method

def rename_csv_data_headers(
        csv_paths: list[str],
        gui_mode: bool
    ) -> None:
    """Utility function to rename the data files columns."""
    for csv_pth in csv_paths:
        old_datetime_and_numeric_col_names = get_csv_column_names(csv_path=csv_pth, data_type=['datetime', 'number'])
        if gui_mode:
            raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
        else:
            print("\n=== Renaming of data files columns ===")
            new_datetime_and_numeric_col_names = label_list_elements_prompt(
                obj_list=old_datetime_and_numeric_col_names,
                usr_prompt=f"Enter the new labels for the datetim and numeric columns in {csv_pth} (comma separated, press enter for default, which is the same names): "
            )
        
        rename_csv_header(
            csv_path=csv_pth,
            new_header=new_datetime_and_numeric_col_names,
            data_type=['datetime', 'number']
        )

def associate_csv_files_with_gauges(
        csv_paths: list[str],
        station_names: dict[str, str],
        gui_mode: bool
    ) -> pd.DataFrame:
    """Utility function to associate the csv files with the gauges."""
    csv_associated_stations = []
    for data_pth in csv_paths:
        station_matches = [x for x in station_names if x in os.path.basename(data_pth).lower()]
        if len(station_matches) != 1:
            warnings.warn(f"No single data-station association found for {data_pth}", stacklevel=2)
            csv_associated_stations.append(None)
        else:
            new_association = station_names[station_matches[0]]
            if new_association in csv_associated_stations:
                warnings.warn(f"Duplicate data-station association found for {data_pth}", stacklevel=2)
                csv_associated_stations.append(None)
            else:
                csv_associated_stations.append(new_association)
    
    if any(x is None for x in csv_associated_stations):
        if gui_mode:
            raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
        else:
            print("\n=== Association of data files with gauges ===")
            csv_associated_stations = label_list_elements_prompt(
                obj_list=[os.path.basename(x) for x in csv_paths],
                usr_prompt=f"Enter the names of the stations (possible values: {'; '.join(station_names.values())}) for each file (comma separated, same order): "
            )

        if len(set(csv_associated_stations)) != len(csv_associated_stations):
            raise ValueError("Duplicate data-station association found. Please check your association names.")
        
        for sta in csv_associated_stations:
            if sta not in station_names.values():
                raise ValueError(f"Invalid data-station association found. Must be one of: {'; '.join(station_names.values())}")
    return csv_associated_stations

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        source_mode: str="station",
        source_type: str="rain",
        source_subtype: str="recordings",
        round_datetimes_to_nearest_minute: bool=True,
        delta_time_hours: float=1,
        aggregation_method: str=None,
        last_date: pd.Timestamp | str=None,
        rename_csv_data_columns: bool=False
    ) -> dict[str, pd.DataFrame]:
    """Main function to import time sensitive data"""
    if not source_mode in SOURCE_MODES:
        raise ValueError("Invalid source mode. Must be one of: " + ", ".join(SOURCE_MODES))
    if not source_type in KNOWN_DYNAMIC_INPUT_TYPES:
        raise ValueError("Invalid source type. Must be one of: " + ", ".join(KNOWN_DYNAMIC_INPUT_TYPES))
    if not source_subtype in DYNAMIC_SUBFOLDERS:
        raise ValueError("Invalid source subtype. Must be one of: " + ", ".join(DYNAMIC_SUBFOLDERS))
    if aggregation_method is not None and not aggregation_method in AGGREGATION_METHODS:
        raise ValueError("Invalid aggregation method. Must be one of:" + ", ".join(AGGREGATION_METHODS))
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    if last_date is not None and not isinstance(last_date, pd.Timestamp):
        raise TypeError("last_date must be a datetime object or None")
    
    if source_mode == 'station':
        allowed_extensions = ['.csv'] # Other extensions must be implemented
    elif source_mode == 'satellite':
        allowed_extensions = ['.nc'] # Other extensions must be implemented
    
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    env, idx_config, rel_filename = obtain_config_idx_and_rel_filename(env, source_type, source_subtype)

    logger.info(f"Importing time-sensitive data [{source_type}] from [{source_subtype}] in [{source_mode}] mode...")
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Directory selection ===")
        data_fold = select_dir_prompt(
            default_dir=env.folders['inputs'][source_type][source_subtype]['path'], 
            content_type=source_type
        )
        
        print("\n=== Data files selection ===")
        data_paths = select_files_in_folder_prompt(
            base_dir=data_fold, 
            src_ext=allowed_extensions, 
            allow_multiple=True, 
            usr_prompt="Number, name or full path of the data files (ex. data.csv): "
        )

        gauge_info_path = None
        if source_mode == 'station':
            print("\n=== Gauge info file selection ===")
            gauge_info_path = select_file_prompt(
                base_dir=env.folders['inputs'][source_type][source_subtype]['path'], 
                usr_prompt="Name or full path of the gauge info file (ex. gauge_info.csv): ", 
                src_ext=allowed_extensions
            )

    fill_method, aggregation_method = get_fill_and_aggregation_methods(source_type, aggregation_method)
    
    if source_mode == 'station':
        if gauge_info_path in data_paths:
            data_paths.remove(gauge_info_path)
        
        if rename_csv_data_columns:
            logger.info("Renaming csv data files columns...")
            rename_csv_data_headers(csv_paths=data_paths, gui_mode=gui_mode)
        
        logger.info("Loading gauge info file...")
        station_df = load_time_sensitive_stations_from_csv(file_path=gauge_info_path)
        station_names ={x.lower(): x for x in station_df['station'].to_list()}

        cust_ids = []
        _, cust_id = env.add_input_file(file_path=gauge_info_path, file_type=source_type, file_subtype=f"sta")
        cust_ids.append(cust_id)

        data_stations = associate_csv_files_with_gauges(csv_paths=data_paths, station_names=station_names, gui_mode=gui_mode)

        data_dict = {}
        for idx, (data_pth, data_sta) in enumerate(zip(data_paths, data_stations)):
            data_dict[data_sta] = load_time_sensitive_data_from_csv(
                file_path=data_pth,
                fill_method=fill_method,
                round_datetime=round_datetimes_to_nearest_minute,
                delta_time_hours=delta_time_hours,
                aggregation_method=aggregation_method,
                last_date=last_date,
                datetime_columns_names=None, # Auto-detect
                numeric_columns_names=None # Auto-detect
            )

            _, cust_id = env.add_input_file(file_path=data_pth, file_type=source_type, file_subtype=f"rec{idx+1}")
            station_df.loc[station_df['station'] == data_sta, 'file_id'] = cust_id
            cust_ids.append(cust_id)

        logger.info("Merging time-sensitive data files with gauges info...")
        time_sensitive_vars = merge_time_sensitive_data_and_stations(
            data_dict=data_dict,
            stations_table=station_df
        )

    else:
        raise NotImplementedError("Satellite mode is not supported in this script yet. Please contact the developer.")
    
    common_start_date = time_sensitive_vars['dates']['start_date'].iloc[0].strftime("%Y-%b-%d %H:%M:%S")
    common_end_date = time_sensitive_vars['dates']['end_date'].iloc[-1].strftime("%Y-%b-%d %H:%M:%S")

    env.config['inputs'][source_type][idx_config]['settings'] = {
        'source_mode': source_mode,
        'source_subtype': source_subtype,
        'delta_time_hours': delta_time_hours,
        'aggregation_method': aggregation_method,
        'common_start_date': common_start_date,
        'common_end_date': common_end_date
    }
    env.config['inputs'][source_type][idx_config]['custom_id'] = cust_ids

    env.collect_input_files(file_type=[source_type], multi_extension=True)

    env.save_variable(variable_to_save=time_sensitive_vars, variable_filename=f"{rel_filename}_vars.pkl")

    return time_sensitive_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import time-sensitive data (rain and temperature)")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    parser.add_argument("--source_mode", type=str, default="station", help="Source mode (station, satellite)")
    parser.add_argument("--source_type", type=str, default="rain", help="Source type (rain, temperature)")
    parser.add_argument("--source_subtype", type=str, default="recordings", help="Source subtype (recordings, forecast)")
    parser.add_argument("--delta_time_hours", type=int, default=1, help="Delta time in hours")
    parser.add_argument("--aggregation_method", type=list[str], default=None, help="Aggregation method (mean, sum, min, max)")
    parser.add_argument("--last_date", type=str, default=None, help="Last date (YYYY-MM-DD HH:mm)")
    parser.add_argument("--round_datetimes_to_nearest_minute", action="store_true", help="Round datetimes to the nearest minute")
    
    args = parser.parse_args()
    
    time_sensitive_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        source_mode=args.source_mode,
        source_type=args.source_type,
        source_subtype=args.source_subtype,
        delta_time_hours=args.delta_time_hours,
        aggregation_method=args.aggregation_method,
        last_date=args.last_date,
        round_datetimes_to_nearest_minute=args.round_datetimes_to_nearest_minute
    )

# %%