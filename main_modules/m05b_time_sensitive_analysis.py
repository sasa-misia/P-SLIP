# %% === Import necessary modules
import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    KNOWN_DYNAMIC_INPUT_TYPES,
    DYNAMIC_SUBFOLDERS
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger, obtain_config_idx_and_rel_filename
from main_modules.m04c_import_time_sensitive_data import get_numeric_data_ranges
logger = setup_logger()
logger.info("=== Analyzing time-sensitive data patterns ===")

# %% === Methods to analyze time-sensitive data
def obtain_delta_time_and_possible_ranges(
        ts_vars: dict,
        source_type: str
    ) -> tuple[pd.Timedelta, list[float]]:
    """
    Obtain delta time and possible ranges of time-sensitive data.

    Args:
        ts_vars (dict): Dictionary containing time-sensitive data variables.
        source_type (str): Type of source data.

    Returns:
        tuple(pd.Timedelta, list[float]): Tuple containing delta time and possible ranges [min, max] of time-sensitive data.
    """
    delta_time = ts_vars['datetimes']['start_date'].diff().mean()
    if delta_time > pd.Timedelta(days=31):
        raise ValueError("Invalid delta time. Must be less than 31 days")

    ts_possible_range = get_numeric_data_ranges(source_type=source_type)

    return delta_time, ts_possible_range

def obtain_quantiles_and_noise(
        ts_vars: dict,
        quantiles: list[float]=[.75, .90],
        ts_possible_range: list[float]=[None, None]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: # TODO: Check and refine this implementation (from MATLAB code)
    """
    Obtain percentiles and noise of time-sensitive data.

    Args:
        ts_vars (dict): Dictionary containing time-sensitive data variables.
        quantiles (list[float], optional): List of quantiles to calculate (default: [0.75]).
        ts_possible_range (list[float], optional): Possible range of time-sensitive data (default: [None, None]).

    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: Tuple containing quantiles and mean noise dictionaries per each numeric variable.
    """
    quantiles_dict = {}
    mean_noise_dict = {}
    for ts_label, ts_data in ts_vars['data'].items():
        ts_data = ts_data.copy()
        lower_range_filter = np.ones((ts_data.shape[0],), dtype=bool) if ts_possible_range[0] is None else (ts_data > ts_possible_range[0]).any(axis=1) # Extremes are not included, only effective rainfall (0 -> no rainfall -> no contribute to statistics)
        upper_range_filter = np.ones((ts_data.shape[0],), dtype=bool) if ts_possible_range[1] is None else (ts_data < ts_possible_range[1]).any(axis=1) # Extremes are not included
        rows_inside_range = lower_range_filter & upper_range_filter
        quantiles_dict[ts_label] = ts_data[rows_inside_range].quantile(quantiles, axis=0)

        mean_noise_dict[ts_label] = quantiles_dict[ts_label].copy()
        for idx, perc_row in quantiles_dict[ts_label].iterrows():
            lower_noise_filter = lower_range_filter
            upper_noise_filter = (ts_data < perc_row).any(axis=1) # Filtering out values greater than selected quantiles, because they represent high-importance events
            rows_for_noise = lower_noise_filter & upper_noise_filter
            mean_noise_dict[ts_label].loc[idx] = ts_data[rows_for_noise].mean(axis=0) # All the values below the quantiles are averaged and represent noise (not significant rainfall events)
    
    return quantiles_dict, mean_noise_dict

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        source_type: str="rain",
        source_subtype: str="recordings",
        quantiles: list[float]=[.75, .90]
    ) -> dict[str, pd.DataFrame]:
    """Main function to obtain statistical information of time-sensitive data."""
    if not source_type in KNOWN_DYNAMIC_INPUT_TYPES:
        raise ValueError("Invalid source type. Must be one of: " + ", ".join(KNOWN_DYNAMIC_INPUT_TYPES))
    if not source_subtype in DYNAMIC_SUBFOLDERS:
        raise ValueError("Invalid source subtype. Must be one of: " + ", ".join(DYNAMIC_SUBFOLDERS))
    
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    env, idx_config, rel_filename = obtain_config_idx_and_rel_filename(env, source_type, source_subtype)

    source_mode = env.config['inputs'][source_type][idx_config]['settings']['source_mode']
    if not source_mode == 'station':
        raise ValueError("Invalid source mode. Must be 'station'")
    
    ts_vars = env.load_variable(variable_filename=f"{rel_filename}_vars.pkl")

    delta_time, ts_possible_range = obtain_delta_time_and_possible_ranges(ts_vars, source_type)

    # Analyze rainfall noise
    quantiles_dict, mean_noise_dict = obtain_quantiles_and_noise(ts_vars, quantiles, ts_possible_range)

    ts_statistics = {'time_delta': delta_time, 'quantiles':quantiles_dict, 'noise':mean_noise_dict}

    ts_vars['statistics'] = ts_statistics

    env.config['inputs'][source_type][idx_config]['settings']['quantiles'] = quantiles

    env.save_variable(variable_to_save=ts_vars, variable_filename=f"{rel_filename}_vars.pkl")

    return ts_statistics

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and obtain statistics of time-sensitive data')
    parser.add_argument('--base_dir', type=str, help='Base directory for analysis')
    parser.add_argument('--gui_mode', action='store_true', help='Run in GUI mode')
    parser.add_argument('--source_type', type=str, default='rain', help='Source type (rain, temperature)')
    parser.add_argument('--source_subtype', type=str, default='recordings', help='Source subtype (recordings, forecast)')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[.75, .90], help='List of quantiles to calculate (default: [0.75, 0.90])')
    args = parser.parse_args()

    main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        source_type=args.source_type,
        source_subtype=args.source_subtype,
        quantiles=args.quantiles
    )