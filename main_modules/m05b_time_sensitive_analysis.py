# %% === Import necessary modules
import os
import sys
import warnings
import argparse
import pandas as pd
import numpy as np
import datetime as dt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    KNOWN_DYNAMIC_INPUT_TYPES,
    DYNAMIC_SUBFOLDERS
)

from psliptools.utilities import (
    get_mask_in_range
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger, obtain_config_idx_and_rel_filename
from main_modules.m04c_import_time_sensitive_data import get_numeric_data_ranges
logger = setup_logger(__name__)
logger.info("=== Analyzing time-sensitive data patterns ===")

# %% === Methods to analyze time-sensitive data
def get_time_sensitive_statistics(
        time_sensitive_vars: dict,
        quantiles: list[float]=[.8, .9],
        numeric_data_range: list[float]=[None, None],
        include_range_extremes: bool=False
    ) -> dict:
    """
    Obtain percentiles and noise of time-sensitive data.

    Args:
        time_sensitive_vars (dict): Dictionary containing time-sensitive data variables.
        quantiles (list[float], optional): List of quantiles to calculate. Defaults to [0.8, 0.9].
        numeric_data_range (list[float], optional): List of numeric data range. Defaults to [None, None].
        include_range_extremes (bool, optional): Whether to include range extremes. Defaults to False.

    Returns:
        dict: Dictionary containing time-sensitive data statistics.
    """
    delta_time = time_sensitive_vars['datetimes']['start_date'].diff().mean()
    if delta_time > pd.Timedelta(days=31):
        raise ValueError("Invalid delta time. Must be less than 31 days")
    
    ts_stats = {
        'data_delta_time': delta_time, 
        'count': {}, 
        'basic':{}, 
        'quantiles':{}, 
        'noise':{'mean':{}, 'max':{}}
    }
    for ts_label, ts_data in time_sensitive_vars['data'].items():
        noise_count_labels = ['noise-' + str(quantile) for quantile in quantiles]
        ts_stats['count'][ts_label] = pd.DataFrame(index=['basic', 'quantiles', *noise_count_labels], columns=ts_data.columns, dtype='Int64')
        ts_stats['basic'][ts_label] = pd.DataFrame(index=['mean', 'std', 'min', 'max', 'median', 'mode'], columns=ts_data.columns, dtype=np.float64)
        ts_stats['quantiles'][ts_label] = pd.DataFrame(index=quantiles, columns=ts_data.columns, dtype=np.float64) # Initialize dataframe
        ts_stats['noise']['mean'][ts_label] = ts_stats['quantiles'][ts_label].copy() # Initialize dataframe
        ts_stats['noise']['max'][ts_label] = ts_stats['quantiles'][ts_label].copy() # Initialize dataframe
        for col in ts_data.columns:
            curr_col = ts_data[col].copy()

            rows_inside_range = get_mask_in_range(
                series=curr_col,
                min_value=numeric_data_range[0],
                max_value=numeric_data_range[1],
                include_min_max=include_range_extremes
            )
            num_obs_range = rows_inside_range.sum()

            ts_stats['count'][ts_label].loc[['basic', 'quantiles'], col] = [num_obs_range, num_obs_range]
            ts_stats['basic'][ts_label].loc[['mean', 'std', 'min', 'max'], col] = curr_col[rows_inside_range].describe().loc[['mean', 'std', 'min', 'max']]
            ts_stats['basic'][ts_label].loc[['median', 'mode'], col] = [curr_col[rows_inside_range].median(), curr_col[rows_inside_range].mode().iloc[0]]
            ts_stats['quantiles'][ts_label][col] = curr_col[rows_inside_range].quantile(quantiles)

            for qnt_idx, qnt_val in ts_stats['quantiles'][ts_label][col].items():
                rows_for_noise = get_mask_in_range(
                    series=curr_col,
                    min_value=numeric_data_range[0],
                    max_value=qnt_val,
                    include_min_max=include_range_extremes
                )
                num_obs_noise = rows_for_noise.sum()

                ts_stats['count'][ts_label].loc['noise-'+str(qnt_idx), col] = num_obs_noise
                ts_stats['noise']['mean'][ts_label].loc[qnt_idx, col] = curr_col[rows_for_noise].mean() # All the values below the quantiles are averaged and the result represents the mean noise (not significant rainfall events)
                ts_stats['noise']['max'][ts_label].loc[qnt_idx, col] = curr_col[rows_for_noise].max() # The maximum is extracted and represents the real recorded maximum noise
    
    return ts_stats

def get_mobile_averages(
        time_sensitive_vars: dict,
        moving_average_window: dt.timedelta | pd.Timedelta=pd.Timedelta(days=31),
        numeric_data_range: list[float]=[None, None],
        include_range_extremes: bool=False,
        cut_outside_range: bool = True,
        weights: list[float] | np.ndarray | pd.Series=None
    ) -> dict:
    """
    Calculate different types of moving averages for time-sensitive data.

    Args:
        time_sensitive_vars (dict): Dictionary containing time-sensitive data.
        moving_average_window (dt.timedelta | pd.Timedelta, optional): Moving average window duration. Defaults to pd.Timedelta(days=31).
        numeric_data_range (list[float], optional): Numeric data range. Defaults to [None, None].
        include_range_extremes (bool, optional): Whether to include range extremes. Defaults to False.
        weights (list[float] | np.ndarray | pd.Series, optional): Weights for weighted moving average. Defaults to None.

    Returns:
        dict: Dictionary containing different types of moving averages.
    """
    # Helper: mask a series according to the numeric range
    def _mask_series(series):
        """Return a copy where out‑of‑range values are set to NaN."""
        mask = get_mask_in_range(
            series=series,
            min_value=numeric_data_range[0],
            max_value=numeric_data_range[1],
            include_min_max=include_range_extremes
        )
        return series.where(mask, np.nan), mask
    
    # Helper: filter and possibly cut results of mobile average if outside range limits
    def _filter_series(series):
        """Return a copy where out-of-range values are set to range limits."""
        series = series.copy()
        lower_mask = (series < numeric_data_range[0]).to_numpy() if numeric_data_range[0] is not None else np.zeros(len(series), dtype=bool)
        upper_mask = (series > numeric_data_range[1]).to_numpy() if numeric_data_range[1] is not None else np.zeros(len(series), dtype=bool)
        if lower_mask.any() and cut_outside_range:
            series[lower_mask] = numeric_data_range[0]
        if upper_mask.any() and cut_outside_range:
            series[upper_mask] = numeric_data_range[1]
        out_of_range_mask = pd.Series(lower_mask | upper_mask, index=series.index, name=series.name, dtype=bool)
        return series, out_of_range_mask
    
    # Helper: weighted moving average
    def _weighted_ma(x, weights_arr):
        """Return the weighted moving average."""
        valid_mask = ~np.isnan(x)
        if not valid_mask.any():
            return np.nan
        idx = x.index[valid_mask]  # Get original indices for valid elements
        valid_x = x[valid_mask]
        valid_w = weights_arr[idx]  # Use original indices to get correct weights
        return np.average(valid_x, weights=valid_w)

    # Helper: hull weighted moving average
    def _hull_wma(x):
        """Return the hull weighted moving average."""
        valid_mask = ~np.isnan(x)
        if not valid_mask.any():
            return np.nan
        valid_idx = x.index[valid_mask]  # Get original indices for valid elements
        valid_w = np.arange(1, len(valid_idx) + 1)  # Weights based on position in valid sequence
        valid_x = x[valid_mask]
        return np.sum(valid_x * valid_w) / np.sum(valid_w)

    # Validate window vs. data delta
    delta_time = time_sensitive_vars['datetimes']['start_date'].diff().mean()
    if delta_time > moving_average_window:
        raise ValueError(f"Invalid moving_average_window [{moving_average_window}]. Must be greater than data delta_time: [{delta_time}]")
    if abs((moving_average_window % delta_time).total_seconds()) > 0.01:
        raise ValueError(f"Invalid moving_average_window [{moving_average_window}]. Must be a multiple of data delta_time: [{delta_time}]")
    rows_window = int(moving_average_window / delta_time)

    if rows_window < 2:
        warnings.warn("Hull moving average will not be calculated: the moving_average_window is smaller than twice the data delta_time.", stacklevel=2)

    # Validate weights
    if weights is None:
        weights_arr = np.ones(time_sensitive_vars['datetimes'].shape[0], dtype=np.float64)
    else:
        weights_arr = np.array(weights, dtype=np.float64)
    
    if weights_arr.size != time_sensitive_vars['datetimes'].shape[0]:
        raise ValueError(f"Invalid weights length [{weights_arr.size}]. Must match data length: [{time_sensitive_vars['datetimes'].shape[0]}]")
    
    # Hull weights
    hull_fast = 2 / (2 + 1)
    hull_slow = 2 / (30 + 1)

    # Initialise result containers
    ts_mobile_averages = {
        'data_delta_time': delta_time,
        'window_delta_time': moving_average_window,
        'count': {},
        'simple': {},
        'exponential': {},
        'cumulative': {},
        'weighted': {},
        'hull': {},
        'adaptive': {}
    }

    out_of_range_mask = {
        'simple': {},
        'exponential': {},
        'cumulative': {},
        'weighted': {},
        'hull': {},
        'adaptive': {}
    }

    for ts_label, ts_data in time_sensitive_vars['data'].items():
        # create empty dataframes for each metric
        ts_mobile_averages['count'][ts_label] = pd.DataFrame(index=ts_data.index, columns=ts_data.columns, dtype='Int64')
        for ts_ma_key, _ in ts_mobile_averages.items():
            if isinstance(ts_mobile_averages[ts_ma_key], dict) and not ts_label in ts_mobile_averages[ts_ma_key]:
                ts_mobile_averages[ts_ma_key][ts_label] = pd.DataFrame(index=ts_data.index, columns=ts_data.columns, dtype=np.float64)
                out_of_range_mask[ts_ma_key][ts_label] = pd.DataFrame(index=ts_data.index, columns=ts_data.columns, dtype=bool)
        
        for col in ts_data.columns:
            curr_col = ts_data[col].copy()

            # Apply range filter once and reuse the masked series
            masked_curr_col, curr_col_mask = _mask_series(curr_col)

            # Count of valid observations inside the rolling window
            count_valid_obs = curr_col_mask.rolling(
                window=rows_window,
                min_periods=rows_window, # This means NaN for partial windows (there are less than rows_window observations)
                closed='right'
            ).sum() # Vectorized sum (pandas sum ignores NaNs)

            ts_mobile_averages['count'][ts_label][col] = count_valid_obs.astype('Int64')

            # -------------------- Simple Moving Average --------------------
            ma_simple = masked_curr_col.rolling(
                window=rows_window,
                min_periods=1,
                closed='right'
            ).mean() # Vectorized mean (pandas mean ignores NaNs)

            ts_mobile_averages['simple'][ts_label][col], out_of_range_mask['simple'][ts_label][col]  = _filter_series(ma_simple)

            # -------------------- Exponential Moving Average ---------------
            ma_exponential = masked_curr_col.ewm(
                span=rows_window,
                min_periods=1,
                adjust=False
            ).mean() # Vectorized mean (pandas mean ignores NaNs)

            ts_mobile_averages['exponential'][ts_label][col], out_of_range_mask['exponential'][ts_label][col] = _filter_series(ma_exponential)

            # -------------------- Cumulative Moving Average ---------------
            ma_cumulative = masked_curr_col.expanding( # With expanding, all the values are cumulative, since the beginning of the series -> this mobile average is very slow and stable
                min_periods=1
            ).mean() # Vectorized mean (pandas mean ignores NaNs)

            ts_mobile_averages['cumulative'][ts_label][col], out_of_range_mask['cumulative'][ts_label][col] = _filter_series(ma_cumulative)

            # -------------------- Weighted Moving Average ------------------
            ma_weighted = masked_curr_col.rolling(
                window=rows_window,
                min_periods=1,
                closed='right'
            ).apply(lambda x: _weighted_ma(x, weights_arr), raw=False) # Remember to exclude NaNs is _weighted_ma

            ts_mobile_averages['weighted'][ts_label][col], out_of_range_mask['weighted'][ts_label][col] = _filter_series(ma_weighted)

            # -------------------- Hull Moving Average ----------------------
            if rows_window >= 2:
                wma_short = masked_curr_col.rolling(
                    window=rows_window // 2,
                    min_periods=1,
                    closed='right'
                ).apply(_hull_wma, raw=False) # Remember to exclude NaNs is _hull_wma

                wma_long = masked_curr_col.rolling(
                    window=rows_window,
                    min_periods=1,
                    closed='right'
                ).apply(_hull_wma, raw=False) # Remember to exclude NaNs is _hull_wma

                raw_hma = (2 * wma_short) - wma_long
                hma_window = max(1, int(np.sqrt(rows_window)))

                ma_hull = raw_hma.rolling(
                    window=hma_window,
                    min_periods=1,
                    closed='right'
                ).mean() # Vectorized mean (pandas mean ignores NaNs)

                ts_mobile_averages['hull'][ts_label][col], out_of_range_mask['hull'][ts_label][col] = _filter_series(ma_hull)

            # -------------------- Adaptive Moving Average -----------------
            change = masked_curr_col.diff(rows_window).abs()

            volatility = masked_curr_col.diff().abs().rolling(
                window=rows_window,
                min_periods=1,
                closed='right'
            ).sum()  # Vectorized sum (pandas sum ignores NaNs)

            efficiency_ratio = pd.Series(np.zeros_like(change), index=change.index)
            valid_vol_mask = (volatility != 0) & (~volatility.isna())
            efficiency_ratio[valid_vol_mask] = change[valid_vol_mask] / volatility[valid_vol_mask]

            smoothing_constant = (efficiency_ratio * (hull_fast - hull_slow) + hull_slow) ** 2
            ma_adaptive = ma_simple.copy()  # Using the simple MA as a base for adaptive MA
            for i in range(rows_window, len(curr_col)):
                curr_val = masked_curr_col.iloc[i]
                corrective_term = smoothing_constant.iloc[i] * (curr_val - ma_adaptive.iloc[i-1])
                if not pd.isna(corrective_term):
                    corrective_term = smoothing_constant.iloc[i] * (curr_val - ma_adaptive.iloc[i-1])
                    ma_adaptive.iloc[i] = ma_adaptive.iloc[i-1] + corrective_term # ma_adaptive can be seen as a corrected version of ma_simple
            
            ts_mobile_averages['adaptive'][ts_label][col], out_of_range_mask['adaptive'][ts_label][col] = _filter_series(ma_adaptive)
    
    warn_msg = "Out of range values were detected and automatically cut" if cut_outside_range else "Out of range values detected (please check it!)"
    for ma_key, ma_dict in out_of_range_mask.items():
        for ts_label, out_of_range_mask in ma_dict.items():
            if out_of_range_mask.any(axis=None):
                warnings.warn(f"{warn_msg} in {ma_key} MA for {ts_label}", stacklevel=2)

    return ts_mobile_averages

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        source_type: str="rain",
        source_subtype: str="recordings",
        quantiles: list[float] | float=[.8, .9],
        numeric_range: list[float]=[None, None],
        include_range_extremes: bool=False,
        cut_outside_range: bool=True,
        moving_average_window: pd.Timedelta=pd.Timedelta(days=31)
    ) -> dict[str, pd.DataFrame]:
    """Main function to obtain statistical information of time-sensitive data."""
    if not source_type in KNOWN_DYNAMIC_INPUT_TYPES:
        raise ValueError("Invalid source type. Must be one of: " + ", ".join(KNOWN_DYNAMIC_INPUT_TYPES))
    if not source_subtype in DYNAMIC_SUBFOLDERS:
        raise ValueError("Invalid source subtype. Must be one of: " + ", ".join(DYNAMIC_SUBFOLDERS))
    if not isinstance(quantiles, list):
        quantiles = [quantiles]
    if not all([isinstance(quantile, float) and 0 <= quantile <= 1 for quantile in quantiles]):
        raise TypeError("quantiles must be a list of floats and all values must be between 0 and 1")
    if not isinstance(numeric_range, list) or len(numeric_range) != 2:
        raise TypeError("numeric_range must be a list of length 2")
    if not all([isinstance(numeric, (float, type(None))) for numeric in numeric_range]):
        raise TypeError("numeric_range must be a list of floats")
    
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    env, idx_config, rel_filename = obtain_config_idx_and_rel_filename(env, source_type, source_subtype)

    source_mode = env.config['inputs'][source_type][idx_config]['settings']['source_mode']
    if not source_mode == 'station':
        raise ValueError("Invalid source mode. Must be 'station'")
    
    ts_vars = env.load_variable(variable_filename=f"{rel_filename}_vars.pkl")

    ts_possible_range = get_numeric_data_ranges(source_type=source_type)

    if numeric_range[0] is not None:
        ts_possible_range[0] = numeric_range[0]
    if numeric_range[1] is not None:
        ts_possible_range[1] = numeric_range[1]

    logger.info(f"Obtaining statistics for {source_type} data ({source_subtype})...")
    ts_statistics = get_time_sensitive_statistics(
        time_sensitive_vars=ts_vars, 
        quantiles=quantiles, 
        numeric_data_range=ts_possible_range, 
        include_range_extremes=include_range_extremes
    )

    logger.info(f"Obtaining mobile averages for {source_type} data ({source_subtype})...")
    ts_mobile_averages = get_mobile_averages(
        time_sensitive_vars=ts_vars,
        moving_average_window=moving_average_window,
        numeric_data_range=ts_possible_range,
        include_range_extremes=include_range_extremes,
        cut_outside_range=cut_outside_range,
        weights=None
    )

    ts_vars['statistics'] = ts_statistics
    ts_vars['mobile_averages'] = ts_mobile_averages

    env.config['inputs'][source_type][idx_config]['settings']['quantiles'] = quantiles

    env.save_variable(variable_to_save=ts_vars, variable_filename=f"{rel_filename}_vars.pkl")

    return ts_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and obtain statistics of time-sensitive data')
    parser.add_argument('--base_dir', type=str, help='Base directory for analysis')
    parser.add_argument('--gui_mode', action='store_true', help='Run in GUI mode')
    parser.add_argument('--source_type', type=str, default='rain', help='Source type (rain, temperature)')
    parser.add_argument('--source_subtype', type=str, default='recordings', help='Source subtype (recordings, forecast)')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[.75, .90], help='List of quantiles to calculate (default: [0.75, 0.90])')
    parser.add_argument('--numeric_range', type=float, nargs=2, default=[None, None], help='Numeric range to consider (default: [None, None])')
    parser.add_argument('--include_range_extremes', action='store_true', help='Include range extremes (default: False)')
    parser.add_argument('--cut_outside_range', action='store_true', help='Cut outside range (default: True)')
    parser.add_argument('--moving_average_window', type=dt.timedelta, default=dt.timedelta(days=31), help='Moving average window (default: 31 days)')
    args = parser.parse_args()

    main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        source_type=args.source_type,
        source_subtype=args.source_subtype,
        quantiles=args.quantiles,
        numeric_range=args.numeric_range,
        include_range_extremes=args.include_range_extremes,
        cut_outside_range=args.cut_outside_range,
        moving_average_window=args.moving_average_window
    )