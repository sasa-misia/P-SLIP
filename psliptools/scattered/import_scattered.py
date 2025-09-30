# %% === Import necessary modules
import warnings
import numpy as np
import pandas as pd
import os

# %% === Method to fill gaps in a series with the mean between the first and last non-empty values
def _fill_missing_values_with_mean(
        column_data: pd.Series
    ) -> pd.Series:
    """
    Fill missing values in a pandas Series with the mean between the first and last non-empty values.

    Args:
        column_data (pd.Series): The pandas numeric Series to fill.

    Returns:
        pd.Series: The filled pandas Series.
    """
    if not isinstance(column_data, pd.Series):
        raise TypeError("col_data must be a pandas Series.")
    if column_data.isnull().all():
        raise ValueError("Column data is entirely missing.")
    if not pd.api.types.is_numeric_dtype(column_data.dtype):
        raise TypeError("Column data must be numeric.")
    
    column_data = column_data.copy()

    # Find gaps (consecutive NaN sequences)
    isnull = column_data.isnull()
    gap_starts = isnull & ~isnull.shift(1, fill_value=False)
    gap_ends = isnull & ~isnull.shift(-1, fill_value=False)
    
    for i in range(len(column_data)):
        if gap_starts.iloc[i]:
            # Find gap end
            j = i
            while j < len(column_data) and isnull.iloc[j]:
                j += 1
            if j < len(column_data):  # Found end
                gap_end = j - 1
                # Get values before and after gap
                prev_val = column_data.iloc[i-1] if i > 0 else column_data.iloc[gap_end+1]
                next_val = column_data.iloc[gap_end+1] if gap_end+1 < len(column_data) else prev_val
                # Calculate mean for the gap
                gap_mean = (prev_val + next_val) / 2
                # Fill the gap
                column_data.iloc[i:gap_end+1] = gap_mean
    
    return column_data

# %% === Method to fill gaps in a series with different methods
def _fill_missing_values_of_numeric_series(
        column_data: pd.Series,
        fill_method: str | int | float='zero'
    ) -> pd.Series:
    """
    Fill missing values in a pandas Series with different methods.

    Args:
        column_data (pd.Series): The pandas numeric Series to fill.
        fill_method (str, optional): The method to use for filling missing values (default: 'zero').
            Possible values are
                1. 'zero' - Fill missing values with 0.
                2. 'mean' - Fill missing values with the mean bewteen the first and last non-empty values.
                3. 'nearest' - Fill missing values with the nearest non-empty value.
                4. 'previous' - Fill missing values with the previous non-empty value.
                5. 'next' - Fill missing values with the next non-empty value.
                6. 'linear' - Fill missing values with the linear interpolation between the first and last non-empty values.
                7. 'quadratic' - Fill missing values with the quadratic interpolation between the first and last non-empty values.
                8. 'cubic' - Fill missing values with the cubic interpolation between the first and last non-empty values.

    Returns:
        pd.Series: The filled pandas Series.
    """
    column_data = column_data.copy()

    if isinstance(fill_method, (int, float)):
        if fill_method == 0:
            fill_method = None
        elif fill_method == 1:
            fill_method = 'zero'
        elif fill_method == 2:
            fill_method = 'mean'
        elif fill_method == 3:
            fill_method = 'nearest'
        elif fill_method == 4:
            fill_method = 'previous'
        elif fill_method == 5:
            fill_method = 'next'
        elif fill_method == 6:
            fill_method = 'linear'
        elif fill_method == 7:
            fill_method = 'quadratic'
        elif fill_method == 8:
            fill_method = 'cubic'
        else:
            raise ValueError(f"Invalid numeric value for fill method: {fill_method}")
    
    if not isinstance(column_data, pd.Series):
        raise TypeError("col_data must be a pandas Series.")
    if column_data.isnull().all():
        raise ValueError("Column data is entirely missing.")
    if not pd.api.types.is_numeric_dtype(column_data.dtype):
        raise TypeError("Column data must be numeric.")
    if not isinstance(fill_method, (str, type(None))):
        raise TypeError("fill_method must be a string or None.")
    
    if fill_method is None:
        return column_data
        
    if fill_method == 'zero':
        column_data = column_data.fillna(0)
    elif fill_method == 'mean':
        column_data = _fill_missing_values_with_mean(column_data)
    elif fill_method == 'nearest':
        column_data = column_data.interpolate(method='nearest')
    elif fill_method == 'previous':
        column_data = column_data.ffill()
    elif fill_method == 'next':
        column_data = column_data.bfill()
    elif fill_method == 'linear':
        column_data = column_data.interpolate(method='linear')
    elif fill_method == 'quadratic':
        column_data = column_data.interpolate(method='quadratic')
    elif fill_method == 'cubic':
        column_data = column_data.interpolate(method='cubic')
    else:
        raise ValueError(f"Invalid fill method: {fill_method}")
    
    return column_data

# %% === Helper method to parse and validate a time-sensitive dataframe
def _parse_time_sensitive_dataframe(
        data_df: pd.DataFrame,
        fill_method: str | int | float=None,
        round_datetime: bool=True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse and validate a time-sensitive dataframe.

    Args:
        data_df (pd.DataFrame): The dataframe to parse and validate.
        fill_method (str, optional): The method to use for filling missing values (default: None).
            Possible values are
                0. None - Do not fill missing values.
                1. 'zero' - Fill missing values with 0.
                2. 'mean' - Fill missing values with the mean bewteen the first and last non-empty values.
                3. 'nearest' - Fill missing values with the nearest non-empty value.
                4. 'previous' - Fill missing values with the previous non-empty value.
                5. 'next' - Fill missing values with the next non-empty value.
                6. 'linear' - Fill missing values with the linear interpolation between the first and last non-empty values.
                7. 'quadratic' - Fill missing values with the quadratic interpolation between the first and last non-empty values.
                8. 'cubic' - Fill missing values with the cubic interpolation between the first and last non-empty values.
        round_datetime (bool, optional): If True, round datetime columns to the nearest minute (default: True).

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame): A tuple containing the parsed dataframe, the datetime columns, and the numeric columns.
    """
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame.")
    
    if not isinstance(fill_method, (str, int, float, type(None))):
        raise TypeError("fill_method must be a string, integer, float, or None.")
    
    if not isinstance(round_datetime, bool):
        raise TypeError("round_datetime must be a boolean.")

    data_df = data_df.copy()

    for col in data_df.columns:
        col_type = data_df[col].dtype
        if col_type == 'object':
            # Check if it might be a date column
            try:
                data_df[col] = pd.to_datetime(data_df[col], errors='raise')
            except:
                pass
        elif pd.api.types.is_numeric_dtype(col_type):
            pass
        else:
            try:
                data_df[col] = pd.to_numeric(data_df[col], errors='raise')
            except:
                pass

    for col in data_df.columns:
        if pd.api.types.is_numeric_dtype(data_df[col].dtype):
            missing_row_ids = data_df[data_df[col].isnull()].index.to_list()
            if len(missing_row_ids) > 0 and fill_method is not None:
                data_df[col] = _fill_missing_values_of_numeric_series(
                    column_data=data_df[col], 
                    fill_method=fill_method
                )
                
                warnings.warn(
                    f"Missing values in column [{col}] at rows {[x + 2 for x in missing_row_ids]} have been filled with {fill_method} mode", # + 2 because the csv first row is for header and starts from 1, not 0!
                    stacklevel=2
                )
            
            data_df[col] = data_df[col].astype('float64')

        if pd.api.types.is_datetime64_any_dtype(data_df[col]):
            missing_row_ids = data_df[data_df[col].isnull()].index.to_list()
            if len(missing_row_ids) > 0:
                if data_df[col].dtype == 'object':
                    data_df[col] = data_df[col].infer_objects(copy=False) # Suggested to convert first to infer_objects
                data_df[col] = data_df[col].interpolate(method='linear')
                warnings.warn(
                    f"Missing values in column [{col}] at rows {[x + 2 for x in missing_row_ids]} have been filled with linear interpolation", # + 2 because the csv first row is for header and starts from 1, not 0!
                    stacklevel=2
                )
            
            if round_datetime:
                data_df[col] = data_df[col].dt.round('1min') # Round to nearest minute

            delta_time_hours = data_df[col].diff().dt.total_seconds() / 3600
            non_uniform_rows = delta_time_hours[delta_time_hours != delta_time_hours.iloc[1]].index.to_list()
            if len(non_uniform_rows) > 1: # > 1 because the first row is always missing
                raise ValueError(f"Time column [{col}] is not uniform. Non-uniform rows: {non_uniform_rows}")

            # Handle timezone-aware datetime conversion
            if data_df[col].dt.tz is not None:
                data_df[col] = data_df[col].dt.tz_localize(None)
            
            data_df[col] = data_df[col].astype('datetime64[ns]')
    
    datetime_df = data_df.select_dtypes(include=['datetime64[ns]'])
    numeric_df = data_df.select_dtypes(include=['float64', 'int64'])

    if len(datetime_df.columns) != 2:
        raise ValueError("Expected two datetime columns in the dataframe (start and end date)")

    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found in the dataframe")

    return data_df, datetime_df, numeric_df

# %% === Method to change time resolution of time-sensitive data
def _change_time_sensitive_dataframe_resolution(
        datetime_df: pd.DataFrame,
        numeric_df: pd.DataFrame | pd.Series,
        delta_time: pd.Timedelta,
        aggregation_method: list[str]=['sum'],
        last_date: pd.Timestamp=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Change the time resolution of a time-sensitive dataframe.

    Args:
        datetime_df (pd.DataFrame): The dataframe containing datetime columns.
        numeric_df (pd.DataFrame | pd.Series): The dataframe containing numeric columns.
        delta_time (pd.Timedelta): The desired time resolution.
        aggregation_method (list[str], optional): The aggregation method to use (default: ['sum']).
            Possible values of each element of the list are 
                1. 'mean' - The aggregated rows will be the mean of the original rows.
                2. 'sum' - The aggregated rows will be the sum of the original rows.
                3. 'min' - The aggregated rows will be the minimum of the original rows.
                4. 'max' - The aggregated rows will be the maximum of the original rows.
        last_date (pd.Timestamp, optional): The last date to include in the aggregation. 
            All dates greater than this will be filtered out. If None, use the last available date.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): A tuple containing the datetime dataframe and the numeric dataframe.
    """
    if not isinstance(datetime_df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for datetime_df")
    if not isinstance(numeric_df, (pd.DataFrame, pd.Series)):
        if isinstance(numeric_df, pd.Series):
            numeric_df = numeric_df.to_frame()
        raise TypeError("Expected a pandas DataFrame or Series for numeric_df")
    if not isinstance(delta_time, pd.Timedelta):
        raise TypeError("Expected a pandas Timedelta for delta_time")
    if not isinstance(aggregation_method, list):
        raise TypeError("Expected a list for aggregation_method")
    if not all(isinstance(method, str) for method in aggregation_method):
        raise TypeError("Expected a list of strings for aggregation_method")
    if last_date is not None and not isinstance(last_date, pd.Timestamp):
        raise TypeError("Expected a pandas Timestamp for last_date")
    
    if len(aggregation_method) == 1:
        aggregation_method = aggregation_method * numeric_df.shape[1]
    
    if len(aggregation_method) != numeric_df.shape[1]:
        raise ValueError("Expected the same number of aggregation methods as numeric columns")
    
    datetime_df = datetime_df.copy()
    numeric_df = numeric_df.copy()

    if not (datetime_df.shape[1] == 2 and (datetime_df.columns == ['start_date', 'end_date']).all()):
        raise ValueError("Expected two datetime columns in the dataframe (start_date and end_date)")
    
    # Filter data based on last_date if provided
    if last_date is not None:
        # Find the closest end_date to the specified last_date
        time_diffs = (datetime_df['end_date'] - last_date).abs()
        closest_idx = time_diffs.idxmin()
        closest_end_date = datetime_df.loc[closest_idx, 'end_date']

        datetimes_after_last_date = datetime_df['end_date'] > closest_end_date
        count_removed = datetimes_after_last_date.sum()
        if datetimes_after_last_date.any():
            warnings.warn(
                f"Filtering out {count_removed} rows after last_date ({last_date})",
                stacklevel=2
            )

        # Filter out dates greater than the closest end_date
        mask = datetime_df['end_date'] <= closest_end_date
        datetime_df = datetime_df[mask].reset_index(drop=True)
        numeric_df = numeric_df[mask].reset_index(drop=True)
    
    # Check if new delta is multiple of existing delta
    if len(datetime_df) > 1:
        existing_delta = datetime_df['end_date'].iloc[1] - datetime_df['end_date'].iloc[0]
        if delta_time.total_seconds() % existing_delta.total_seconds() != 0:
            raise ValueError(f"New delta time ({delta_time}) must be a multiple of existing delta time ({existing_delta})")
    else:
        raise ValueError("Not enough data points for aggregation after filtering")
    
    # Calculate the number of complete groups backwards from last_end_date
    expected_rows_per_group = int(delta_time.total_seconds() / existing_delta.total_seconds())
    
    # Create group labels by assigning each row to a group starting from the end
    groups = []
    current_group = 0
    remaining_in_group = expected_rows_per_group
    
    # Go backwards through the dataframe
    for i in range(len(datetime_df) - 1, -1, -1):
        groups.insert(0, current_group) # This is to add at the beginning of the list, not append at the end
        remaining_in_group -= 1
        
        if remaining_in_group == 0:
            current_group += 1
            remaining_in_group = expected_rows_per_group
    
    # Check for incomplete group at the beginning
    first_group_complete = True
    if len(groups) > 0 and groups.count(groups[0]) < expected_rows_per_group:
        first_group_complete = False
        warnings.warn(
            f"Incomplete group found at the beginning with {groups.count(groups[0])} rows. " # groups.count(groups[0]) to count how many rows are in the first group, because count returns the number of times the value appears in the list
            f"This group will be aggregated with available data and start_date will be adjusted to maintain {delta_time} duration.",
            stacklevel=2
        )
    
    # Convert groups to pandas Series for grouping
    groups_series = pd.Series(groups, index=datetime_df.index)
    
    # Aggregate datetime columns: first start_date and last end_date for each group
    datetime_agg_df = datetime_df.groupby(groups_series).agg({
        'start_date': 'first',
        'end_date': 'last'
    })
    
    # Aggregate numeric columns with specified method
    numeric_agg_df = pd.DataFrame()
    for idx, col in enumerate(numeric_df.columns):
        curr_aggregation_method = aggregation_method[idx]
        if curr_aggregation_method == 'mean':
            numeric_agg_df[col] = numeric_df.groupby(groups_series)[col].mean()
        elif curr_aggregation_method == 'sum':
            numeric_agg_df[col] = numeric_df.groupby(groups_series)[col].sum()
        elif curr_aggregation_method == 'min':
            numeric_agg_df[col] = numeric_df.groupby(groups_series)[col].min()
        elif curr_aggregation_method == 'max':
            numeric_agg_df[col] = numeric_df.groupby(groups_series)[col].max()
        else:
            raise ValueError(f"Invalid aggregation method at index {idx}: {curr_aggregation_method}")
    
    # Reverse the order to have chronological order (earliest group first)
    datetime_agg_df = datetime_agg_df.iloc[::-1].reset_index(drop=True)
    numeric_agg_df = numeric_agg_df.iloc[::-1].reset_index(drop=True)
    
    # Adjust start_date for incomplete first group to maintain correct duration
    if not first_group_complete and len(datetime_agg_df) > 0:
        datetime_agg_df.loc[0, 'start_date'] = datetime_agg_df.loc[0, 'end_date'] - delta_time

    if len(datetime_agg_df) != len(numeric_agg_df):
        raise ValueError("Number of rows in datetime dataframe and numeric dataframe do not match")
    
    return datetime_agg_df, numeric_agg_df

# %% === Method to load time-sensitive scattered data in csv format
def load_time_sensitive_data_from_csv(
        file_path: str,
        fill_method: str | int | float=None,
        round_datetime: bool=True,
        delta_time_hours: float | int=None,
        aggregation_method: list[str]=['sum'],
        last_date: pd.Timestamp=None,
        datetime_columns_names: list[str]=['start_date', 'end_date'],
        numeric_columns_names: list[str]=None
    ) -> pd.DataFrame:
    """
    Load time-sensitive data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        fill_method (str, optional): The method to use for filling missing values (default: None).
            Possible values are
                0. None - Do not fill missing values.
                1. 'zero' - Fill missing values with 0.
                2. 'mean' - Fill missing values with the mean bewteen the first and last non-empty values.
                3. 'nearest' - Fill missing values with the nearest non-empty value.
                4. 'previous' - Fill missing values with the previous non-empty value.
                5. 'next' - Fill missing values with the next non-empty value.
                6. 'linear' - Fill missing values with the linear interpolation between the first and last non-empty values.
                7. 'quadratic' - Fill missing values with the quadratic interpolation between the first and last non-empty values.
                8. 'cubic' - Fill missing values with the cubic interpolation between the first and last non-empty values.
        round_datetime (bool, optional): If True, round datetime columns to the nearest minute (default: True).
        delta_time_hours (float, optional): The time interval in hours for aggregating data (default: None).
        aggregation_method (list[str], optional): A list of aggregation methods for aggregating data (default: ['sum']).
        last_date (pd.Timestamp, optional): The last date to include in the aggregation. 
            All dates greater than this will be filtered out. If None, use the last available date.
        datetime_columns_names (list[str], optional): A list of datetime column names (default: None).
        numeric_columns_names (list[str], optional): A list of numeric column names (default: None).

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        raise TypeError("file_path must be a string and the file must exist.")
    if not isinstance(datetime_columns_names, (list)):
        raise TypeError("datetime_names must be a list of strings.")
    if not isinstance(numeric_columns_names, (list, type(None))):
        raise TypeError("value_names must be a list of strings or None.")
    if not isinstance(fill_method, (str, int, float, type(None))):
        raise TypeError("fill_method must be a string, integer or None.")
    if not isinstance(round_datetime, bool):
        raise TypeError("round_datetime must be a boolean.")
    if not isinstance(delta_time_hours, (float, int, type(None))):
        raise TypeError("delta_time_hours must be a float, integer or None.")
    
    data_df = pd.read_csv(file_path)

    _, datetime_df, numeric_df = _parse_time_sensitive_dataframe(data_df, fill_method=fill_method, round_datetime=round_datetime)

    if len(datetime_df.columns) != 2:
        raise ValueError(f"Time-sensitive part of the csv has {len(datetime_df.columns)} columns. Expected 2 columns: start_date, end_date")
    
    if len(datetime_columns_names) != len(datetime_df.columns):
        raise ValueError(f"Time-sensitive part of the csv has {len(datetime_df.columns)} columns. Current datetime_columns_names list has {len(datetime_columns_names)} elements!")

    datetime_df.columns = datetime_columns_names

    if datetime_df.iloc[0,1] < datetime_df.iloc[0,0]:
        raise ValueError("Start date of the time-sensitive data is greater than end date.")

    diff_end_start = datetime_df[datetime_columns_names[1]] - datetime_df[datetime_columns_names[0]]
    not_uniform_row_ids = diff_end_start[diff_end_start != diff_end_start.iloc[1]].index.to_list()
    if not_uniform_row_ids:
        raise ValueError("Duration of the time-sensitive data (end-start) is not uniform. Non-uniform rows: " + str(not_uniform_row_ids))

    if numeric_columns_names:
        if len(numeric_columns_names) != len(numeric_df.columns):
            raise ValueError(f"Numeric part of the csv has {len(numeric_df.columns)} columns. Current numeric_columns_names list has {len(numeric_columns_names)} elements!")
        numeric_df.columns = numeric_columns_names

    if delta_time_hours:
        datetime_df, numeric_df = _change_time_sensitive_dataframe_resolution(
            numeric_df=numeric_df,
            datetime_df=datetime_df,
            delta_time=pd.Timedelta(hours=delta_time_hours),
            aggregation_method=aggregation_method,
            last_date=last_date
        )
    
    data_df = pd.concat([datetime_df, numeric_df], axis=1)
    
    return data_df

# %% === Method to detect column type of gauges table
def _advanced_column_detection(
        series: pd.Series
    ) -> dict:
    """
    Advanced column detection using multiple heuristics.
    Returns confidence scores for different column types.

    Args:
        series (pd.Series): The column to detect from a dataframe.

    Returns:
        dict: A dictionary containing confidence scores for different column types.
    """
    name_lower = series.name.lower()
    sample_data = series.to_list()
    results = {
        'station': 0.0,
        'latitude': 0.0,
        'longitude': 0.0,
        'altitude': 0.0
    }
    
    # Pattern matching (multilingual)
    patterns = {
        'station': ['stat', 'staz', 'station', 'gauge', 'id', 'stazione', 'poste'],
        'latitude': ['lat', 'latitude', 'latitud', 'breite'],
        'longitude': ['lon', 'long', 'longitude', 'länge', 'longitud'],
        'altitude': ['alt', 'elev', 'height', 'quota', 'höhe', 'altura']
    }
    
    for col_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if pattern in name_lower:
                results[col_type] += 0.35
    
    # Data type analysis
    numeric_count = sum(1 for x in sample_data if isinstance(x, (int, float)))
    string_count = sum(1 for x in sample_data if isinstance(x, str))
    
    if numeric_count > string_count:
        results['station'] -= 0.2
    else:
        results['station'] += 0.2
        results['latitude'] -= 0.1
        results['longitude'] -= 0.1
        results['altitude'] -= 0.1
    
    # Value range analysis (for numeric columns)
    if numeric_count > 0:
        numeric_values = [x for x in sample_data if isinstance(x, (int, float))]
        min_val, max_val = min(numeric_values), max(numeric_values)
        
        # Latitude typically between -90 and 90
        if -90 <= min_val <= 90 and -90 <= max_val <= 90:
            results['latitude'] += 0.4
        
        # Longitude typically between -180 and 180
        if -180 <= min_val <= 180 and -180 <= max_val <= 180:
            results['longitude'] += 0.4
        
        # Altitude typically positive
        if min_val >= 0:
            results['altitude'] += 0.2
    
    return results

# %% === Method to detect column type smartly
def _smart_column_detection(
        df: pd.DataFrame, 
        target_columns: list[str]=None,
        confidence_threshold: float=0.5
    ) -> tuple[dict, dict]:
    """
    Smart column detection with fallback strategies.

    Args:
        df (pd.DataFrame): The DataFrame to detect columns from.
        target_columns (list): The target columns to detect.
        confidence_threshold (float): The confidence threshold for column detection.

    Returns:
        tuple(dict, dict): A tuple containing the detected columns and their confidence scores.
    """
    detected = {}
    confidence_scores = {}
    
    for col in df.columns:
        scores = _advanced_column_detection(df[col])
        
        for col_type, score in scores.items():
            if score > confidence_threshold:
                if col_type not in detected or score > confidence_scores.get(col_type, 0):
                    detected[col_type] = col
                    confidence_scores[col_type] = score
    
    # Fallback: if not detected, use the first column that is compatible
    if target_columns:
        for col_type in target_columns:
            if col_type not in detected:
                for col in df.columns:
                    sample_data = df[col].tolist()
                    if col_type == 'station':
                        compatible = all(isinstance(x, str) for x in sample_data)
                    else:
                        compatible = all(isinstance(x, (int, float)) for x in sample_data)
                    
                    if compatible and col not in detected.values():
                        detected[col_type] = col
                        confidence_scores[col_type] = 0.0
                        warnings.warn(f"Fallback: using column [{col}] as {col_type} column.", stacklevel=2)
                        break
    
    return detected, confidence_scores

# %% === Method to load gauges table in csv format
def load_time_sensitive_gauges_from_csv(
        file_path: str,
        station_column: str | int=None,
        longitude_column: str | int=None,
        latitude_column: str | int=None,
        altitude_column: str | int=None
    ) -> pd.DataFrame:
    """
    Load gauges table from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        station_column (str | int, optional): The name or index of the station column (default: None).
        longitude_column (str | int, optional): The name or index of the longitude column (default: None).
        latitude_column (str | int, optional): The name or index of the latitude column (default: None).
        altitude_column (str | int, optional): The name or index of the altitude column (default: None).

    Returns:
        pd.DataFrame: A DataFrame containing the loaded gauges table.
    """
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        raise TypeError("file_path must be a string and the file must exist.")
    if not isinstance(station_column, (str, int, type(None))):
        raise TypeError("station_column must be a string, integer or None.")
    if not isinstance(longitude_column, (str, int, type(None))):
        raise TypeError("longitude_column must be a string, integer or None.")
    if not isinstance(latitude_column, (str, int, type(None))):
        raise TypeError("latitude_column must be a string, integer or None.")
    if not isinstance(altitude_column, (str, int, type(None))):
        raise TypeError("altitude_column must be a string, integer or None.")
    
    station_df = pd.read_csv(file_path)

    auto_col_detection, _ = _smart_column_detection(station_df, ['station', 'longitude', 'latitude', 'altitude'])

    if isinstance(station_column, int):
        station_column = station_df.columns[station_column]
    elif isinstance(station_column, str):
        if station_column not in station_df.columns:
            raise ValueError(f"Column [{station_column}] not found in the DataFrame.")
    elif isinstance(station_column, type(None)):
        station_column = auto_col_detection['station']
        warnings.warn(f"Auto detection: using column [{col}] as the station column.", stacklevel=2)
    
    if isinstance(longitude_column, int):
        longitude_column = station_df.columns[longitude_column]
    elif isinstance(longitude_column, str):
        if longitude_column not in station_df.columns:
            raise ValueError(f"Column [{longitude_column}] not found in the DataFrame.")
    elif isinstance(longitude_column, type(None)):
        longitude_column = auto_col_detection['longitude']
        warnings.warn(f"Auto detection: using column [{col}] as the longitude column.", stacklevel=2)
    
    if isinstance(latitude_column, int):
        latitude_column = station_df.columns[latitude_column]
    elif isinstance(latitude_column, str):
        if latitude_column not in station_df.columns:
            raise ValueError(f"Column [{latitude_column}] not found in the DataFrame.")
    elif isinstance(latitude_column, type(None)):
        latitude_column = auto_col_detection['latitude']
        warnings.warn(f"Auto detection: using column [{col}] as the latitude column.", stacklevel=2)
    
    if isinstance(altitude_column, int):
        altitude_column = station_df.columns[altitude_column]
    elif isinstance(altitude_column, str):
        if altitude_column not in station_df.columns:
            raise ValueError(f"Column [{altitude_column}] not found in the DataFrame.")
    elif isinstance(altitude_column, type(None)):
        altitude_column = auto_col_detection['altitude']
        warnings.warn(f"Auto detection: using column [{col}] as the altitude column.", stacklevel=2)

    selected_columns = list(dict.fromkeys([station_column, longitude_column, latitude_column, altitude_column])) # Remove duplicates while preserving order
    if len(selected_columns) != 4:
        raise ValueError("The gauges table must have 4 columns: station, longitude, latitude and altitude.")

    station_df = station_df[selected_columns]
    station_df.columns = ["station", "longitude", "latitude", "altitude"]
    
    return station_df

# %% === Method to merge and align time-sensitive data and gauges table
def merge_scattered_time_sensitive_data(
        time_sensitive_data: list[pd.DataFrame],
        time_sensitive_gauges: list[str],
        gauges_table: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
    """
    Merge and align time-sensitive data with gauges table.

    Args:
        time_sensitive_data (list[pd.DataFrame]): A list of DataFrames containing the time-sensitive data.
        time_sensitive_gauges (list[str]): A list of gauge names.
        gauges_table (pd.DataFrame): A DataFrame containing the gauges table.

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping gauge names to aligned DataFrames data.
    """