# %% === Import necessary modules
import warnings
import pandas as pd
import numpy as np
import os
from dateutil import parser
import re

# %% === Helper method to infer date format
def _infer_date_format(
        series: pd.Series, 
        allow_multiple_formats: bool = False,
        samples_to_test: int = 50
    ) -> str:
    """
    Infer the date format of a pandas Series.
    
    Args:
        series (pd.Series): The pandas Series to infer the date format from.
        allow_multiple_formats (bool, optional): Whether to allow multiple date formats (default: False).
        
    Returns:
        str: The inferred date format.
    """
    date_format_not_found_string = "Unable to infer date format"
    try:
        if allow_multiple_formats:
            # Try to parse dates using dateutil parser
            try:
                series.head(samples_to_test).apply(lambda x: parser.parse(x))
                return 'dateutil'
            except ValueError:
                raise ValueError(date_format_not_found_string)
        else:
            formats = set()
            date_sep_chars = r'[-/.]' # Possible characters to separate date are [-], [/], or [.]
            time_sep_chars = r'[:.]' # Possible characters to separate time are [:], or [.]

            date_pattern = re.compile(rf'(\d{{4}}{date_sep_chars}\d{{1,2}}{date_sep_chars}\d{{1,2}}|\d{{1,2}}{date_sep_chars}\d{{1,2}}{date_sep_chars}\d{{4}}|\d{{8}})')
            time_pattern = re.compile(rf'(\d{{1,2}}{time_sep_chars}\d{{1,2}}{time_sep_chars}\d{{1,2}}|\d{{1,2}}{time_sep_chars}\d{{1,2}})')
            
            for full_date_str in series.head(samples_to_test):
                if len(full_date_str) not in [8, 10, 16, 19]: # YYYYMMDD, YYYY-MM-DD, YYYY-MM-DD HH:MM, YYYY-MM-DD HH:MM:SS
                    raise ValueError(date_format_not_found_string)
                
                date_match = date_pattern.search(full_date_str)
                time_match = time_pattern.search(full_date_str)
                
                if date_match:
                    date_str = date_match.group()
                    date_sep = [x for x in date_str if not x.isdigit()]
                    if len(date_sep) == 0 and len(date_str) == 8: # No separators
                        if int(date_str[:4]) > 1900 and 1 <= int(date_str[4:6]) <= 31 and 1 <= int(date_str[6:]) <= 31: # Probably starts with year
                            if date_str[4:6] > '12': # Day is in the middle
                                date_format = ['%Y%d%m']
                            elif date_str[6:] > '12': # Day is at the end
                                date_format = ['%Y%m%d']
                            else: # Not sure if day is in the middle or at the end, include both
                                date_format = [
                                    '%Y%d%m',
                                    '%Y%m%d'
                                ]
                        elif int(date_str[4:]) > 1900 and 1 <= int(date_str[0:2]) <= 31 and 1 <= int(date_str[2:4]) <= 31: # probably ends with year
                            if date_str[0:2] > '12': # Day is at the beginning
                                date_format = ['%d%m%Y']
                            elif date_str[2:4] > '12': # Day is in the middle
                                date_format = ['%m%d%Y']
                            else: # Not sure if day is at the beginning or in the middle, include both
                                date_format = [
                                    '%d%m%Y',
                                    '%m%d%Y'
                                ]
                        else:
                            raise ValueError(f"{date_format_not_found_string}. Date string with 8 characters: {date_str}")
                    
                    elif len(date_sep) == 2 and len(date_str) == 10: # Separators (which can be different, in case of user inconsistency)
                        date_splitted = date_str.replace(date_sep[1], date_sep[0]).split(date_sep[0])
                        if len(date_splitted[0]) == 4: # Starts with year
                            if int(date_splitted[1]) > 12: # Day is in the middle
                                date_format = [f'%Y{date_sep[0]}%d{date_sep[1]}%m']
                            elif int(date_splitted[2]) > 12: # Day is at the end
                                date_format = [f'%Y{date_sep[0]}%m{date_sep[1]}%d']
                            else: # Not sure if day is in the middle or at the end, include both
                                date_format = [
                                    f'%Y{date_sep[0]}%d{date_sep[1]}%m',
                                    f'%Y{date_sep[0]}%m{date_sep[1]}%d'
                                ]
                        elif len(date_splitted[2]) == 4: # Ends with year
                            if int(date_splitted[0]) > 12: # Day is at the beginning
                                date_format = [f'%d{date_sep[0]}%m{date_sep[1]}%Y']
                            elif int(date_splitted[1]) > 12: # Day is in the middle
                                date_format = [f'%m{date_sep[0]}%d{date_sep[1]}%Y']
                            else: # Not sure if day is at the beginning or in the middle, include both
                                date_format = [
                                    f'%d{date_sep[0]}%m{date_sep[1]}%Y',
                                    f'%m{date_sep[0]}%d{date_sep[1]}%Y'
                                ]
                        else:
                            raise ValueError(f"{date_format_not_found_string}. Date string with 10 characters: {date_str}")
                        
                    else:
                        raise ValueError(f"{date_format_not_found_string}. Date string has not 8 or 10 characters: {full_date_str}")
                    
                    if time_match:
                        time_str = time_match.group()
                        time_sep = [x for x in time_str if not x.isdigit()]
                        if len(time_sep) == 1: # Found just hour and minute
                            time_format = f'%H{time_sep[0]}%M'
                        elif len(time_sep) == 2: # Found hour, minute and second
                            time_format = f'%H{time_sep[0]}%M{time_sep[1]}%S'
                        else:
                            raise ValueError(f"{date_format_not_found_string}. Time string with 0 or more than 2 separators: {time_str}")
                        
                        for curr_dt_format in date_format:
                            if date_match.start() < time_match.start():
                                date_time_separator = full_date_str[date_match.end():time_match.start()]
                                formats.add(f'{curr_dt_format}{date_time_separator}{time_format}')
                            else:
                                date_time_separator = full_date_str[time_match.end():date_match.start()]
                                formats.add(f'{time_format}{date_time_separator}{curr_dt_format}')
                    else:
                        for curr_dt_format in date_format:
                            formats.add(curr_dt_format)
                else:
                    raise ValueError(f"{date_format_not_found_string}. No date match found: {full_date_str}")
            
            if len(formats) == 1:
                return list(formats)[0]
            else:
                # Create compatibility matrix
                compatibility_matrix = np.zeros((len(series.head(samples_to_test)), len(formats)), dtype='bool')
                formats_list = list(formats)
                
                for i, full_date_str in enumerate(series.head(samples_to_test)):
                    for j, date_format in enumerate(formats_list):
                        try:
                            pd.to_datetime(full_date_str, format=date_format)
                            compatibility_matrix[i, j] = True
                        except ValueError:
                            pass
                
                compatible_formats = np.all(compatibility_matrix, axis=0)
                compatible_formats_count = np.sum(compatible_formats)
                
                if compatible_formats_count == 1:
                    return formats_list[np.argmax(compatible_formats)]
                elif compatible_formats_count > 1:
                    compatible_formats_list = [formats_list[i] for i, x in enumerate(compatible_formats) if x]
                    raise ValueError(f"Multiple compatible date formats detected: {compatible_formats_list}")
                else:
                    raise ValueError("No compatible date format found")
    
    except Exception as e:
        raise ValueError(f"Error inferring date format: {str(e)}")

# %% === Helper method to parse date with inferred format
def _parse_date_with_inferred_format(
        date_str: str, 
        date_format: str, 
        fuzzy: bool = False
    ) -> pd.Timestamp:
    """
    Parse a date string using the inferred date format.
    
    Args:
        date_str (str): The date string to parse.
        date_format (str): The inferred date format.
        fuzzy (bool, optional): Whether to use fuzzy parsing (default: False).
        
    Returns:
        pd.Timestamp: The parsed date.
    """
    if date_format == 'dateutil':
        return parser.parse(date_str, fuzzy=fuzzy)
    else:
        return pd.to_datetime(date_str, format=date_format)

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
        round_datetime: bool=True,
        allow_multiple_date_formats: bool=False,
        force_datetime_consistency: bool=False
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
        allow_multiple_date_formats (bool, optional): If True, allow multiple date formats in the same column (default: False).

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
                # Infer the date format
                date_format = _infer_date_format(data_df[col], allow_multiple_date_formats)
                data_df[col] = data_df[col].apply(lambda x: _parse_date_with_inferred_format(x, date_format, fuzzy=True))
            except:
                pass
        elif pd.api.types.is_numeric_dtype(col_type):
            try:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce', downcast='infer')
            except:
                pass
        else:
            pass

    for col in data_df.columns:
        if pd.api.types.is_numeric_dtype(data_df[col].dtype):
            pre_data_type = data_df[col].dtype
            missing_row_ids = data_df[data_df[col].isnull()].index.to_list()
            if len(missing_row_ids) > 0 and fill_method is not None:
                data_df[col] = _fill_missing_values_of_numeric_series(
                    column_data=data_df[col], 
                    fill_method=fill_method
                )
                
                warnings.warn(
                    f"Missing values in column [{col}] at rows {[x + 2 for x in missing_row_ids]} have been filled with {fill_method} mode", # + 2 because the csv first row is for header, plus, the index should start from 1, not 0!
                    stacklevel=2
                )
            
            data_df[col] = data_df[col].astype(pre_data_type)

        if pd.api.types.is_datetime64_any_dtype(data_df[col]):
            missing_row_ids = data_df[data_df[col].isnull()].index.to_list()
            if len(missing_row_ids) > 0:
                if data_df[col].dtype == 'object':
                    data_df[col] = data_df[col].infer_objects(copy=False) # Suggested to convert first to infer_objects
                data_df[col] = data_df[col].interpolate(method='linear')

                warnings.warn(
                    f"Missing values in column [{col}] at rows {missing_row_ids} (+2 if you are looking into csv) have been filled with linear interpolation", # + 2 because the csv first row is for header and should start from 1, not 0!
                    stacklevel=2
                )
            
            if round_datetime:
                data_df[col] = data_df[col].dt.round('1min') # Round to nearest minute

            delta_time_hours_num = data_df[col].diff().dt.total_seconds() / 3600
            non_uniform_rows = delta_time_hours_num[delta_time_hours_num != delta_time_hours_num.iloc[1]].index.to_list()
            non_uniform_rows.remove(0) # Remove index 0 because it is the first row and it is always missing
            if len(non_uniform_rows) >= 1: # > 1 because the first row is always missing
                warn_main_msg = f"Time column [{col}] is not uniform. Non-uniform rows (+2 if you are looking into csv): {non_uniform_rows}."
                warn_add_msg = "Since force_datetime_consistency is True, the missing rows will be tried to be added and the method will be run again."
                if force_datetime_consistency:
                    warnings.warn(f"{warn_main_msg} {warn_add_msg}", stacklevel=2)

                    min_delta_time_hours_num = delta_time_hours_num.min()
                    min_delta_time = pd.Timedelta(hours=min_delta_time_hours_num)
                    new_data_df = pd.DataFrame(columns=data_df.columns).astype(data_df.dtypes.to_dict())
                    new_data_df = new_data_df.astype({col: 'Int64' for col in new_data_df.select_dtypes(include=['int64']).columns}) # To maintain int64 type with missing values
                    min_datetime = data_df[col].min()
                    max_datetime = data_df[col].max()
                    new_datetime_series = pd.date_range(start=min_datetime, end=max_datetime, freq=min_delta_time)
                    new_data_df[col] = new_datetime_series

                    for new_idx in range(len(new_datetime_series)):
                        closest_idx = np.where( (data_df[col] - new_data_df[col].iloc[new_idx]).abs() < min_delta_time/2 )[0]
                        if len(closest_idx) == 1:
                            new_data_df.iloc[new_idx] = data_df.iloc[closest_idx.item()]
                    
                    return _parse_time_sensitive_dataframe(
                        data_df=new_data_df,
                        fill_method=fill_method,
                        round_datetime=round_datetime,
                        allow_multiple_date_formats=allow_multiple_date_formats,
                        force_datetime_consistency=False
                    )

                else:
                    raise ValueError(f"{warn_main_msg}")

            # Handle timezone-aware datetime conversion
            if data_df[col].dt.tz is not None:
                data_df[col] = data_df[col].dt.tz_localize(None)
            
            data_df[col] = data_df[col].astype('datetime64[ns]')
    
    datetime_df = data_df.select_dtypes(include=['datetime64[ns]'])
    numeric_df = data_df.select_dtypes(include=['float64', 'int64'])

    if len(datetime_df.columns) != 1 and len(datetime_df.columns) != 2:
        raise ValueError("Expected one or two datetime columns in the dataframe (just start, or end, or start + end dates)")

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

# %% === Helper method to detect column type of gauges table
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
    series = series.copy()
    name_lower = series.name.lower()
    sample_data = series.to_list()
    
    # Enhanced pattern matching with expected data types
    patterns = {
        'station': {
            'patterns': ['stat', 'staz', 'gauge', 'id', 'poste'],
            'expected_type': 'string'
        },
        'numeric_station_id': {
            'patterns': ['stat', 'staz', 'gauge', 'id', 'poste'],
            'expected_type': 'numeric'
        },
        'latitude': {
            'patterns': ['lat', 'latit', 'breite'],
            'expected_type': 'numeric',
            'range': (-90, 90)
        },
        'longitude': {
            'patterns': ['lon', 'long', 'länge'],
            'expected_type': 'numeric',
            'range': (-180, 180)
        },
        'altitude': {
            'patterns': ['alt', 'elev', 'height', 'quota', 'höhe'],
            'expected_type': 'numeric',
            'range': (0, 3000)  # Positive values
        },
        'start_date': {
            'patterns': ['start', 'inizio', 'begin', 'debut', 'anfang', 'inicio', 'from', 'dat'],
            'expected_type': 'datetime'
        },
        'end_date': {
            'patterns': ['end', 'fin', 'to', 'dat'],
            'expected_type': 'datetime'
        },
        'cumulative_rain': {
            'patterns': ['cum', 'tot', 'somma', 'sum', 'rain', 'pioggia', 'regen', 'lluvia'],
            'expected_type': 'numeric',
            'range': (0, 1000)  # Reasonable rain values
        },
        'min_cumulative_rain': {
            'patterns': ['min', 'mín', 'niedrigst', 'cum', 'somma', 'sum', 'rain', 'pioggia', 'regen', 'lluvia'],
            'expected_type': 'numeric',
            'range': (0, 1000)  # Reasonable rain values
        },
        'max_cumulative_rain': {
            'patterns': ['max', 'mín', 'niedrigst', 'mas', 'máx', 'höchst', 'cum', 'somma', 'sum', 'rain', 'pioggia', 'regen', 'lluvia'],
            'expected_type': 'numeric',
            'range': (0, 1000)  # Reasonable rain values
        },
        'peak_rain': {
            'patterns': ['pk', 'peak', 'max', 'mas', 'máx', 'höchst', 'picco', 'rain', 'pioggia', 'regen', 'lluvia'],
            'expected_type': 'numeric',
            'range': (0, 1000)  # Reasonable rain values
        },
        'average_temperature': {
            'patterns': ['avg', 'aver', 'med', 'durchschnitt', 'promedio', 'temp'],
            'expected_type': 'numeric',
            'range': (-50, 50)  # Reasonable temperature range
        },
        'maximum_temperature': {
            'patterns': ['max', 'mas', 'máx', 'höchst', 'temp'],
            'expected_type': 'numeric',
            'range': (-50, 50)  # Reasonable temperature range
        },
        'minimum_temperature': {
            'patterns': ['min', 'mín', 'niedrigst', 'temp'],
            'expected_type': 'numeric',
            'range': (-50, 50)  # Reasonable temperature range
        }
    }

    results = {x: 0.0 for x in patterns.keys()}
    
    LOAD_BUFFER_SIZE = 50
    BASE_PATTERN_MATCH_SCORE = 0.3
    ADDITIONAL_PATTERN_MATCH_SCORE = 0.15
    TYPE_MATCH_SCORE = 0.3
    RANGE_MATCH_SCORE = 0.3
    
    # Enhanced type analysis with datetime conversion attempt
    datetime_count = 0
    numeric_count = 0
    string_count = 0
    
    # Try to convert to datetime first - only if values are strings
    datetime_converted = []
    try:
        # Only attempt datetime conversion if we have string values
        sample_head = series.head(LOAD_BUFFER_SIZE)
        if sample_head.dtype == 'object' or any(isinstance(x, str) for x in sample_head):
            datetime_test = pd.to_datetime(sample_head, errors='coerce')
            datetime_count = datetime_test.notna().sum()
            datetime_converted = datetime_test.tolist()
    except:
        pass
    
    # Count remaining types
    for i, value in enumerate(sample_data[:LOAD_BUFFER_SIZE]):  # Check first LOAD_BUFFER_SIZE values
        if datetime_count > 0 and i < len(datetime_converted) and not pd.isna(datetime_converted[i]):
            # This value was successfully converted to datetime
            continue
        elif isinstance(value, (int, float)):
            numeric_count += 1
        elif isinstance(value, str):
            string_count += 1
    
    # Determine dominant type
    type_counts = {
        'datetime': datetime_count,
        'numeric': numeric_count,
        'string': string_count
    }
    dominant_type = max(type_counts, key=type_counts.get)
    is_datetime_dominant = dominant_type == 'datetime'
    is_numeric_dominant = dominant_type == 'numeric'
    is_string_dominant = dominant_type == 'string'
    
    # Unified pattern and type analysis
    for col_type, pattern_info in patterns.items():
        pattern_list = pattern_info['patterns']
        expected_type = pattern_info['expected_type']
        
        # Pattern matching
        pattern_count = sum(1 for pattern in pattern_list if pattern in name_lower)
        if pattern_count > 0:
            results[col_type] += BASE_PATTERN_MATCH_SCORE + (pattern_count - 1) * ADDITIONAL_PATTERN_MATCH_SCORE
        
        # Enhanced type compatibility check
        if expected_type == 'string':
            if is_string_dominant:
                results[col_type] += TYPE_MATCH_SCORE
            else:
                results[col_type] -= TYPE_MATCH_SCORE
        elif expected_type == 'datetime':
            if is_datetime_dominant:
                results[col_type] += TYPE_MATCH_SCORE
            elif is_string_dominant:
                results[col_type] += TYPE_MATCH_SCORE / 3  # Partial credit for string (could be unconverted datetime)
            else:
                results[col_type] -= TYPE_MATCH_SCORE
        elif expected_type == 'numeric':
            if is_numeric_dominant:
                results[col_type] += TYPE_MATCH_SCORE
            else:
                results[col_type] -= TYPE_MATCH_SCORE
        else:
            raise ValueError(f"Unknown expected type: {expected_type}")
        
        # Range validation for numeric columns (only if numeric dominant)
        if expected_type == 'numeric' and is_numeric_dominant and 'range' in pattern_info:
            # Get numeric values from the full sample (not just first 50)
            numeric_values_full = [x for x in sample_data if isinstance(x, (int, float)) and not pd.isna(x)]
            if numeric_values_full:
                min_val, max_val = min(numeric_values_full), max(numeric_values_full)
                range_min, range_max = pattern_info['range']
                in_range = True
                
                if range_min is not None and min_val < range_min:
                    in_range = False
                if range_max is not None and max_val > range_max:
                    in_range = False
                
                if in_range:
                    results[col_type] += RANGE_MATCH_SCORE
    
    return results

# %% === Helper method to perform a cross-validation of detection and scores
def _cross_columns_check(
        df: pd.DataFrame,
        detected: dict,
        confidence_scores: dict,
        all_scores: dict
    ) -> tuple[dict, dict]:
    """
    Perform a cross-validation of detection and scores.

    Args:
        df (pd.DataFrame): The DataFrame to detect columns from.
        detected (dict): The detected columns (keys: column names, values: detected types).
        confidence_scores (dict): The confidence scores for detected columns.
        all_scores (dict): All scores for all columns.

    Returns:
        tuple(dict, dict): A tuple containing the detected columns and their confidence scores, eventually swapped.
    """
    # Create reverse mapping for easier access (only for assigned columns)
    type_to_column = {col_type: col for col, col_type in detected.items() if col_type is not None}
    
    # Cross-column relationship analysis - re-evaluate assignments if needed
    if len(type_to_column) >= 2:
        # Temperature relationships: max > avg > min (check majority of values)
        if 'maximum_temperature' in type_to_column and 'average_temperature' in type_to_column:
            max_col = type_to_column['maximum_temperature']
            avg_col = type_to_column['average_temperature']
            
            # Check majority of values (at least 70% should satisfy the relationship)
            valid_comparisons = (df[max_col] > df[avg_col]).dropna()
            if len(valid_comparisons) > 0 and valid_comparisons.mean() < 0.7:
                # Relationship violation: max should be greater than avg for majority
                detected[max_col] = 'average_temperature'
                detected[avg_col] = 'maximum_temperature'
                confidence_scores[max_col] = all_scores[avg_col]['average_temperature']
                confidence_scores[avg_col] = all_scores[max_col]['maximum_temperature']
                warnings.warn(f"Swapped temperature columns based on value relationships: {max_col} <-> {avg_col}", stacklevel=2)
        
        if 'average_temperature' in type_to_column and 'minimum_temperature' in type_to_column:
            avg_col = type_to_column['average_temperature']
            min_col = type_to_column['minimum_temperature']
            
            # Check majority of values
            valid_comparisons = (df[avg_col] > df[min_col]).dropna()
            if len(valid_comparisons) > 0 and valid_comparisons.mean() < 0.7:
                # Relationship violation: avg should be greater than min for majority
                detected[avg_col] = 'minimum_temperature'
                detected[min_col] = 'average_temperature'
                confidence_scores[avg_col] = all_scores[min_col]['minimum_temperature']
                confidence_scores[min_col] = all_scores[avg_col]['average_temperature']
                warnings.warn(f"Swapped temperature columns based on value relationships: {avg_col} <-> {min_col}", stacklevel=2)
        
        # Date relationships: end_date > start_date (check majority of values)
        if 'start_date' in type_to_column and 'end_date' in type_to_column:
            start_col = type_to_column['start_date']
            end_col = type_to_column['end_date']
            
            # Try to convert to datetime first, don't just check dtype
            start_series = pd.to_datetime(df[start_col], errors='coerce')
            end_series = pd.to_datetime(df[end_col], errors='coerce')
            
            # Only check if we have valid datetime conversions
            valid_mask = start_series.notna() & end_series.notna()
            if valid_mask.sum() > 0:  # At least some valid datetime pairs
                # Check majority of valid values
                valid_comparisons = (end_series[valid_mask] > start_series[valid_mask])
                if len(valid_comparisons) > 0 and valid_comparisons.mean() < 0.7:
                    # Relationship violation: end_date should be after start_date for majority
                    detected[start_col] = 'end_date'
                    detected[end_col] = 'start_date'
                    confidence_scores[start_col] = all_scores[end_col]['end_date']
                    confidence_scores[end_col] = all_scores[start_col]['start_date']
                    warnings.warn(f"Swapped date columns based on temporal relationships: {start_col} <-> {end_col}", stacklevel=2)
        
        # Rain relationships: peak rain >= cumulative rain (check majority of values)
        if 'peak_rain' in type_to_column and 'cumulative_rain' in type_to_column:
            peak_col = type_to_column['peak_rain']
            cumul_col = type_to_column['cumulative_rain']
            
            # Check majority of values
            valid_comparisons = (df[peak_col] >= df[cumul_col]).dropna()
            if len(valid_comparisons) > 0 and valid_comparisons.mean() < 0.7:
                # Relationship violation: peak should be >= cumulative for majority
                detected[peak_col] = 'cumulative_rain'
                detected[cumul_col] = 'peak_rain'
                confidence_scores[peak_col] = all_scores[cumul_col]['cumulative_rain']
                confidence_scores[cumul_col] = all_scores[peak_col]['peak_rain']
                warnings.warn(f"Swapped rain columns based on value relationships: {peak_col} <-> {cumul_col}", stacklevel=2)
    
    return detected, confidence_scores

# %% === Helper method to detect column type smartly
def _smart_columns_detection(
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
    df = df.copy()
    detected = {}  # Keys: column names, Values: detected type or None
    scores = {}  # Keys: column names, Values: confidence score for detected type
    all_scores = {}  # Store scores for all columns to allow re-evaluation
    
    # Initialize detected and confidence_scores for all columns
    for col in df.columns:
        detected[col] = None
        scores[col] = 0.0
    
    # First pass: collect scores for all columns
    for col in df.columns:
        scores_results = _advanced_column_detection(series=df[col])
        all_scores[col] = scores_results
        
        # Find the best type for this column
        best_type = None
        best_score = 0.0
        
        for col_type, score in scores_results.items():
            if score > confidence_threshold and score > best_score:
                best_type = col_type
                best_score = score
        
        if best_type:
            detected[col] = best_type
            scores[col] = best_score
    
    # Resolve conflicts: if multiple columns are assigned to the same type,
    # keep only the one with highest score
    type_to_column = {}
    for col, col_type in detected.items():
        if col_type is not None:  # Only consider columns with detected types
            if col_type not in type_to_column:
                type_to_column[col_type] = col  # Store only the column name
            else:
                # Compare scores between current assigned column and new column
                last_col_assigned = type_to_column[col_type]
                if scores[col] > scores[last_col_assigned]:
                    # New column has higher score, demote the current one
                    detected[last_col_assigned] = None
                    scores[last_col_assigned] = 0.0
                    type_to_column[col_type] = col  # Update to new column
                else:
                    # Current column has higher score, demote the new one
                    detected[col] = None
                    scores[col] = 0.0
    
    # Cross-column relationship check
    detected, scores = _cross_columns_check(df, detected, scores, all_scores)
    
    # Fallback: if target columns are specified and not detected, use compatible columns
    if target_columns:
        # Create reverse mapping from type to column
        type_assigned = {col_type: col for col, col_type in detected.items()}
        
        for col_type in target_columns:
            if col_type not in type_assigned:
                # Find compatible column that hasn't been assigned yet
                for col in df.columns:
                    if detected[col] is None:  # Only consider unassigned columns
                        sample_data = df[col].tolist()
                        
                        if col_type == 'station':
                            compatible = all(isinstance(x, str) for x in sample_data)
                        elif col_type in [
                            'latitude', 'longitude', 'altitude', 
                            'cumulative_rain', 'peak_rain', 
                            'average_temperature', 'maximum_temperature', 'minimum_temperature']:
                            compatible = all(isinstance(x, (int, float)) for x in sample_data)
                        elif col_type in ['start_date', 'end_date']:
                            # For dates, check if they can be converted to datetime
                            try:
                                datetime_test = pd.to_datetime(df[col].head(20), errors='coerce')
                                compatible = datetime_test.notna().sum() > 10  # At least 10 valid dates
                            except:
                                compatible = False
                        else:
                            compatible = False
                        
                        if compatible:
                            detected[col] = col_type
                            scores[col] = 0.0  # Fallback assignment has 0 confidence
                            warnings.warn(f"Fallback: using column [{col}] as {col_type} column.", stacklevel=2)
                            break
    
    return detected, scores

# %% === Method to load time-sensitive scattered data in csv format
def load_time_sensitive_data_from_csv(
        file_path: str,
        fill_method: str | int | float=None,
        round_datetime: bool=True,
        delta_time_hours: float | int=None,
        aggregation_method: list[str]=['sum'],
        last_date: pd.Timestamp=None,
        datetime_columns_names: list[str]=None,
        numeric_columns_names: list[str]=None,
        force_datetime_consistency: bool=False
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
        datetime_columns_names (list[str], optional): A list of datetime column names, which can be start_date or end_date, or both, in a list (default: None).
        numeric_columns_names (list[str], optional): A list of numeric column names (default: None).
        force_datetime_consistency (bool, optional): If True, force consistency of time series, which means that if datetime row is missing, 
            then it will be added to have consistency of deltatime and numeric columns will be filled with the fill_method (default: False).

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        raise TypeError("file_path must be a string and the file must exist.")
    if not isinstance(datetime_columns_names, (list, type(None))):
        raise TypeError("datetime_names must be a list of strings or None.")
    if not isinstance(numeric_columns_names, (list, type(None))):
        raise TypeError("value_names must be a list of strings or None.")
    if not isinstance(fill_method, (str, int, float, type(None))):
        raise TypeError("fill_method must be a string, integer or None.")
    if not isinstance(round_datetime, bool):
        raise TypeError("round_datetime must be a boolean.")
    if not isinstance(delta_time_hours, (float, int, type(None))):
        raise TypeError("delta_time_hours must be a float, integer or None.")
    if not isinstance(force_datetime_consistency, bool):
        raise TypeError("force_datetime_consistency must be a boolean.")
    
    raw_data_df = pd.read_csv(file_path, sep=None, engine='python') # Sometimes separators are not just commas (,) but maybe semicolons (;) or something else

    try:
        _, datetime_df, numeric_df = _parse_time_sensitive_dataframe(
            data_df=raw_data_df,
            fill_method=fill_method, 
            round_datetime=round_datetime,
            force_datetime_consistency=force_datetime_consistency
        )
    except Exception as e:
        raise ValueError(f"Error parsing the csv file [{file_path}]: {e}")

    if len(datetime_df.columns) == 1:
        warnings.warn(f"Datetime part of the csv has only one column. Assuming it is start_date or end_date depending on the column name. File: {file_path}", stacklevel=2)

    if len(datetime_df.columns) != 1 and len(datetime_df.columns) != 2:
        raise ValueError(f"Datetime part of the csv has {len(datetime_df.columns)} columns. Expected 1 or 2 columns: start_date, end_date, or both. File: {file_path}")

    if datetime_columns_names:
        if len(datetime_columns_names) != len(datetime_df.columns):
            raise ValueError(f"Datetime part of the csv has {len(datetime_df.columns)} columns. Current datetime_columns_names list has {len(datetime_columns_names)} elements! File: {file_path}")
        if not all([x in ['start_date', 'end_date'] for x in datetime_columns_names]):
            raise ValueError(f"datetime_columns_names must be 'start_date' or 'end_date'! File: {file_path}")
        datetime_df.columns = datetime_columns_names
    else:
        auto_datetime_col_detection, _ = _smart_columns_detection(df=datetime_df)
        if any([x is None for _, x in auto_datetime_col_detection.items()]):
            raise ValueError(f"Unable to detect datetime columns. Please provide datetime_columns_names manually. File: {file_path}")
        datetime_df.columns = [x for _, x in auto_datetime_col_detection.items()]
        warnings.warn(f"Auto detection: using columns {[x for x in auto_datetime_col_detection.keys()]} as {[x for x in auto_datetime_col_detection.values()]} columns. File: {file_path}", stacklevel=2)
    
    if len(datetime_df.columns) == 1:
        if datetime_df.columns[0] == 'start_date':
            datetime_df['end_date'] = datetime_df['start_date'] + datetime_df['start_date'].diff().mean(skipna=True)
        elif datetime_df.columns[0] == 'end_date':
            datetime_df['start_date'] = datetime_df['end_date'] - datetime_df['end_date'].diff().mean(skipna=True)
        else:
            raise ValueError(f"Datetime part of the csv has only one column and it is not start_date or end_date! Please use two columns or assign names to datetime_columns_names manually. File: {file_path}")

    if datetime_df.loc[0,'end_date'] < datetime_df.loc[0,'start_date']:
        raise ValueError(f"Start date is greater than end date. File: {file_path}")

    diff_end_start = datetime_df.loc[:,'end_date'] - datetime_df.loc[:,'start_date']
    not_uniform_row_ids = diff_end_start[diff_end_start != diff_end_start.iloc[1]].index.to_list()
    if not_uniform_row_ids:
        raise ValueError("Duration of the time-sensitive data (end-start) is not uniform. Non-uniform rows: " + str(not_uniform_row_ids) + f". File: {file_path}")

    if numeric_columns_names:
        if len(numeric_columns_names) != len(numeric_df.columns):
            raise ValueError(f"Numeric part of the csv has {len(numeric_df.columns)} columns. Current numeric_columns_names list has {len(numeric_columns_names)} elements! File: {file_path}")
        numeric_df.columns = numeric_columns_names
    else:
        auto_numeric_col_detection, _ = _smart_columns_detection(df=numeric_df)
        if any([x is None for _, x in auto_numeric_col_detection.items()]):
            raise ValueError(f"Unable to detect numeric columns. Please provide numeric_columns_names manually. File: {file_path}")
        numeric_df.columns = [x for _, x in auto_numeric_col_detection.items()]
        warnings.warn(f"Auto detection: using columns {[x for x in auto_numeric_col_detection.keys()]} as {[x for x in auto_numeric_col_detection.values()]} columns. File: {file_path}", stacklevel=2)

    if delta_time_hours:
        try:
            datetime_df, numeric_df = _change_time_sensitive_dataframe_resolution(
                numeric_df=numeric_df,
                datetime_df=datetime_df,
                delta_time=pd.Timedelta(hours=delta_time_hours),
                aggregation_method=aggregation_method,
                last_date=last_date
            )
        except Exception as e:
            raise ValueError(f"Error changing the time resolution of the csv file [{file_path}]: {e}")
    
    data_df = pd.concat([datetime_df, numeric_df], axis=1)
    
    return data_df

# %% === Method to load gauges table in csv format
def load_time_sensitive_stations_from_csv(
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
        pd.DataFrame: A DataFrame containing the table of gauges.
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
    
    station_df = pd.read_csv(file_path, sep=None, engine='python') # Sometimes the delimiter is not comma (,)

    auto_col_detection, _ = _smart_columns_detection(df=station_df, target_columns=['station', 'longitude', 'latitude', 'altitude'])
    
    # Create reverse mapping for easier access (only assigned columns)
    type_to_column = {col_type: col for col, col_type in auto_col_detection.items() if col_type is not None}

    if isinstance(station_column, int):
        station_column = station_df.columns[station_column]
    elif isinstance(station_column, str):
        if station_column not in station_df.columns:
            raise ValueError(f"Column [{station_column}] not found in the DataFrame. File: {file_path}")
    elif isinstance(station_column, type(None)):
        station_key = 'station'
        station_column = type_to_column.get(station_key)
        if station_column is None:
            raise ValueError(f"Station column not detected and no manual specification provided. File: {file_path}")
        warnings.warn(f"Auto detection: using column [{station_column}] as the [{station_key}] column. File: {file_path}", stacklevel=2)
    
    if isinstance(longitude_column, int):
        longitude_column = station_df.columns[longitude_column]
    elif isinstance(longitude_column, str):
        if longitude_column not in station_df.columns:
            raise ValueError(f"Column [{longitude_column}] not found in the DataFrame. File: {file_path}")
    elif isinstance(longitude_column, type(None)):
        longitude_key = 'longitude'
        longitude_column = type_to_column.get(longitude_key)
        if longitude_column is None:
            raise ValueError(f"Longitude column not detected and no manual specification provided. File: {file_path}")
        warnings.warn(f"Auto detection: using column [{longitude_column}] as the [{longitude_key}] column.", stacklevel=2)
    
    if isinstance(latitude_column, int):
        latitude_column = station_df.columns[latitude_column]
    elif isinstance(latitude_column, str):
        if latitude_column not in station_df.columns:
            raise ValueError(f"Column [{latitude_column}] not found in the DataFrame. File: {file_path}")
    elif isinstance(latitude_column, type(None)):
        latitude_key = 'latitude'
        latitude_column = type_to_column.get(latitude_key)
        if latitude_column is None:
            raise ValueError(f"Latitude column not detected and no manual specification provided. File: {file_path}")
        warnings.warn(f"Auto detection: using column [{latitude_column}] as the [{latitude_key}] column. File: {file_path}", stacklevel=2)
    
    if isinstance(altitude_column, int):
        altitude_column = station_df.columns[altitude_column]
    elif isinstance(altitude_column, str):
        if altitude_column not in station_df.columns:
            raise ValueError(f"Column [{altitude_column}] not found in the DataFrame. File: {file_path}")
    elif isinstance(altitude_column, type(None)):
        altitude_key = 'altitude'
        altitude_column = type_to_column.get(altitude_key)
        if altitude_column is None:
            raise ValueError(f"Altitude column not detected and no manual specification provided. File: {file_path}")
        warnings.warn(f"Auto detection: using column [{altitude_column}] as the [{altitude_key}] column. File: {file_path}", stacklevel=2)

    selected_columns = list(dict.fromkeys([station_column, longitude_column, latitude_column, altitude_column])) # Remove duplicates while preserving order
    if len(selected_columns) != 4:
        raise ValueError(f"The gauges table must have 4 columns: station, longitude, latitude and altitude. File: {file_path}")

    station_df = station_df[selected_columns]
    station_df.columns = ["station", "longitude", "latitude", "altitude"]
    
    return station_df

# %% === Method to merge and align time-sensitive data and gauges table
def merge_time_sensitive_data_and_stations(
        data_dict: dict[str, pd.DataFrame],
        stations_table: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
    """
    Merge and align time-sensitive data with gauges table.

    Args:
        data_dict (dict[str, pd.DataFrame]): A dictionary mapping gauge names to DataFrames data.
        stations_table (pd.DataFrame): A DataFrame containing the gauges table.

    Returns:
        dict[str, pd.DataFrame]: A dictionary containing structured, merged, and aligned data.
    """
    if not(isinstance(data_dict, dict)):
        raise TypeError("data_dict must be a dictionary.")
    if not(isinstance(stations_table, pd.DataFrame)):
        raise TypeError("stations_table must be a pandas DataFrame.")
    
    data_dict = data_dict.copy()
    stations_table = stations_table.copy()

    stations_table = stations_table[stations_table["station"].isin(list(data_dict.keys()))]

    station_names = stations_table["station"].to_list()
    for data_label in data_dict:
        if data_label not in station_names:
            raise ValueError(f"Data label [{data_label}] of input data_dict not found in the stations_table.")
    
    start_dates, end_dates = [], []
    for data in data_dict.values():
        start_dates.append(data['start_date'].iloc[0])
        end_dates.append(data['end_date'].iloc[-1])

    shared_start_date = max(start_dates)
    shared_end_date = min(end_dates)

    numeric_columns_names = []
    for data_label, data_df in data_dict.items():
        mask_dates = (data_df['start_date'] >= shared_start_date) & (data_df['end_date'] <= shared_end_date)
        if not mask_dates.all():
            if not mask_dates.any():
                raise ValueError(f"All data for gauge [{data_label}] is outside of the shared time period.")
            warnings.warn(f"Some data for gauge [{data_label}] is outside of the shared time period -> ignoring those data.", stacklevel=2)
        filtered_data_df = data_df[mask_dates].reset_index(drop=True)
        filtered_columns_names = filtered_data_df.columns[(filtered_data_df.columns != 'start_date') & (filtered_data_df.columns != 'end_date')].to_list()
        for filt_col_name in filtered_columns_names:
            if filt_col_name not in numeric_columns_names:
                numeric_columns_names.append(filt_col_name)
        data_dict[data_label] = filtered_data_df
    
    time_sensitive_vars = {'stations':stations_table, 'dates': None}
    for col_name in numeric_columns_names:
        time_sensitive_vars[col_name] = None
    for sta in station_names:
        curr_datetimes = data_dict[sta].loc[:, ['start_date', 'end_date']]
        if time_sensitive_vars['dates'] is None:
            time_sensitive_vars['dates'] = curr_datetimes
        else:
            start_diff = (time_sensitive_vars['dates']['start_date'] - curr_datetimes['start_date']).abs()
            end_diff = (time_sensitive_vars['dates']['end_date'] - curr_datetimes['end_date']).abs()
            if (start_diff > pd.Timedelta(seconds=1)).any():
                raise ValueError(f"Data for gauge [{sta}] has different start dates at indices : {start_diff[start_diff > pd.Timedelta(seconds=1)].index}, differing by {start_diff[start_diff > pd.Timedelta(seconds=1)]}.")
            if (end_diff > pd.Timedelta(seconds=1)).any():
                raise ValueError(f"Data for gauge [{sta}] has different end dates at indices : {end_diff[end_diff > pd.Timedelta(seconds=1)].index}, differing by {end_diff[end_diff > pd.Timedelta(seconds=1)]}.")
        for col_name in numeric_columns_names:
            if col_name in data_dict[sta].columns:
                curr_numeric = data_dict[sta].loc[:, [col_name]]
                curr_numeric.columns = [sta]
                if time_sensitive_vars[col_name] is None:
                    time_sensitive_vars[col_name] = curr_numeric
                else:
                    time_sensitive_vars[col_name] = pd.concat([time_sensitive_vars[col_name], curr_numeric], axis=1)
    return time_sensitive_vars

# %%