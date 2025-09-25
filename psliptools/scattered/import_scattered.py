# %% === Import necessary modules
import warnings
import numpy as np
import pandas as pd

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

# %% === Helper method to parse and validate a time-sensitive dataframe
def _parse_time_sensitive_dataframe(
        data_df: pd.DataFrame,
        fill_method: str | int=None,
        round_datetime: bool=True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse and validate a time-sensitive dataframe.

    Args:
        data_df (pd.DataFrame): The dataframe to parse and validate.
        fill_method (str, optional): The method to use for filling missing values (default: None).
            Possible values are
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
    
    if isinstance(fill_method, (int, float)):
        if fill_method == 1:
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
    
    if not isinstance(fill_method, (str, type(None))):
        raise TypeError("fill_method must be a string or None.")
    
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
                if fill_method == 'zero':
                    data_df[col] = data_df[col].fillna(0)
                elif fill_method == 'mean':
                    data_df[col] = _fill_missing_values_with_mean(data_df[col])
                elif fill_method == 'nearest':
                    data_df[col] = data_df[col].interpolate(method='nearest')
                elif fill_method == 'previous':
                    data_df[col] = data_df[col].ffill()
                elif fill_method == 'next':
                    data_df[col] = data_df[col].bfill()
                elif fill_method == 'linear':
                    data_df[col] = data_df[col].interpolate(method='linear')
                elif fill_method == 'quadratic':
                    data_df[col] = data_df[col].interpolate(method='quadratic')
                elif fill_method == 'cubic':
                    data_df[col] = data_df[col].interpolate(method='cubic')
                else:
                    raise ValueError(f"Invalid fill method: {fill_method}")
                
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

# %% === Method to load time-sensitive scattered data in csv format
def load_time_sensitive_data_from_csv(
        file_path: str,
        value_names: list[str]=None,
        fill_method: str | int=None,
        round_datetime: bool=True
    ) -> pd.DataFrame:
    """
    Load time-sensitive data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        value_names (list[str]): A list of value names corresponding to the value columns.
        fill_method (str, optional): The method to use for filling missing values (default: None).
            Possible values are
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
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    data_df = pd.read_csv(file_path)

    _, datetime_df, numeric_df = _parse_time_sensitive_dataframe(data_df, fill_method=None, round_datetime=True)

    datetime_df.columns = ['start_date', 'end_date']

    diff_end_start = datetime_df['end_date'] - datetime_df['start_date']
    not_uniform_row_ids = diff_end_start[diff_end_start != diff_end_start.iloc[1]].index.to_list()
    if not_uniform_row_ids:
        raise ValueError("Duration of the time-sensitive data (end-start) is not uniform. Non-uniform rows: " + str(not_uniform_row_ids))

    if value_names:
        if len(value_names) != len(numeric_df.columns):
            raise ValueError(f"Numeric part of the csv has {len(numeric_df.columns)} columns. Current values_names list has {len(value_names)} elements!")
        numeric_df.columns = value_names
    
    data_df = pd.concat([datetime_df, numeric_df], axis=1)
    
    return data_df

# %% === Method to load gauges table in csv format
def load_time_sensitive_gauges_from_csv(
        file_path: str
    ) -> pd.DataFrame:
    """
    Load gauges table from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded gauges table.
    """

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