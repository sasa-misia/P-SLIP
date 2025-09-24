# %% === Import necessary modules
import warnings
import numpy as np
import pandas as pd

# %% === Helper method to parse and validate a time-sensitive dataframe
def _parse_time_sensitive_dataframe(
        data_df: pd.DataFrame,
        fill_value: float=None,
        round_datetime: bool=True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse and validate a time-sensitive dataframe.

    Args:
        data_df (pd.DataFrame): The dataframe to parse and validate.
        fill_value (float, optional): The value to use for filling missing values (default: None).
        round_datetime (bool, optional): If True, round datetime columns to the nearest minute (default: True).

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame): A tuple containing the parsed dataframe, the datetime columns, and the numeric columns.
    """
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
            if len(missing_row_ids) > 0 and fill_value is not None:
                data_df[col] = data_df[col].fillna(fill_value)
                warnings.warn(
                    f"Missing values in column [{col}] at rows {missing_row_ids} have been filled with {fill_value}",
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
                    f"Missing values in column [{col}] at rows {missing_row_ids} have been filled with linear interpolation",
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
        start_date_column: str | int=None,
        end_date_column: str | int=None,
        value_columns: list[str | int]=None,
        value_names: list[str]=None,
        date_format: str=None,
    ) -> pd.DataFrame:
    """
    Load time-sensitive data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        start_date_column (str | int): The name or index of the start date column.
        end_date_column (str | int): The name or index of the end date column.
        value_columns (list[str | int]): A list of column names or indices for the values.
        value_names (list[str]): A list of value names corresponding to the value columns.
        date_format (str, optional): The format of the date columns (default: None).

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    data_df = pd.read_csv(file_path)
    
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