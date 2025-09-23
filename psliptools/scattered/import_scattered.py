# %% === Import necessary modules
import warnings
import numpy as np
import pandas as pd

# %% === Method to load time-sensitive scattered data in csv format
def load_time_sensitive_data_from_csv(
        file_path: str,
        start_date_column: str | int,
        end_date_column: str | int,
        value_columns: list[str | int],
        value_names: list[str],
        date_format: str='%Y-%m-%d',
    ) -> pd.DataFrame:
    """
    Load time-sensitive data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        start_date_column (str | int): The name or index of the start date column.
        end_date_column (str | int): The name or index of the end date column.
        value_columns (list[str | int]): A list of column names or indices for the values.
        value_names (list[str]): A list of value names corresponding to the value columns.
        date_format (str, optional): The format of the date columns. Defaults to '%Y-%m-%d'.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """

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