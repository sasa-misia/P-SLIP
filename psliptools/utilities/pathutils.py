import pandas as pd
import os

def _resolve_path(base_dir: str, path: str) -> str:
    """
    Resolve a path to an absolute path, joining with base_dir if path is relative.

    Args:
        base_dir (str): The base directory to join with if path is relative.
        path (str): The path to resolve.

    Returns:
        str: The absolute path.
    """
    if not os.path.isabs(path):
        return os.path.abspath(os.path.join(base_dir, path))
    return path

def get_raw_path(base_inp_dir: str, file_type: str, csv_filename: str = 'input_files.csv') -> str:
    """
    Get the absolute path to a specified P-SLIP folder type as defined in the CSV file containing the list of inputs.

    Args:
        base_inp_dir (str): Directory where the CSV file is located and used as the root for relative paths.
        file_type (str): The type of file to search for (e.g., 'study_area').
        csv_filename (str): The name of the CSV file (default: 'input_files.csv').

    Returns:
        str: The absolute path to the specified folder type.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no or multiple entries for the file_type are found, or if the path is invalid.
    """
    input_files_path = os.path.join(base_inp_dir, csv_filename)
    if not os.path.exists(input_files_path):
        raise FileNotFoundError(f"{csv_filename} not found at {input_files_path}")
    input_files_df = pd.read_csv(input_files_path)
    matches = input_files_df[input_files_df['type'] == file_type]
    if matches.empty:
        raise ValueError(f"No entry with type '{file_type}' found in {csv_filename}")
    if len(matches) > 1:
        raise ValueError(f"Multiple entries with type '{file_type}' found in {csv_filename}. Please ensure only one exists.")
    folder_path = matches.iloc[0]['path']
    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError(f"The value in column 'path' is empty or invalid for folder type '{file_type}'.")
    return _resolve_path(base_inp_dir, folder_path)

def get_path_from_csv(csv_path: str, key_column: str, key_value: str, path_column: str) -> str:
    """
    Search a CSV file for the row where key_column == key_value and return the value of path_column.

    Args:
        csv_path (str): Path to the CSV file.
        key_column (str): The column name to search for the key value.
        key_value (str): The value to search for in the key_column.
        path_column (str): The column name from which to retrieve the path.

    Returns:
        str: The absolute path found in the specified path_column for the given key_value.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no row is found with the specified key or if multiple rows are found.
        ValueError: If the path in the specified path_column is empty or invalid.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    df = pd.read_csv(csv_path)
    matches = df[df[key_column] == key_value]
    if matches.empty:
        raise ValueError(f"No row found with {key_column} == '{key_value}' in {csv_path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple rows found with {key_column} == '{key_value}' in {csv_path}. Only one result expected.")
    folder_path = matches.iloc[0][path_column]
    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError(f"The value in column '{path_column}' is empty or invalid for key '{key_value}'.")
    # Use the directory containing the CSV file as base_dir
    base_dir = os.path.dirname(csv_path)
    return _resolve_path(base_dir, folder_path)