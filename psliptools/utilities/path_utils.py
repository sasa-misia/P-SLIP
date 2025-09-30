# %% === Import necessary libraries
import pandas as pd
import os

# %% === Helper method to resolve relative paths
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

# %% === Method to get the absolute path to a specified P-SLIP folder
def get_raw_fold(csv_path: str, fold_type: str, fold_subtype: str = None) -> str:
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
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    input_files_df = pd.read_csv(csv_path)
    if fold_subtype:
        matches = input_files_df[(input_files_df['type'] == fold_type) & (input_files_df['subtype'] == fold_subtype)]
    else:
        matches = input_files_df[input_files_df['type'] == fold_type]

    if matches.empty:
        raise ValueError(f"No entry with type '{fold_type}' found in {csv_path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple entries with type '{fold_type}' found in {csv_path}. Please ensure only one folder exists.")
    folder_path = matches.iloc[0]['path']
    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError(f"The value in column 'path' is empty or invalid for folder type '{fold_type}'.")
    
    resolved_path = _resolve_path(os.path.dirname(csv_path), folder_path)

    if os.path.isdir(resolved_path):
        return resolved_path
    else:
        raise ValueError(f"The value in column 'path' is not a valid directory for folder type '{fold_type}'.")

# %% === Method to get the absolute paths to specified P-SLIP file types
def get_raw_files(csv_path: str, file_type: str, file_subtype: str = None) -> list[str]:
    """
    Get the absolute paths to specified P-SLIP folder types as defined in the CSV file containing the list of inputs.

    Args:
        csv_path (str): Path to the CSV file.
        file_type (str): The type of file to search for (e.g., 'study_area').
        file_subtype (str, optional): The subtype of file to search for.

    Returns:
        list[str]: A list of absolute paths to the specified folder types.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no entries for the file_type are found, or if any path is invalid.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    input_files_df = pd.read_csv(csv_path)
    if file_subtype:
        matches = input_files_df[(input_files_df['type'] == file_type) & (input_files_df['subtype'] == file_subtype)]
    else:
        matches = input_files_df[input_files_df['type'] == file_type]
    
    if matches.empty:
        raise ValueError(f"No entry with type '{file_type}' found in {csv_path}")

    base_dir = os.path.dirname(csv_path)
    folder_paths = matches['path'].tolist()
    
    resolved_paths = [_resolve_path(base_dir, p) for p in folder_paths if isinstance(p, str) and p.strip()]
    
    if not resolved_paths:
        raise ValueError(f"All paths are empty or invalid for folder type '{file_type}'.")
    
    return resolved_paths

# %% === Method to get the absolute path to a specified P-SLIP folder
def get_fold_from_csv(csv_path: str, key_column: str, key_value: str, path_column: str) -> str:
    """
    Search a CSV file for the row where key_column == key_value (only unique match 
    is accepted) and return the value of path_column.`

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