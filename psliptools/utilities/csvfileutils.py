import pandas as pd
import os

def _is_relative_path(raw_inp_path: str, csv_base_dir: str) -> bool:
    """
    Check if the given path is relative to the base directory.

    Args:
        raw_inp_path (str): Input file path.
        csv_base_dir (str): Base directory.

    Returns:
        bool: True if the path is relative to the base directory, False otherwise.
    """
    abs_pth = os.path.abspath(os.path.join(csv_base_dir, raw_inp_path)) if not os.path.isabs(raw_inp_path) else raw_inp_path
    abs_inp = os.path.abspath(csv_base_dir)
    return abs_pth.startswith(abs_inp)

def parse_csv_internal_path_field(bool_field: str | int | float, path_field: str, csv_base_dir: str) -> bool:
    """
    Parse a supposed boolean field that indicates if a path is internal, from the CSV containing the paths of the raw input files.

    Args:
        bool_field (str|int|float): Value from the column that should be boolean.
        path_field (str): Path of the raw file from the CSV.
        csv_base_dir (str): Base directory of the CSV file.

    Returns:
        bool: True if the path is internal, False otherwise (that file is stored outside the csv_base_dir).
    """
    if isinstance(bool_field, str):
        val_lower = bool_field.strip().lower()
        if val_lower in ("true", "1"):
            return True
        elif val_lower in ("false", "0"):
            return False
    elif isinstance(bool_field, (int, float)):
        if bool_field in (1, 0):
            return bool(bool_field)
    # If value is missing or not recognized, fallback to path check
    return _is_relative_path(path_field, csv_base_dir)

def update_csv_path_field(csv_path: str, path_field: str = 'path') -> bool:
    """
    Check if the raw input files CSV exists and validate paths.
    If any path is external, prompt the user to update it.

    Args:
        csv_path (str): Path to the CSV file.
        path_field (str): Name of the column containing the file paths.

    Returns:
        bool: True if any path was updated, False otherwise.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    # Read the CSV into a DataFrame
    csv_df = pd.read_csv(csv_path)

    csv_base_dir = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path)

    # Identify rows where the file is not internal (i.e., files outside the csv_base_dir)
    external_mask = ~csv_df.apply(
        lambda row: _is_relative_path(row[path_field], csv_base_dir), axis=1
    )

    if external_mask.any():
        print(f"\nSome paths are external to the 'inputs' folder:")
        for idx, row in csv_df[external_mask].iterrows():
            print(f"  - {row[path_field]}")

        for idx in csv_df[external_mask].index:
            old_path = csv_df.at[idx, path_field]
            new_path = input(f"Enter new absolute path for file '{old_path}': ").strip()
            if new_path:
                csv_df.at[idx, 'path'] = new_path
        csv_df.to_csv(csv_path, index=False)
        print(f"{csv_filename} updated with new paths.")
        return True
    else:
        print(f"All input files of {csv_filename} are internal to the 'inputs' folder.")
        return False
    
def check_raw_path(csv_path: str, file_type: str) -> bool:
    """
    Check if a specified P-SLIP folder type exists in the CSV file containing the list of inputs.

    Args:
        csv_path (str): Path to the CSV file.
        file_type (str): The type of file to check for (e.g., 'study_area').

    Returns:
        bool: True if the folder type exists, False otherwise.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    input_files_df = pd.read_csv(csv_path)
    return file_type in input_files_df['type'].values

def add_row_to_csv(csv_path: str, path_to_add: str, path_type: str, force_rewrite: bool) -> bool:
    """
    Add a new row to the CSV file containing the list of inputs.

    Args:
        csv_path (str): Path to the CSV file.
        path_to_add (str): Path to the file to add.
        path_type (str): Type of the file to add (e.g., 'study_area').
        force_rewrite (bool): If True, will overwrite the existing row if it exists.

    Returns:
        bool: True if the row was added, False if it already exists.
    """

    # Read the CSV into a DataFrame
    csv_df = pd.read_csv(csv_path)

    csv_base_dir = os.path.dirname(csv_path)

    is_internal = _is_relative_path(path_to_add, csv_base_dir)
    if is_internal:
        path_to_add = os.path.relpath(path_to_add, csv_base_dir)

    # Check if the row already exists
    row_exists = any(csv_df["type"] == path_type)

    # If the row doesn't exist, add it
    if not row_exists:
        csv_df = csv_df.append({"path": path_to_add, "type": path_type, "internal": is_internal}, ignore_index=True)
        csv_df.to_csv(csv_path, index=False)
        return True
    elif row_exists and force_rewrite:
        # If the row exists and force_rewrite is True, update the existing row
        row_index = csv_df[csv_df["type"] == path_type].index[0]
        csv_df.at[row_index, "path"] = path_to_add
        csv_df.at[row_index, "type"] = path_type
        csv_df.at[row_index, "internal"] = is_internal
        csv_df.to_csv(csv_path, index=False)
        return True
    else:
        return False