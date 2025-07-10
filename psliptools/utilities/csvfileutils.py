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
                csv_df.at[idx, path_field] = new_path
        csv_df.to_csv(csv_path, index=False)
        print(f"{csv_filename} updated with new paths.")
        return True
    else:
        print(f"All input files of {csv_filename} are internal to the 'inputs' folder.")
        return False
    
def check_raw_path(csv_path: str, type: str, subtype: str = None) -> bool:
    """
    Check if a specified P-SLIP folder/file exists in the CSV file containing the list of inputs.

    Args:
        csv_path (str): Path to the CSV file.
        type (str): The type of folder/file to check for (e.g., 'study_area').
        subtype (str, optional): The subtype of folder/file to check for.

    Returns:
        bool: True if the folder type exists, False otherwise.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    input_files_df = pd.read_csv(csv_path)

    entry_exists = type in input_files_df['type'].values
    if subtype:
        entry_exists = entry_exists and subtype in input_files_df['subtype'].values
    return entry_exists

def add_row_to_csv(
        csv_path: str, 
        path_to_add: str, 
        path_type: str, 
        path_subtype: str = None, 
        force_rewrite: bool = True
    ) -> tuple[bool, str]:
    """
    Add a new row to the CSV file containing the list of inputs.

    Args:
        csv_path (str): Path to the CSV file.
        path_to_add (str): Path to the file to add.
        path_type (str): Type of the file to add (e.g., 'study_area').
        force_rewrite (bool): If True, will overwrite the existing row if it exists.

    Returns:
        bool: True if the row was added successfully, False if it already exists and force_rewrite is False.
        str: The new ID of the added row.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """

    # Read the CSV into a DataFrame
    csv_df = pd.read_csv(csv_path, header=0)

    csv_base_dir = os.path.dirname(csv_path)

    is_internal = _is_relative_path(path_to_add, csv_base_dir)
    if is_internal:
        path_to_add = os.path.relpath(path_to_add, csv_base_dir)

    # Check if the row already exists
    row_matches = csv_df["type"] == path_type # Logical
    if path_subtype:
        row_matches = row_matches and any(csv_df["subtype"] == path_subtype)
    
    row_index = csv_df[row_matches].index

    # Generate a new ID
    prefix_len = 3
    existing_ids = csv_df[csv_df["type"] == path_type]["custom_id"]
    if not existing_ids.empty:
        new_id = f"{path_type[:prefix_len].upper()}{str(int(existing_ids.max()[prefix_len:]) + 1).zfill(prefix_len)}"
    else:
        new_id = f"{path_type[:prefix_len].upper()}001"

    # If the row doesn't exist, add it
    if not any(row_matches):
        new_row = {"custom_id": new_id, "path": path_to_add, "type": path_type, "internal": is_internal}
        if path_subtype:
            new_row["subtype"] = path_subtype
        csv_df = pd.concat([csv_df, pd.DataFrame([new_row])], ignore_index=True)
        csv_df.to_csv(csv_path, index=False)
        return (True, new_id)
    elif any(row_matches) and force_rewrite:
        # If the row exists and force_rewrite is True, update the existing row
        if len(row_index) > 1:
            raise ValueError(f"Multiple rows found for {path_type} with subtype {path_subtype}")
        csv_df.loc[row_index[0], ["custom_id", "path", "type", "internal"]] = [new_id, path_to_add, path_type, is_internal]
        if path_subtype:
            csv_df.at[row_index[0], "subtype"] = path_subtype
        csv_df.to_csv(csv_path, index=False)
        return (True, new_id)
    elif any(row_matches) and not force_rewrite:
        print(f"Row for {path_type} with subtype {path_subtype} already exists in {csv_path}. Use --force to overwrite.")
        return (False, None)
    else:
        raise ValueError(f"Row for {path_type} with subtype {path_subtype} not added in {csv_path}")