# %% ===  Import necessary modules
import os
import chardet
import csv
import warnings
import pandas as pd

# %% === Function to check if a path is relative to a base directory
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

# %% === Function to parse a supposed boolean field that indicates if a path is internal
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

# %% === Function to check if any path is external and prompt the user to update it
def update_external_paths_in_csv(csv_path: str, path_field: str = 'path') -> bool:
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

# %% === Function to check if a specified P-SLIP folder/file exists in the CSV file containing the list of inputs
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

# %% === Function to add a new row to the CSV file containing the list of inputs
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
    row_matches = csv_df["type"] == path_type # Logical array
    if path_subtype:
        row_matches = row_matches & (csv_df["subtype"] == path_subtype) # Not and but & because it is a logical array and not a scalar!
    
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

# %% === Function to merge manually detect datetime columns, that might be read as strings, with auto-detection
def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Manually detect columns that contain datetime strings but might be read as object/string type.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        
    Returns:
        list[str]: List of column names that likely contain datetime data.
    """
    manually_detected_datetime_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            # Try to convert a sample to datetime
            sample_size = min(20, len(df))
            sample_data = df[col].head(sample_size)
            
            # Check if values look like datetime strings
            datetime_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
            ]
            
            pattern_matches = any(
                sample_data.str.contains(pattern, na=False).any() 
                for pattern in datetime_patterns
            )
            
            if pattern_matches:
                manually_detected_datetime_cols.append(col)
    
    auto_detected_datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.to_list()

    detected_datetime_cols = list(dict.fromkeys(manually_detected_datetime_cols + auto_detected_datetime_cols)) # Remove duplicated preserving the order
    
    return detected_datetime_cols

# %% === Function to detect actual string columns (not just object dtype)
def _detect_string_columns(df: pd.DataFrame, exclude_datetimes: bool=True) -> list[str]:
    """
    Detect columns that contain actual string data (not just object dtype).
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        exclude_datetime_cols (list[str]): Columns to exclude (datetime columns).
        
    Returns:
        list[str]: List of column names that contain actual string data.
    """
    string_cols = []
    
    if exclude_datetimes:
        exclude_datetime_cols = _detect_datetime_columns(df)
    else:
        exclude_datetime_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object' and col not in exclude_datetime_cols:
            # Check if the column contains actual string data
            sample_size = min(20, len(df))
            sample_data = df[col].head(sample_size)
            
            # Check if values are strings (not mixed types)
            if sample_data.apply(lambda x: isinstance(x, str)).all():
                string_cols.append(col)
    
    return string_cols

# %% === Function to obtain the name of the columns with a specific data type in a CSV file
def get_csv_column_names(
        csv_path: str, 
        data_type: str | list[str]=None
    ) -> list[str]:
    """
    Obtain the name of the columns with a specific data type in a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        data_type (str): Data type to search for (e.g., 'number', 'string', 'datetime', 'timedelta', 'bool', 'category', 'object', etc.).

    Returns:
        list[str]: List of column names with the specified data type.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    csv_first_rows = pd.read_csv(csv_path, nrows=5)
    
    if data_type is None:
        dtype_cols = csv_first_rows.columns.to_list()
    else:
        if not isinstance(data_type, list):
            data_type = [data_type]
        
        if [isinstance(x, str) for x in data_type].count(False) > 0:
            raise ValueError("data_type must be a string or a list of strings")
        
        # Manual datetime detection for string columns that might be datetime
        datetime_cols = _detect_datetime_columns(csv_first_rows)
        
        # Detect actual string columns (not just object dtype)
        string_cols = _detect_string_columns(csv_first_rows, datetime_cols)
        
        # Get generic object columns (excluding datetime and string columns)
        pandas_object_cols = csv_first_rows.select_dtypes(include=['object']).columns.to_list()
        object_cols = [col for col in pandas_object_cols if col not in datetime_cols and col not in string_cols]
        
        # Map data_type requests to actual column lists
        type_mapping = {
            'datetime': datetime_cols,
            'string': string_cols,
            'object': object_cols  # Generic object columns (not strings or datetime)
        }
        
        # For standard types, use pandas select_dtypes, otherwise, for 'datetime', 'string', and 'object' use the internal method, which is different from pandas
        dtype_cols = []
        for dtype in data_type:
            if dtype in type_mapping:
                dtype_cols.extend(type_mapping[dtype])
            else:
                # Use pandas for other types
                dtype_cols.extend(csv_first_rows.select_dtypes(include=[dtype]).columns.to_list())
        
        # Remove duplicates while preserving order
        dtype_cols = list(dict.fromkeys(dtype_cols))
    
    return dtype_cols

# %% === Function to rename the header row of a CSV file
def rename_csv_header(
        csv_path: str, 
        new_header: list[str],
        data_type: str | list[str]=None
    ) -> None:
    """
    Rename the header row of a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        data_type (str): Data type to search for (e.g., 'number', 'string', etc.).
        new_header (str): New header name.
    """
    csv_df = pd.read_csv(csv_path, header=0)
    old_sel_header = get_csv_column_names(csv_path, data_type)
    if len(old_sel_header) != len(new_header):
        raise ValueError("The number of new headers must match the number of selected columns.")
    
    # Create a mapping dictionary for column renaming
    rename_mapping = dict(zip(old_sel_header, new_header))
    
    # Rename only the selected columns
    csv_df.rename(columns=rename_mapping, inplace=True)
    csv_df.to_csv(csv_path, index=False)

# %% === Helper function to get the csv delimiter
def _get_csv_delimiter(
        csv_path: str,
        encoding: str=None
    ) -> str:
    """
    Get the delimiter used in a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        encoding (str, optional): Encoding of the CSV file (default is None, which means it will be detected).

    Returns:
        str: The delimiter used in the CSV file.
    """
    if encoding is None:
        # Detect encoding
        with open(csv_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    
    # Detect separator
    with open(csv_path, 'r', encoding=encoding) as f:
        file_size = os.path.getsize(csv_path)
        sample_size = min(4096, file_size) if file_size > 0 else 1024
        sample = f.read(sample_size)
        
        if not sample.strip():
            warnings.warn(f"Empty or whitespace-only file: {csv_path} -> using default separator [,]", stacklevel=2)
            sep = ','  # Default for empty or whitespace-only files
        else:
            sniffer = csv.Sniffer()
            try:
                sep = sniffer.sniff(sample, delimiters=',;\t|').delimiter
            except csv.Error:
                # Manual fallback: count delimiters in sample and pick the most common
                delimiter_counts = {}
                for delim in ',;\t|':
                    count = sample.count(delim)
                    if count > 0:
                        delimiter_counts[delim] = count
                if delimiter_counts:
                    sep = max(delimiter_counts, key=delimiter_counts.get)
                else:
                    warnings.warn(f"Unable to detect separator in file: {csv_path} -> using default separator [,]", stacklevel=2)
                    sep = ','  # Ultimate default if no delimiters found
    
    return sep

# %% === Function to read a CSV file with the more comprehensive pandas read_csv function
def read_generic_csv(
        csv_path: str,
        sep: str=None,
        regular: bool=True
    ) -> pd.DataFrame:
    """
    Read a CSV file with the more comprehensive pandas read_csv function.
    
    Args:
        csv_path (str): Path to the CSV file.
        sep (str): The separator character to use. If not provided, it will be detected.
        regular (bool): Regular means that the csv is consistent in rows and columns (no invalid rows or rows with less columns). 
            If True, use the regular pandas read_csv function. If False, use the more 
            comprehensive read_csv function, but header will not be detected.
            In case of errors, switch to False this option.
    
    Returns:
        pd.DataFrame: The DataFrame read from the CSV file.
    """
    # Detect encoding
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read())
        charenc = result['encoding']
    
    # Detect separator if not provided
    if sep is None:
        sep = _get_csv_delimiter(csv_path, charenc)
    
    if regular:
        read_df = pd.read_csv(csv_path, encoding=charenc, sep=sep)
    else:
        # Detect max_cols and potential header in one pass
        max_cols = 0
        header_row = None
        col_names = None
        with open(csv_path, 'r', encoding=charenc) as f:
            for i, line in enumerate(f):
                if line.strip():
                    cols = [c.strip().strip('"\'\'') for c in line.split(sep)]
                    num_cols = len(cols)
                    if num_cols > max_cols:
                        max_cols = num_cols
                        # Reset header if max_cols increased
                        header_row = None
                        col_names = None
                    if header_row is None and num_cols == max_cols and all(c for c in cols):
                        header_row = i
                        col_names = cols
        
        if header_row is None:
            col_names = [f'col_{i}' for i in range(max_cols)]

        read_df = pd.read_csv(csv_path, encoding=charenc, sep=sep, engine='python', on_bad_lines='warn', names=col_names, header=None)
    
    return read_df

# %%
