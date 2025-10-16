# %% === Import necessary libraries
import numpy as np
import pandas as pd
import warnings

# %% === Method to compare two dataframes
def compare_dataframes(
        dataframe1: pd.DataFrame, 
        dataframe2: pd.DataFrame,
        row_order: bool = True
    ) -> np.ndarray:
    """
    Compare the elements of two dataframes, with or without considering the order of the rows.

    Args:
        dataframe1 (pd.DataFrame): The first dataframe to compare.
        dataframe2 (pd.DataFrame): The second dataframe to compare.
        row_order (bool, optional): If True, the order of the rows is considered. Defaults to True.
        
    Returns:
        np.ndarray: A boolean matrix with the same shape as the dataframes, where True means the elements are equal.
    """
    if dataframe1.shape != dataframe2.shape:
        raise ValueError(f"Dataframes must have the same shape: {dataframe1.shape} != {dataframe2.shape}")
    
    equality_matrix = np.zeros(dataframe1.shape, dtype=bool)
    for r1, row1 in dataframe1.iterrows():
        if row_order:
            row2 = dataframe2.iloc[r1]
        else:
            r2 = np.where(dataframe2.iloc[:, 0] == row1.iloc[0])[0]
            if len(r2) == 0:
                continue
            elif len(r2) > 1:
                raise ValueError(f"Multiple rows with the same first column value: {row1.iloc[0]} (not allowed with row_order=False)")
            row2 = dataframe2.iloc[r2]
            
        for c1, (_, item1) in enumerate(row1.items()):
            item2 = row2[c1]
            if pd.isna(item1) and pd.isna(item2):
                equality_matrix[r1, c1] = True
            elif pd.isna(item1) or pd.isna(item2):
                equality_matrix[r1, c1] = False
            else:
                equality_matrix[r1, c1] = np.array_equal(item1, item2)

    return equality_matrix

# %% === Method to compare elements of two dataframes
def compare_dataframes_columns(
        dataframe1: pd.DataFrame, 
        dataframe2: pd.DataFrame,
        columns_df1: list[str],
        columns_df2: list[str],
        row_order: bool = True
    ) -> np.ndarray:
    """
    Compare the elements of two dataframes based on the columns provided. 
    No matter the order of the rows between the two dataframes.

    Args:
        dataframe1 (pd.DataFrame): The first dataframe to compare.
        dataframe2 (pd.DataFrame): The second dataframe to compare.
        columns_df1 (list[str]): The columns of the first dataframe to compare.
        columns_df2 (list[str]): The columns of the second dataframe to compare (it must be in the same order as columns_df1).
        row_order (bool, optional): If True, the order of the rows is considered. Defaults to True.
    
    Returns:
        np.ndarray: A boolean matrix where each row corresponds to a row in dataframe1 and each column corresponds to the number of columns provided.
    """
    if len(columns_df1) != len(columns_df2):
        raise ValueError(f"Dataframes must have the same number of columns to compare: {len(columns_df1)} != {len(columns_df2)}")
    
    is_in_dataframe2 = np.zeros((dataframe1.shape[0], len(columns_df1)), dtype=bool)
    for r1, row1 in dataframe1.iterrows():
        if row_order:
            row2 = dataframe2.iloc[r1].iloc[0] # With .iloc[0] it becomes a Series. Please use iloc and not loc, otherwise rows after the first will not be found...
        else:
            r2 = np.where(dataframe2.loc[:, columns_df2[0]] == row1.loc[columns_df1[0]])[0]
            if len(r2) == 0:
                continue
            elif len(r2) > 1:
                raise ValueError(f"Multiple rows with the same first column value: {row1.iloc[0]} (not allowed with row_order=False)")
            row2 = dataframe2.iloc[r2].iloc[0] # With .iloc[0] it becomes a Series. Please use iloc and not loc, otherwise rows after the first will not be found...

        for c1, (col1, col2) in enumerate(zip(columns_df1, columns_df2)):
            item1 = row1[col1]
            item2 = row2[col2]
            if pd.isna(item1) and pd.isna(item2):
                is_in_dataframe2[r1, c1] = True
            elif pd.isna(item1) or pd.isna(item2):
                is_in_dataframe2[r1, c1] = False
            else:
                is_in_dataframe2[r1, c1] = np.array_equal(item1, item2)

    return is_in_dataframe2

# %% === Method to obtain values from a dataframe
def get_list_of_values_from_dataframe(
        dataframe: pd.DataFrame, 
        keys: str,
        vals_column: str,
        keys_column: str=None,
        single_match: bool = False
    ) -> list[any]:
    """
    Get values from a dataframe based on a keys and a values column.

    Args:
        dataframe (pd.DataFrame): The dataframe to search in.
        keys (str): The keys to search for.
        vals_column (str): The column name of the values to pick in the dataframe.
        keys_column (str, optional): The column name of the keys in the dataframe.
        single_match (bool, optional): If True, raise an error if multiple matches are found (default is False).

    Returns:
        list[any]: A list of objects corresponding to the keys, picked from the vals_column.
    """
    out_list = []

    if isinstance(keys, str):
        keys = [keys]

    if single_match and len(keys) > 1:
        raise ValueError("Multiple keys not allowed when single_match is True")

    if keys_column is None:
        # Trying to find the key column
        keys_column = []
        for col in dataframe.columns:
            if keys in dataframe[col].unique():
                keys_column.append(col)
        if len(keys_column) == 0:
            raise ValueError(f"Keys {keys} not found in the dataframe")
        elif len(keys_column) > 1:
            raise ValueError(f"Multiple keys found in the dataframe: {keys_column}")
        keys_column = keys_column[0]
    
    # Check if keys are present in the dataframe
    for key in keys:
        if not key in dataframe[keys_column].unique():
            raise ValueError(f"Key: {key} not found in the dataframe at column: {keys_column}")

    csv_df = dataframe[dataframe[keys_column].isin(keys)]
    for _, row in csv_df.iterrows():
        out_list.append(row[vals_column])
    
    if single_match and len(out_list) > 1:
        raise ValueError(f"Multiple matches found for {keys} in the dataframe, but single_match is set to True")

    return out_list

# %% === Method to get the boolean indices of the series within the given range
def get_mask_in_range(
        series: pd.Series, 
        min_value: float=None, 
        max_value: float=None,
        include_min_max: bool=True
    ) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if not isinstance(min_value, (int, float)) and min_value is not None:
        raise TypeError("min_value must be a number.")
    if not isinstance(max_value, (int, float)) and max_value is not None:
        raise TypeError("max_value must be a number.")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value.")
    if not isinstance(include_min_max, bool):
        raise TypeError("include_min_max must be a boolean.")
    
    greater_range_filter = (series > min_value) if min_value is not None else pd.Series([True] * series.shape[0], index=series.index)
    lower_range_filter = (series < max_value) if max_value is not None else pd.Series([True] * series.shape[0], index=series.index)
    if include_min_max:
        greater_range_filter = greater_range_filter | (series == min_value) if min_value is not None else greater_range_filter
        lower_range_filter = lower_range_filter | (series == max_value) if max_value is not None else lower_range_filter

    mask_inside_range = greater_range_filter & lower_range_filter

    return mask_inside_range

# %% === Method to filter numeric data by numeric range
def filter_numeric_series(
    series: pd.Series,
    min_value: float=None,
    max_value: float=None,
    include_extremes: bool=True,
    filler_value: float=np.nan
    ) -> pd.Series:
    """
    Filter a numeric pandas Series by a numeric range.

    Args:
        series (pd.Series): The numeric pandas Series to filter.
        min_value (float, optional): The minimum value to include in the range (default: None).
        max_value (float, optional): The maximum value to include in the range (default: None).
        include_extremes (bool, optional): Whether to include the minimum and maximum values in the range (default: False).
        filler_value (float, optional): The value to fill the filtered values with (default: np.nan).

    Returns:
        pd.Series: The filtered numeric pandas Series, where values outside the range are filled with the specified value.
    """
    series = series.copy()
    rows_with_valid_data = get_mask_in_range(
        series=series, 
        min_value=min_value, 
        max_value=max_value, 
        include_min_max=include_extremes
    )
    series.loc[~rows_with_valid_data] = filler_value

    if (~rows_with_valid_data).any():
        warnings.warn(f"Some data values were filtered out using range [{min_value} - {max_value}].", stacklevel=2)

    return series

# %% === Helper method to fill gaps in a series with the mean between the first and last non-empty values
def _fill_missing_values_of_numeric_series_with_mean(
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
def fill_missing_values_of_numeric_series(
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
        column_data = _fill_missing_values_of_numeric_series_with_mean(column_data)
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

# %%