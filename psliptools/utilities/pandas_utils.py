# %% === Import necessary libraries
import numpy as np
import pandas as pd

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

# %% === Function to obtain values from a dataframe
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

# %%
