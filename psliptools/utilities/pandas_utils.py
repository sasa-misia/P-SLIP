#%% # Import necessary libraries
import numpy as np
import pandas as pd

#%% # Function to compare two dataframes
def compare_dataframes(
        dataframe1: pd.DataFrame, 
        dataframe2: pd.DataFrame
    ) -> np.ndarray:
    if dataframe1.shape != dataframe2.shape:
        raise ValueError(f"Dataframes must have the same shape: {dataframe1.shape} != {dataframe2.shape}")
    equality_matrix = np.zeros(dataframe1.shape, dtype=bool)
    for i, row in dataframe1.iterrows():
        temp_bb = dataframe2.iloc[i]
        for j, (_, item) in enumerate(row.items()):
            if np.isscalar(item):
                equality_matrix[i, j] = item == temp_bb[j]
            else:
                equality_matrix[i, j] = (item == temp_bb[j]).all()
    return equality_matrix

#%%