# %% === Import necessary modules
import os
import warnings
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .manage_raster import get_2d_mask_from_1d_idx
from psliptools.utilities.pandas_utils import get_list_of_values_from_dataframe

# %% === Pydantic model for parameter validation
class GridGenerationParams(BaseModel):
    indices: list[list]
    classes: list[str]
    shapes: tuple[int, int] | list[tuple[int, int]]
    csv_paths: str | list[str]
    csv_parameter_column: str
    csv_classes_column: str
    out_type: str = Field(default='float32')
    no_data: float | int | str = Field(default=0)
    
    # === Indices structure validation ===
    @field_validator('indices')
    @classmethod
    def validate_indices_structure(cls, v):
        """Validate consistent indices structure with numpy arrays."""
        if not v:
            raise ValueError("indices cannot be an empty list")
        
        if isinstance(v, pd.DataFrame):
            v = v.values.tolist()
            
        expected_length = len(v[0])
        for i, sublist in enumerate(v):
            if len(sublist) != expected_length:
                raise ValueError(
                    f"All sublists in 'indices' must have length {expected_length}. "
                    f"Sublist[{i}] has {len(sublist)} elements"
                )
            
            for j, item in enumerate(sublist):
                # Convert to numpy array if not already one
                if not isinstance(item, np.ndarray):
                    try:
                        v[i][j] = np.asarray(item)
                        item = v[i][j]  # Update reference for validation
                    except Exception as e:
                        raise ValueError(
                            f"Could not convert indices[{i}][{j}] to numpy array: {str(e)}"
                        )
                
                # Validate array dimensionality (must be 1D or scalar)
                if item.ndim > 1:
                    raise ValueError(
                        f"indices[{i}][{j}] must be 1D array or scalar, "
                        f"found {item.ndim}D array with shape {item.shape}"
                    )
        return v
    
    # === Normalization and validation of classes, shapes, and csv_paths ===
    @field_validator('classes', 'shapes', 'csv_paths')
    @classmethod
    def normalize_and_validate_fields(cls, v, info):
        """Normalize classes, shapes and csv_paths to consistent list formats."""
        field_name = info.field_name

        indices = info.data.get('indices')
        if indices is None: # Ensure indices is already validated
            return v  # Posticipate normalization and validation
        
        if field_name == 'shapes':
            expected_length = len(indices[0])
        else:
            expected_length = len(indices)

        if isinstance(v, pd.DataFrame):
            v = v.values.tolist()
        
        if isinstance(v, tuple) or isinstance(v, str):
            v = [v for _ in range(expected_length)]
        
        if len(v) != expected_length:
            raise ValueError(f"{field_name} length ({len(v)}) must match the expected length ({expected_length})")
        return v
    
    # === Out type validation ===
    @field_validator('out_type')
    @classmethod
    def validate_out_type(cls, v):
        """Validate supported numpy data types."""
        supported_types = [
            'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 
            'uint8', 'uint16', 'uint32', 'uint64', 'bool', 'U1', 'U5', 'U10', 'U20' # U stand for unicode
        ]
        if v not in supported_types:
            raise ValueError(f"out_type '{v}' not in supported types: {supported_types}")
        return v
    
    # === CSV content validation ===
    @field_validator('csv_paths')
    @classmethod
    def validate_csv_content(cls, v, info):
        """Validate CSV files contain the required columns."""
        csv_parameter_column = info.data.get('csv_parameter_column')
        csv_classes_column = info.data.get('csv_classes_column')
        
        # Posticipate validation if columns are not available yet
        if csv_parameter_column is None or csv_classes_column is None:
            return v
        
        for csv_path in set(v):  # Check each unique file only once
            try:
                columns = pd.read_csv(csv_path, nrows=0).columns
                missing_columns = []
                
                if csv_parameter_column not in columns:
                    missing_columns.append(csv_parameter_column)
                if csv_classes_column not in columns:
                    missing_columns.append(csv_classes_column)
                
                if missing_columns:
                    raise ValueError(
                        f"Columns {missing_columns} not found in CSV file: {csv_path}. "
                        f"Available columns: {list(columns)}"
                    )
                    
            except FileNotFoundError:
                raise ValueError(f"CSV file not found: {csv_path}")
            except Exception as e:
                raise ValueError(f"Error reading {csv_path}: {str(e)}")        
        return v

# %% === Method to create a raster grid of a specific parameter
def generate_grids_from_indices(
        indices: list[list[np.ndarray]],
        classes: list[str],
        shapes: tuple[int, int] | list[tuple[int, int]],  # <-- Usando pipe
        csv_paths: str | list[str],  # <-- Usando pipe
        csv_parameter_column: str,
        csv_classes_column: str,
        out_type: str='float32',
        no_data: float | int | str=0
    ) -> np.ndarray:
    """Generate a raster grid from a P-SLIP dataframe of indices and the associated AnalysisBaseGrid."""
    # Validate all parameters using Pydantic
    params = GridGenerationParams(
        indices=indices,
        classes=classes,
        shapes=shapes,
        csv_paths=csv_paths,
        csv_parameter_column=csv_parameter_column,
        csv_classes_column=csv_classes_column,
        out_type=out_type,
        no_data=no_data
    )
    
    # Create a list of empty rasters with appropriate shape and data type
    rasters = [np.full(shape, params.no_data, dtype=params.out_type) for shape in params.shapes]
    for ps in range(len(params.indices)):
        # Order must be inverted because first indices are more important as priority
        pr = len(params.indices) - ps - 1 # This is the reversed order of params.indices
        parameter_df = pd.read_csv(params.csv_paths[pr])

        value = np.asarray(
            get_list_of_values_from_dataframe(
                dataframe=parameter_df, 
                keys=params.classes[pr], 
                vals_column=params.csv_parameter_column,
                keys_column=params.csv_classes_column,
                single_match=True
            )[0], 
            dtype=params.out_type
        )
        
        for i, inds_1d in enumerate(params.indices[pr]):
            mask = get_2d_mask_from_1d_idx(inds_1d, rasters[i].shape)
            rasters[i][mask] = value
    return rasters

# %%