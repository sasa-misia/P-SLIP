# %% === Import necessary modules
import os
import warnings
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .coordinates import convert_grids_and_profile_to_prj, are_coords_geographic, get_xy_grids_from_profile, are_grids_ordered
from .manage_raster import get_2d_mask_from_1d_idx
from ..utilities.pandas_utils import get_list_of_values_from_dataframe

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
        shapes: tuple[int, int] | list[tuple[int, int]],
        csv_paths: str | list[str],
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

# %% === Function to gerenerate gradient rasters
def generate_gradient_rasters( # TODO: Check this function
        dtm: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        profile: dict=None,
        out_type: str='float32',
        no_data: float | int | str=0
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradients in x and y directions from DTM, longitude, and latitude grids.
    
    Args:
        dtm (np.ndarray): Array of DTM values (it must be in meters and 2D).
        lon (np.ndarray): Array of longitude values (it must be in degrees and 2D).
        lat (np.ndarray): Array of latitude values (it must be in degrees and 2D).
        profile (dict, optional): Raster profile for coordinate conversion (default: None).
        out_type (str, optional): The data type of the output rasters (default: 'float32').
        no_data (float | int | str, optional): The no_data value of the input and output rasters (default: 0).
    
    Returns:
        tuple(np.ndarray, np.ndarray): Tuple containing dz_dx and dz_dy gradient rasters.
    """
    if not isinstance(dtm, np.ndarray) or not isinstance(lon, np.ndarray) or not isinstance(lat, np.ndarray):
        raise ValueError("DTM, longitude, and latitude must be numpy arrays.")
    
    # Validate input arrays
    if dtm.ndim != 2:
        raise ValueError("DTM must be a 2D array")
    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError("Longitude and latitude must be 2D arrays")
    if dtm.shape != lon.shape or dtm.shape != lat.shape:
        raise ValueError("DTM, longitude, and latitude arrays must have the same shape")
    
    if profile:
        if lon is not None or lat is not None:
            raise ValueError('If profile is provided, lon and lat must not be provided')
        lon, lat = get_xy_grids_from_profile(profile)
    
    # Check if coordinates are geographic and convert to projected CRS if needed
    if are_coords_geographic(lon, lat):
        # Convert to projected coordinates (meters)
        x_proj, y_proj, _ = convert_grids_and_profile_to_prj(lon, lat)
    else:
        # Already in projected coordinates
        x_proj, y_proj = lon, lat
    
    # Check if grids are ordered
    if not are_grids_ordered(x_proj, y_proj):
        raise ValueError("x_proj and y_proj grids must be monotonically ordered along their axes.")
    
    # Calculate pixel sizes in x and y directions
    dx = np.abs(np.mean(x_proj[:, 1:] - x_proj[:, :-1]))
    dy = np.abs(np.mean(y_proj[1:, :] - y_proj[:-1, :]))
    
    # Calculate gradients using numpy gradient
    dz_dy, dz_dx = np.gradient(dtm, dy, dx)
    
    # Handle no_data values
    if no_data is not None:
        dtm_mask = (dtm == no_data)
        dz_dx[dtm_mask] = no_data
        dz_dy[dtm_mask] = no_data
    
    # Convert to specified output type
    dz_dx = dz_dx.astype(out_type)
    dz_dy = dz_dy.astype(out_type)
    
    return dz_dx, dz_dy

# %% Method to create slope and aspect rasters from dtm, longitude, and latitude grids
def generate_slope_and_aspect_rasters( # TODO: Check this function
        dtm: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        profile: dict=None,
        out_type: str='float32',
        no_data: float | int | str=0
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate slope and aspect rasters from DTM, longitude, and latitude grids.
    
    Args:
        dtm (np.ndarray): Array of DTM values (it must be in meters and 2D).
        lon (np.ndarray): Array of longitude values (it must be in degrees and 2D).
        lat (np.ndarray): Array of latitude values (it must be in degrees and 2D).
        profile (dict, optional): Raster profile for coordinate conversion (default: None).
        out_type (str, optional): The data type of the output rasters (default: 'float32').
        no_data (float | int | str, optional): The no_data value of the input and output rasters (default: 0).
    
    Returns:
        tuple(np.ndarray, np.ndarray): Tuple containing the slope and aspect rasters.
    """
    # TODO: Check and validate the entire function (optional: find a faster alternative)

    if not isinstance(dtm, np.ndarray) or not isinstance(lon, np.ndarray) or not isinstance(lat, np.ndarray):
        raise ValueError("DTM, longitude, and latitude must be numpy arrays.")
    
    # Validate input arrays
    if dtm.ndim != 2:
        raise ValueError("DTM must be a 2D array")
    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError("Longitude and latitude must be 2D arrays")
    if dtm.shape != lon.shape or dtm.shape != lat.shape:
        raise ValueError("DTM, longitude, and latitude arrays must have the same shape")
    
    if profile:
        if lon or lat:
            raise ValueError('If profile is provided, lon and lat must not be provided')
        lon, lat = get_xy_grids_from_profile(profile)
    
    # Check if coordinates are geographic and convert to projected CRS if needed
    if are_coords_geographic(lon, lat):
        # Convert to projected coordinates (meters)
        x_proj, y_proj, _ = convert_grids_and_profile_to_prj(lon, lat)
    else:
        # Already in projected coordinates
        x_proj, y_proj = lon, lat
    
    # Check if grids are ordered
    if not are_grids_ordered(x_proj, y_proj):
        raise ValueError("x_proj and y_proj grids must be monotonically ordered along their axes.")
    
    # Calculate pixel sizes in x and y directions
    dx = np.abs(np.mean(x_proj[:, 1:] - x_proj[:, :-1]))
    dy = np.abs(np.mean(y_proj[1:, :] - y_proj[:-1, :]))
    
    # Calculate gradients using numpy gradient
    dz_dy, dz_dx = np.gradient(dtm, dy, dx)
    
    # Calculate slope (in degrees)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Calculate aspect (in degrees from north, clockwise)
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    
    # Adjust aspect to be between 0-360 degrees
    aspect_deg = (aspect_deg + 360) % 360
    
    # Handle no_data values
    if no_data is not None:
        dtm_mask = (dtm == no_data)
        slope_deg[dtm_mask] = no_data
        aspect_deg[dtm_mask] = no_data
    
    # Convert to specified output type
    slope_deg = slope_deg.astype(out_type)
    aspect_deg = aspect_deg.astype(out_type)
    
    return slope_deg, aspect_deg

# %% Method to create curvature rasters from dtm
def generate_curvature_rasters( # TODO: Check this function
        dtm: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        profile: dict=None,
        out_type: str='float32',
        no_data: float | int | str=0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate profile, planform, and twisting curvature rasters from DTM, longitude, and latitude grids.
    
    Args:
        dtm (np.ndarray): Array of DTM values (it must be in meters and 2D).
        lon (np.ndarray): Array of longitude values (it must be in degrees and 2D).
        lat (np.ndarray): Array of latitude values (it must be in degrees and 2D).
        profile (dict, optional): Raster profile for coordinate conversion (default: None).
        out_type (str, optional): The data type of the output rasters (default: 'float32').
        no_data (float | int | str, optional): The no_data value of the input and output rasters (default: 0).
    
    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): Tuple containing profile, planform, and twisting curvature rasters.
    """
    # TODO: Check and validate the entire function (optional: find a faster alternative)

    if not isinstance(dtm, np.ndarray) or not isinstance(lon, np.ndarray) or not isinstance(lat, np.ndarray):
        raise ValueError("DTM, longitude, and latitude must be numpy arrays.")
    
    # Validate input arrays
    if dtm.ndim != 2:
        raise ValueError("DTM must be a 2D array")
    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError("Longitude and latitude must be 2D arrays")
    if dtm.shape != lon.shape or dtm.shape != lat.shape:
        raise ValueError("DTM, longitude, and latitude arrays must have the same shape")
    
    if profile:
        if lon or lat:
            raise ValueError('If profile is provided, lon and lat must not be provided')
        lon, lat = get_xy_grids_from_profile(profile)
    
    # Check if coordinates are geographic and convert to projected CRS if needed
    if are_coords_geographic(lon, lat):
        # Convert to projected coordinates (meters)
        x_proj, y_proj, _ = convert_grids_and_profile_to_prj(lon, lat)
    else:
        # Already in projected coordinates
        x_proj, y_proj = lon, lat
    
    # Check if grids are ordered
    if not are_grids_ordered(x_proj, y_proj):
        raise ValueError("x_proj and y_proj grids must be monotonically ordered along their axes.")
    
    # Calculate pixel sizes in x and y directions
    dx = np.abs(np.mean(x_proj[:, 1:] - x_proj[:, :-1]))
    dy = np.abs(np.mean(y_proj[1:, :] - y_proj[:-1, :]))
    
    # Calculate first derivatives (slope components)
    dz_dy, dz_dx = np.gradient(dtm, dy, dx)
    
    # Calculate second derivatives
    dz_dx2 = np.gradient(dz_dx, dx, axis=1)
    dz_dy2 = np.gradient(dz_dy, dy, axis=0)
    dz_dxdy = np.gradient(dz_dx, dy, axis=0)
    
    # Calculate slope magnitude
    slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
    
    # Calculate curvature components
    # Profile curvature (curvature in the direction of maximum slope)
    profile_curvature = (dz_dx2 * dz_dx**2 + 2 * dz_dxdy * dz_dx * dz_dy + dz_dy2 * dz_dy**2) / \
                        (slope_magnitude**2 + 1e-10)  # Add small value to avoid division by zero
    
    # Planform curvature (curvature perpendicular to the direction of maximum slope)
    planform_curvature = (dz_dx2 * dz_dy**2 - 2 * dz_dxdy * dz_dx * dz_dy + dz_dy2 * dz_dx**2) / \
                         (slope_magnitude**2 + 1e-10)  # Add small value to avoid division by zero
    
    # Twisting curvature (torsional curvature)
    twisting_curvature = (dz_dx2 * dz_dx * dz_dy + dz_dxdy * (dz_dx**2 - dz_dy**2) - dz_dy2 * dz_dx * dz_dy) / \
                        (slope_magnitude**2 + 1e-10)  # Add small value to avoid division by zero
    
    # Handle no_data values
    if no_data is not None:
        dtm_mask = (dtm == no_data)
        profile_curvature[dtm_mask] = no_data
        planform_curvature[dtm_mask] = no_data
        twisting_curvature[dtm_mask] = no_data
    
    # Convert to specified output type
    profile_curvature = profile_curvature.astype(out_type)
    planform_curvature = planform_curvature.astype(out_type)
    twisting_curvature = twisting_curvature.astype(out_type)
    
    return profile_curvature, planform_curvature, twisting_curvature

# %%
