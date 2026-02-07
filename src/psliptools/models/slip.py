# %% === Import necessary modules
import numpy as np
import pandas as pd
import warnings

# %% === Function to evaluate the factor of safety of a slope thorugh the SLIP model
def run_slip_model(
        slope : np.ndarray,
        soil_specific_gravity : np.ndarray,
        soil_cohesion : np.ndarray,
        root_cohesion : np.ndarray,
        soil_friction : np.ndarray,
        soil_drainage : np.ndarray,
        infiltration_coefficient : np.ndarray,
        A_slip : np.ndarray,
        soil_porosity : np.ndarray,
        rain_history : pd.DataFrame, # rain history dataframe, it must contain the columns 'start_date', 'end_date', and 'rain', where rain is in mm!
        instability_depth : float = 1.2, # depth of analysis, in meters!
        soil_saturation : float = 0.8, # the base / starting soil saturation (0 to 1)
        lambda_slip : float = 0.4, # default SLIP value
        alpha_slip : float = 3.4, # default SLIP value
        predisposing_time_window : pd.Timedelta = pd.Timedelta(days=30), # default SLIP value
        datetime_match_tolerance : pd.Timedelta = pd.Timedelta(seconds=1) # Tolerance for date matching
    ) -> pd.DataFrame :
    """
    Evaluate the factor of safety of a slope thorugh the SLIP model.

    Args:
        slope (np.ndarray): The slope of the slope.
        soil_specific_gravity (np.ndarray): The soil specific gravity of the slope.
        soil_cohesion (np.ndarray): The soil cohesion of the slope.
        root_cohesion (np.ndarray): The root cohesion of the slope.
        soil_friction (np.ndarray): The soil friction of the slope.
        soil_drainage (np.ndarray): The soil drainage of the slope.
        infiltration_coefficient (np.ndarray): The infiltration coefficient of the slope.
        A_slip (np.ndarray): The A slip of the slope.
        soil_porosity (np.ndarray): The soil porosity of the slope.
        rain_history (pd.DataFrame): The rain history dataframe, it must contain the columns 'start_date', 'end_date', and 'rain', where rain is in mm!
        instability_depth (float, optional): The depth of analysis, in meters!. Defaults to 1.2.
        soil_saturation (float, optional): The base / starting soil saturation (0 to 1). Defaults to 0.8.
        lambda_slip (float, optional): The lambda slip of the model. Defaults to 0.4.
        alpha_slip (float, optional): The alpha slip of the model. Defaults to 3.4.
        predisposing_time_window (pd.Timedelta, optional): The predisposing time window of the model. Defaults to pd.Timedelta(days=30).
        datetime_match_tolerance (pd.Timedelta, optional): Tolerance for date matching. Defaults to pd.Timedelta(seconds=1).

    Returns:
        pd.DataFrame: The history of the factor of safety of the slope.
    """
    # Input validation
    slope = np.atleast_1d(slope)
    soil_specific_gravity = np.atleast_1d(soil_specific_gravity)
    soil_cohesion = np.atleast_1d(soil_cohesion)
    root_cohesion = np.atleast_1d(root_cohesion)
    soil_friction = np.atleast_1d(soil_friction)
    soil_drainage = np.atleast_1d(soil_drainage)
    infiltration_coefficient = np.atleast_1d(infiltration_coefficient)
    A_slip = np.atleast_1d(A_slip)
    soil_porosity = np.atleast_1d(soil_porosity)
    if not isinstance(lambda_slip, float):
        raise TypeError('lambda_slip must be a float')
    if not isinstance(alpha_slip, float):
        raise TypeError('alpha_slip must be a float')
    if not isinstance(predisposing_time_window, pd.Timedelta):
        raise TypeError('predisposing_time_window must be a pandas Timedelta')
    if not isinstance(rain_history, pd.DataFrame):
        raise TypeError('rain_history must be a pandas DataFrame')
    
    # Validate required columns
    required_columns = ['start_date', 'end_date', 'rain']
    missing_columns = [col for col in required_columns if col not in rain_history.columns]
    if missing_columns:
        raise ValueError(f"rain_history must contain columns: {required_columns}. Missing: {missing_columns}")
    
    # Ensure rain arrays are at least 1D
    rain_history = rain_history.copy()
    rain_history['rain'] = rain_history['rain'].apply(np.atleast_1d)
    
    # Check that all input arrays have the same shape
    target_shape = slope.shape
    input_arrays = { # slope excluded, because it is the target shape
        'soil_specific_gravity': soil_specific_gravity,
        'soil_cohesion': soil_cohesion,
        'root_cohesion': root_cohesion,
        'soil_friction': soil_friction,
        'soil_drainage': soil_drainage,
        'infiltration_coefficient': infiltration_coefficient,
        'A_slip': A_slip,
        'soil_porosity': soil_porosity
    }
    for name, arr in input_arrays.items():
        if arr.shape != target_shape:
            raise ValueError(f"All input arrays must have the same shape {target_shape}, found {arr.shape} in '{name}'")
    
    # Verify rain arrays match the input shape
    for idx, rain_arr in rain_history['rain'].items():
        if rain_arr.shape != target_shape:
            raise ValueError(f"rain_history at index [{idx}] (column 'rain') has shape {rain_arr.shape}, expected {target_shape}")
    
    # Check date consistency in rain_history
    if not (rain_history['end_date'] > rain_history['start_date']).all():
        raise ValueError("All end_date must be greater than start_date in rain_history")
    if not rain_history['start_date'].is_monotonic_increasing:
        raise ValueError("start_date column must be in ascending order")
    
    gamma_water = 10 # kN/m3
    rain_history['m_rate'] = [np.full(target_shape, np.nan) for _ in range(len(rain_history))]
    for rh_idx, rh_row in rain_history.iterrows():
        end_idx = rh_idx
        target_end_date = rh_row['end_date']
        predisposing_start_date = target_end_date - predisposing_time_window
        
        # Find start_idx based on predisposing_start_date with tolerance
        mask = abs(rain_history['start_date'] - predisposing_start_date) <= datetime_match_tolerance
        matching_indices = rain_history[mask].index
        
        if len(matching_indices) == 0:
            # No match found, already initialized as NaN, skip to next iteration
            continue
        elif len(matching_indices) > 1:
            # Multiple matches, take the lowest index and warn
            warnings.warn(f"Multiple start_date matches found for rain_history index {rh_idx}, using the lowest index", stacklevel=2)
            start_idx = int(matching_indices.min())
        else:
            start_idx = int(matching_indices[0])
        
        residual_rain = np.full(target_shape, 0.0)
        for curr_idx in range(start_idx, end_idx + 1):
            curr_end_date = rain_history.iloc[curr_idx]['end_date']
            curr_hours_to_target = (target_end_date - curr_end_date).total_seconds() / 3600 # In hours because kt measure unit is m/hour
            residual_rain += rain_history['rain'].iloc[curr_idx] / 1000 * np.exp(-soil_drainage * curr_hours_to_target) # /1000 because rain is in mm
        
        m_rate = np.minimum(residual_rain * infiltration_coefficient / (soil_porosity * instability_depth * (1 - soil_saturation)), 1) # Capped at 1 (maximum infiltration rate = completely saturated soil)

        rain_history.at[rh_idx, 'm_rate'] = m_rate
    
    fs_history = rain_history.loc[:, ['start_date', 'end_date', 'm_rate']].copy()
    fs_history['fs'] = None
    for idx, row in fs_history.iterrows():
        m_rate = row['m_rate']
        c_total = soil_cohesion + root_cohesion + A_slip * soil_saturation * ((1-soil_saturation)**lambda_slip) * ((1-m_rate)**alpha_slip) # Effective cohesion + root cohesion + apparent cohesion
        net_weight = np.cos(np.radians(slope)) * instability_depth * gamma_water * (
            m_rate * (soil_porosity-1) + soil_specific_gravity * (1-soil_porosity) + soil_saturation * soil_porosity * (1-m_rate)
        ) # Net weight of soil (without water pressure).
        total_weight = np.cos(np.radians(slope)) * instability_depth * gamma_water *(
            m_rate * soil_porosity + soil_specific_gravity * (1-soil_porosity) + soil_saturation * soil_porosity * (1-m_rate)
        ) # Total weight of soil (with water pressure)

        fs_history.at[idx, 'fs'] = (net_weight * np.cos(np.radians(slope)) * np.tan(np.radians(soil_friction)) + c_total) / (total_weight * np.sin(np.radians(slope)))

    return fs_history

# %%
