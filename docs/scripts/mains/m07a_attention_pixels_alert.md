# m07a_attention_pixels_alert.py

## Purpose

This script implements a comprehensive alert system for landslide susceptibility monitoring. It identifies "attention pixels" (grid cells intersected by landslide paths) and monitors them against time-sensitive triggering data (typically rainfall) to generate alerts when critical thresholds are exceeded.

## Overview

The script provides a sophisticated early warning system that:

- **Identifies attention pixels**: All grid cells that lie on potential landslide paths
- **Associates monitoring stations**: Links each attention pixel to nearest rain stations
- **Detects threshold exceedances**: Monitors time-series data against user-defined thresholds
- **Groups events**: Combines consecutive exceedances into discrete alert events
- **Identifies critical paths**: Ranks landslide paths by risk for each alert event
- **Exports comprehensive results**: Generates detailed CSV reports and saves analysis state

## Prerequisites

- **m04d_landslides_paths.py**: Must be completed (provides landslide path data)
- **m04c_import_time_sensitive_data.py**: Must be completed (provides time-series data)
- **m05b_time_sensitive_analysis.py**: Optional but recommended (provides statistical thresholds)

## Key Inputs

### Required Inputs

- **Landslide paths data**: Loaded from `landslide_paths_vars.pkl`
- **Time-series data**: Loaded from `{source_type}_{source_subtype}_vars.pkl`
- **Analysis environment**: Loaded from base directory

### Optional Inputs

- **Alert thresholds**: CSV file with station-specific thresholds (generated automatically if not provided)
- **Trigger mode**: Method for triggering alerts (currently only 'rainfall-threshold' implemented)
- **Event time tolerance**: Time window for grouping consecutive exceedances
- **Top K paths**: Number of critical paths to identify per activation

## Outputs

### PKL File

- **`alert_vars.pkl`**: Complete analysis state including:
  - `attention_pixels`: DataFrame with all pixels on landslide paths
  - `activation_datetimes`: DataFrame with all alert events
  - `alert_thresholds`: Dictionary with threshold configurations

### CSV Files (in `outputs/tables/attention_pixels_alerts/`)

1. **`attention_pixels.csv`**: All grid cells on landslide paths
   - DTM index, 2D indices, coordinates
   - Associated rain stations for each pixel

2. **`activation_datetimes.csv`**: All alert events with details
   - Event timestamps and grouping
   - Trigger mode and metric information
   - Activated stations and pixels
   - Top critical landslide paths

3. **`{trigger_mode}_alert_thresholds.csv`**: Threshold configuration
   - Station-specific thresholds
   - Alert metric and mode
   - Maximum observed values

4. **`critical_landslide_paths.csv`**: High-risk paths during alerts
   - Path IDs and characteristics
   - Realism scores and starting points

## CLI Usage

```bash
# Basic usage with interactive threshold setup
python m07a_attention_pixels_alert.py --base_dir /path/to/analysis

# Use custom thresholds file
python m07a_attention_pixels_alert.py --base_dir /path/to/analysis --alert_thresholds custom_thresholds.csv

# Custom event grouping tolerance
python m07a_attention_pixels_alert.py --base_dir /path/to/analysis --events_time_tolerance 3 --trigger_mode rainfall-threshold

# Disable critical path identification
python m07a_attention_pixels_alert.py --base_dir /path/to/analysis --top_k_paths_per_activation 0
```

## Detailed Description

This script implements a comprehensive early warning system that combines spatial landslide path data with temporal rainfall monitoring. It identifies which grid cells are at risk, monitors them against rainfall thresholds, and generates detailed alerts when conditions become critical.

For more on alert systems, see [Analysis Structure Guide](../../analysis_structure/).

[← m05b_time_sensitive_analysis](m05b_time_sensitive_analysis.md) | [← Scripts Guide](../scripts_guide.md)

## Script Logic

### Main Function: `main()`

The main function orchestrates the complete alert system:

1. **Input Validation**:
   - Validates `trigger_mode` against `POSSIBLE_TRIGGER_MODES`
   - Validates `alert_thresholds` format if provided
   - Validates `default_thr_mode` against available modes
   - Validates `events_time_tolerance` and `top_k_paths_per_activation`

2. **Environment and Data Loading**:
   - Loads analysis environment with `get_or_create_analysis_environment()`
   - Loads landslide paths data from `landslide_paths_vars.pkl`
   - Identifies unique attention pixels across all DTMs

3. **Attention Pixel Identification**:
   - Extracts all unique grid cells from landslide paths
   - Gets coordinates for each attention pixel
   - Associates rain stations with attention pixels

4. **Trigger Mode Processing** (currently only rainfall-threshold):
   - **Data Source Selection**: Prompts for source subtype if not in GUI mode
   - **Configuration Indexing**: Obtains config using `obtain_config_idx_and_rel_filename()`
   - **Station Association**: Links attention pixels to monitoring stations
   - **Threshold Management**: Generates or loads alert thresholds
   - **Alert Detection**: Identifies threshold exceedances
   - **Event Grouping**: Combines consecutive exceedances into events
   - **Pixel Activation**: Identifies which pixels are activated in each event

5. **Critical Path Identification**:
   - For each alert event, identifies top K critical landslide paths
   - Ranks paths by realism score within activated pixels
   - Handles multiple starting points separately

6. **Results Export**:
   - Saves comprehensive CSV files
   - Updates PKL file with analysis state
   - Returns alert variables dictionary

### Helper Functions

#### `get_top_k_paths(paths_df, dtm, idx_2d, k=3, separate_starting_points=False)`
- **Purpose**: Identifies top-k most realistic landslide paths passing through given pixels
- **Logic**:
  1. Filters paths by DTM and checks which pass through given 2D indices
  2. Ranks paths by `path_realism_score`
  3. If `separate_starting_points=True`, selects top K paths from each starting point
  4. Returns list of path IDs (padded with None if fewer than K paths found)
- **Returns**: List of path IDs (strings)

#### `get_attention_pixel_coordinates(env, attention_pixels_df)`
- **Purpose**: Converts attention pixel indices to geographic coordinates
- **Logic**:
  1. Loads ABG (Aligned Base Grid) data
  2. For each attention pixel, extracts coordinates from corresponding DTM
  3. Creates array of longitude/latitude coordinates
- **Returns**: List of numpy arrays (one per DTM)

#### `get_rain_station_ids(attention_coords, stations_df)`
- **Purpose**: Associates each attention pixel with nearest rain station
- **Logic**:
  1. For each set of attention pixel coordinates
  2. Uses `get_closest_point_id()` to find nearest station
  3. Returns station indices for each pixel
- **Returns**: List of station ID arrays

#### `get_station_pixels_association(attention_pixels_df)`
- **Purpose**: Creates reverse mapping from stations to associated pixels
- **Logic**:
  1. Identifies all unique rain station names
  2. For each station, filters attention pixels that reference it
  3. Creates DataFrame for each station with associated pixels
- **Returns**: Dictionary `{station_name: pixels_dataframe}`

#### `_format_iterable_with_tab(iterable, prefix="\t")`
- **Purpose**: Formats iterables for user display with proper indentation
- **Logic**: Converts lists, arrays, Series, or dicts to formatted strings
- **Returns**: Formatted string

### Flow Diagram

```mermaid
graph TD
    A[Start m07a] --> B[Load Environment & Paths]
    B --> C[Identify Attention Pixels]
    C --> D[Get Pixel Coordinates]
    D --> E[Associate Rain Stations]
    E --> F{Trigger Mode?}
    F -->|rainfall-threshold| G[Load Rainfall Data]
    G --> H{Thresholds Provided?}
    H -->|No| I[Generate Default Thresholds]
    H -->|Yes| J[Load Custom Thresholds]
    I --> K[Detect Threshold Exceedances]
    J --> K
    K --> L[Group into Events]
    L --> M[Identify Activated Pixels]
    M --> N{Find Critical Paths?}
    N -->|Yes| O[Rank Paths by Realism]
    N -->|No| P[Skip Path Ranking]
    O --> Q[Export Results]
    P --> Q
    Q --> R[Save Alert Variables]
    R --> S[Return Alert Data]
    
    F -->|safety-factor| T[Not Implemented Error]
    F -->|machine-learning| T
    T --> S
```

## Inputs and Parameters

### CLI Arguments

| Parameter | Description | Options/Format | Effect | Default |
|-----------|-------------|----------------|---------|---------|
| `--base_dir` | Base directory for analysis | Valid path string | Loads environment from this directory | Current directory |
| `--gui_mode` | Run in GUI mode | Flag (no value) | Not implemented yet | `False` |
| `--trigger_mode` | Alert triggering method | `rainfall-threshold`, `safety-factor`, `machine-learning` | Selects trigger mode (only rainfall-threshold currently implemented) | `rainfall-threshold` |
| `--alert_thresholds` | Custom thresholds file | File path or pandas Series | Uses custom thresholds instead of generating defaults | `None` |
| `--default_thr_mode` | Default threshold mode | `quantiles`, `max-percentage` | Method for generating default thresholds | `quantiles` |
| `--events_time_tolerance` | Event grouping tolerance | Python timedelta | Time window for grouping consecutive exceedances | `5 days` |
| `--top_k_paths_per_activation` | Critical paths per event | Integer ≥ 0 | Number of top paths to identify per alert event | `5` |

### Configuration Constants

```python
POSSIBLE_TRIGGER_MODES = [
    'rainfall-threshold', 
    'safety-factor', 
    'machine-learning'
]

DEFAULT_THRESHOLD_PERC = {
    'quantiles': 0.9975,      # 99.75th percentile
    'max-percentage': 0.75    # 75% of maximum value
}

DEFAULT_ALERT_THR_FILE = {
    'rainfall-threshold': 'rainfall_alert_thresholds.csv',
    'safety-factor': 'safety_factor_alert_thresholds.csv',
    'machine-learning': 'machine_learning_alert_thresholds.csv'
}

REMOVE_OUTLIERS_IN_THRESHOLDS = True  # Uses IQR method for outlier detection
```

### Interactive Prompts

When CLI arguments are not provided:

- **Source subtype selection**: "Select the source subtype:"
  - Options from `DYNAMIC_SUBFOLDERS` (typically `recordings`, `forecast`)
  
- **Alert metric selection**: "Select the alert metric to use for triggering:"
  - Options from available metrics in data (e.g., `precipitation`, `intensity`)
  
- **Alert metric mode selection**: "Select the way you want to consider the alert metric:"
  - Options: `straight_data`, `mobile_average_simple`, `mobile_average_exponential`, etc.

- **Threshold confirmation**: Displays info about max values and default thresholds
  - User can accept defaults or provide custom threshold file

## Effects and Behavior

### Attention Pixel Identification

1. **Path Extraction**: 
   - Iterates through all landslide paths across all DTMs
   - Extracts all unique 2D grid indices from path coordinates
   - Creates comprehensive list of "attention pixels"

2. **Coordinate Conversion**:
   - Converts grid indices to geographic coordinates
   - Handles multiple DTMs with different resolutions
   - Preserves spatial relationships

3. **Station Association**:
   - For each attention pixel, finds nearest rain station
   - Creates bidirectional mapping (pixels→stations, stations→pixels)
   - Enables efficient monitoring of all at-risk locations

### Threshold Management

1. **Default Threshold Generation**:
   - **Quantile Method**: Uses 99.75th percentile of historical data
   - **Max-Percentage Method**: Uses 75% of maximum observed value
   - **Outlier Removal**: Uses IQR method (1.5×IQR) to exclude outliers
   - **Per-Station Thresholds**: Calculates individual thresholds for each station

2. **Threshold File Creation**:
   - Generates CSV with station, max values, and default thresholds
   - Saves to `user_control/` folder
   - User can edit and reuse custom thresholds

3. **Threshold Application**:
   - Loads thresholds as pandas Series indexed by station
   - Applies thresholds to time-series data
   - Creates boolean mask of exceedances

### Alert Detection and Event Grouping

1. **Threshold Exceedance Detection**:
   - Compares time-series data against thresholds
   - Identifies all timestamps where threshold exceeded
   - Records which stations exceeded thresholds

2. **Event Grouping**:
   - Groups consecutive exceedances within `events_time_tolerance`
   - Assigns unique event labels (e.g., `rt1`, `rt2`, `rt3`)
   - Handles multiple stations exceeding simultaneously

3. **Pixel Activation**:
   - For each event, identifies which attention pixels activated
   - Aggregates pixels from all activated stations
   - Removes duplicates within each DTM
   - Records geographic coordinates of activated pixels

### Critical Path Identification

1. **Path Ranking**:
   - For each activated pixel, identifies passing landslide paths
   - Ranks paths by `path_realism_score`
   - Selects top K paths per starting point (if `separate_starting_points=True`)

2. **Path Deduplication**:
   - Collects all unique critical paths across all events
   - Removes duplicates while preserving rankings
   - Creates comprehensive list of high-risk paths

3. **Path Information Extraction**:
   - Extracts complete path information from landslide paths DataFrame
   - Includes geometry, realism scores, starting points
   - Enables detailed analysis of critical paths

### Data Export and Storage

1. **CSV Export**:
   - **Attention pixels**: Complete list of at-risk locations
   - **Activation datetimes**: Detailed event log with all metadata
   - **Alert thresholds**: Threshold configuration for reproducibility
   - **Critical paths**: High-risk landslide paths during alerts

2. **PKL Storage**:
   - Saves complete analysis state for later reloading
   - Enables incremental analysis and updates
   - Preserves all intermediate calculations

## Code Example

```python
from m07a_attention_pixels_alert import main
import datetime as dt

# Run alert system with custom settings
alert_vars = main(
    base_dir="/path/to/analysis",
    trigger_mode="rainfall-threshold",
    events_time_tolerance=dt.timedelta(days=3),
    top_k_paths_per_activation=10
)

# Access results
attention_pixels = alert_vars['attention_pixels']
activation_events = alert_vars['activation_datetimes']
thresholds = alert_vars['alert_thresholds']

# Analyze specific event
event_1 = activation_events[activation_events['event'] == 'rt1']
print(f"Event rt1 activated {len(event_1['activated_pixels'].iloc[0])} pixels")
print(f"Critical paths: {event_1['top_critical_landslide_path_ids'].iloc[0]}")

# Check threshold configuration
rain_thresholds = thresholds['rainfall-threshold']
print(f"Station thresholds: {rain_thresholds['threshold'].to_dict()}")
```

## Integration with Workflow

### Position in Workflow

This script typically runs after:
1. **m04d_landslides_paths.py**: Provides landslide path data
2. **m04c_import_time_sensitive_data.py**: Provides time-series data
3. **m05b_time_sensitive_analysis.py**: Optional - provides statistical thresholds

### Use Cases

- **Early warning systems**: Monitor rainfall against critical thresholds
- **Event analysis**: Investigate historical landslide-triggering events
- **Risk assessment**: Identify high-risk areas and critical paths
- **Operational monitoring**: Real-time alert generation for landslide risk
- **Scenario analysis**: Test different threshold configurations

### Output Usage

The alert system outputs are used for:
- **Operational decision making**: Alert dissemination to authorities
- **Risk communication**: Public warning systems
- **Post-event analysis**: Understanding landslide triggers
- **Model validation**: Comparing predicted vs. actual events
- **Threshold optimization**: Refining alert thresholds based on performance

## Troubleshooting

### Common Issues

1. **"Invalid trigger_mode"**:
   - Solution: Ensure trigger_mode is one of `POSSIBLE_TRIGGER_MODES`
   - Note: Only `rainfall-threshold` is currently implemented
   
2. **"GUI mode not implemented yet"**:
   - Solution: Run script in command-line mode (default)
   
3. **"Invalid source mode. Must be 'station'"**:
   - Solution: Ensure time-series data was imported with station mode
   
4. **"Some alert metric data is empty"**:
   - Solution: Check time-series data completeness
   - Verify station names match between data and configuration

5. **Memory issues with large datasets**:
   - Solution: Reduce `top_k_paths_per_activation` parameter
   - Process fewer events at a time
   - Increase system RAM

### Debug Tips

- Check log file for detailed processing information
- Verify landslide paths data contains valid paths
- Ensure time-series data spans expected time range
- Test with small subset of attention pixels first
- Monitor memory usage during critical path identification

## Related Documentation

- [Analysis Structure Guide](../../analysis_structure/): Understanding alert system architecture
- [Configuration Guide](../../config_guide.md): Setting up alert thresholds
- [m04d_landslides_paths.md](m04d_landslides_paths.md): Landslide path generation
- [m05b_time_sensitive_analysis.md](m05b_time_sensitive_analysis.md): Time-series statistics for thresholds

[← m05b_time_sensitive_analysis](m05b_time_sensitive_analysis.md) | [← Scripts Guide](../scripts_guide.md)