# m04c_import_time_sensitive_data.py

## Purpose
Import, process, and organize time-series data from meteorological stations or satellite sources, including data cleaning, gap filling, temporal alignment, and spatial interpolation to create comprehensive datasets for rainfall and temperature analysis in landslide studies.

## Detailed Script Logic

### Core Workflow
1. **Environment and Configuration**: Loads analysis environment and determines data type configuration
2. **Source Mode Selection**: Chooses between station-based or satellite-based data acquisition
3. **Data Directory Selection**: Identifies folders containing time-series data files
4. **File Association**: Links data files with station metadata (for station mode)
5. **Data Loading and Validation**: Imports time-series data with quality checks
6. **Temporal Processing**: Aligns timestamps, fills gaps, and aggregates data
7. **Spatial Organization**: Structures data by location and time
8. **Configuration Updates**: Registers files and settings in environment
9. **Data Persistence**: Saves processed time-series data for analysis

### Detailed Processing Steps

#### Step 1: Environment and Type Configuration
- **Function**: `obtain_config_idx_and_rel_filename()` determines data type setup
- **Source Types**: `rain`, `temperature` (from `KNOWN_DYNAMIC_INPUT_TYPES`)
- **Source Subtypes**: `recordings`, `forecast` (from `DYNAMIC_SUBFOLDERS`)
- **Effect**: Establishes data type-specific processing parameters

#### Step 2: Source Mode Selection
- **Station Mode**: Point-based measurements from meteorological stations
  - Requires gauge info file with station locations
  - Supports multiple stations with individual data files
  - Enables spatial interpolation between stations
  
- **Satellite Mode**: Gridded data from satellite products
  - Processes raster time series (NetCDF, GeoTIFF sequences)
  - Covers entire study area uniformly
  - Higher spatial resolution but may have temporal gaps

#### Step 3: Data Directory and File Selection
- **Directory Prompt**: `select_dir_prompt()` chooses data folder
- **File Selection**: `select_files_in_folder_prompt()` selects multiple data files
- **Supported Formats**: CSV, NetCDF, climate data formats
- **Effect**: Identifies all relevant time-series data sources

#### Step 4: Station Metadata Processing (Station Mode Only)
- **Gauge Info Loading**: `load_time_sensitive_stations_from_csv()` imports station metadata
- **Required Columns**: `station`, `lon`, `lat` (station name and coordinates)
- **Station Name Cleaning**: `clean_station_name()` ensures filesystem-safe names
- **File-Station Association**: Links data files to specific stations
- **Effect**: Creates spatial reference framework for point data

#### Step 5: Data Loading and Quality Control
- **Function**: `load_time_sensitive_data_from_csv()` processes time-series data
- **Temporal Alignment**: 
  - Rounds timestamps to consistent intervals
  - Handles time zone conversions
  - Ensures consistent datetime formatting
  
- **Data Validation**:
  - Range checking (e.g., rain ≥ 0, -50°C ≤ temperature ≤ 50°C)
  - Outlier detection and handling
  - Missing value identification
  
- **Gap Filling** (Single Station):
  - `zero`: Fill with zeros (appropriate for rainfall)
  - `mean`: Fill with mean value
  - `nearest`: Use nearest available value
  - `previous`/`next`: Forward/backward fill
  - `linear`: Linear interpolation
  - `quadratic`/`cubic`: Polynomial interpolation
  - `auto`: Automatic selection based on data type

#### Step 6: Multi-Station Processing (Station Mode)
- **Spatial Interpolation** (Multiple Stations):
  - `nearest`: Nearest station value
  - `linear`: Linear interpolation between stations
  - `cubic`: Cubic spline interpolation
  - `rbf`: Radial basis function interpolation
  - `idw`: Inverse distance weighting
  - `auto`: Automatic method selection
  
- **Temporal Aggregation**:
  - `sum`: Total accumulation (rainfall)
  - `mean`: Average value (temperature)
  - `min`/`max`: Extreme values
  - Applied when multiple observations within time window

#### Step 7: Data Structure Assembly
- **Time-Series DataFrame**: Organized by station and timestamp
- **Station Metadata**: Spatial coordinates and file associations
- **Quality Flags**: Data quality indicators and processing metadata
- **Effect**: Creates analysis-ready time-series dataset

#### Step 8: Configuration and File Registration
- **Environment Updates**: `env.add_input_file()` registers all data files
- **Settings Storage**: Saves processing parameters for reproducibility
- **File Tracking**: Maintains provenance of all data sources

## Prerequisites
- **Required**: `m01a_study_area.py` (provides study area boundaries)
- **Files**: Time-series data files in appropriate input directories
- **Environment**: Analysis environment must be initialized

## Inputs / Parameters

### CLI Arguments
- `--base_dir` (string, required):
  - **Options**: Valid directory path containing analysis environment
  - **Effect**: Loads environment and determines data paths
  - **Default**: None (prompts interactively)

- `--gui_mode` (boolean flag):
  - **Options**: True/False
  - **Effect**: Reserved for future GUI integration
  - **Default**: False

- `--source_type` (string):
  - **Options**: `rain`, `temperature`
  - **Effect**: Determines data type and processing parameters
  - **Default**: `rain`
  - **Logic Influence**:
    - `rain`: Applies rainfall-specific validation (≥0), zero fill method, sum aggregation
    - `temperature`: Applies temperature-specific validation (-50 to 50°C), linear fill method, mean aggregation

- `--source_subtype` (string):
  - **Options**: `recordings`, `forecast`
  - **Effect**: Determines input subdirectory and output filename
  - **Default**: `recordings`
  - **Logic Influence**: Controls which `inputs/` subfolder to use

- `--source_mode` (string):
  - **Options**: `station`, `satellite`
  - **Effect**: Determines data acquisition and processing method
  - **Default**: `station`
  - **Logic Influence**:
    - `station`: Enables gauge info file selection, multi-station processing
    - `satellite`: Processes gridded data, no station metadata needed

- `--delta_time_hours` (integer, optional):
  - **Options**: Any positive integer (typically 1, 3, 6, 12, 24)
  - **Effect**: Temporal resolution for data aggregation
  - **Default**: None (uses native data resolution)
  - **Logic Influence**: Larger values = coarser temporal resolution, smoother data

- `--round_datetimes_to_nearest_minute` (integer):
  - **Options**: 1, 5, 10, 15, 30, 60 minutes
  - **Effect**: Rounds timestamps to consistent intervals
  - **Default**: 10
  - **Logic Influence**: Smaller values = more precise timing, larger values = standardized intervals

- `--fill_method_single` (string):
  - **Options**: `zero`, `mean`, `nearest`, `previous`, `next`, `linear`, `quadratic`, `cubic`, `auto`
  - **Effect**: Method for filling gaps in single-station data
  - **Default**: `auto` (rain=zero, temperature=linear)
  - **Logic Influence**: Affects data continuity and statistical properties

- `--fill_method_multiple` (string):
  - **Options**: `nearest`, `linear`, `cubic`, `rbf`, `idw`, `auto`
  - **Effect**: Spatial interpolation method for multi-station gaps
  - **Default**: `auto` (uses `nearest`)
  - **Logic Influence**: Determines spatial interpolation quality

- `--aggregation_method` (list of strings):
  - **Options**: `mean`, `sum`, `min`, `max`
  - **Effect**: Temporal aggregation method for overlapping observations
  - **Default**: `sum` for rain, `mean` for temperature
  - **Logic Influence**: Affects cumulative vs average values

- `--last_date` (string, optional):
  - **Format**: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
  - **Effect**: Cutoff date for data import
  - **Default**: None (imports all available data)
  - **Logic Influence**: Limits analysis period, reduces processing time

- `--numeric_data_range` (list of floats, optional):
  - **Format**: `[min_value, max_value]`
  - **Effect**: Valid data range for quality control
  - **Default**: `[0, None]` for rain, `[-50, 50]` for temperature
  - **Logic Influence**: Filters outliers and invalid measurements

- `--rename_csv_data_columns` (boolean flag):
  - **Options**: True/False
  - **Effect**: Enables interactive column name editing
  - **Default**: False
  - **Logic Influence**: Allows standardization of column names

- `--force_consistency` (boolean flag):
  - **Options**: True/False
  - **Effect**: Forces uniform datetime intervals across all stations
  - **Default**: False
  - **Logic Influence**: True ensures temporal alignment, False preserves original sampling

### Input Files

**Gauge Info File** (`gauge_info.csv` - Station Mode Required):
```csv
station,lon,lat,elevation,network
ST001,-5.123,40.456,850,regional_network
ST002,-5.145,40.478,920,regional_network
```
- **Required Columns**: `station`, `lon`, `lat`
- **Optional Columns**: `elevation`, `network`, `location`, etc.
- **Effect**: Provides station locations and metadata
- **Coordinate System**: Geographic (WGS84, EPSG:4326)

**Time-Series Data Files** (CSV format):
```csv
datetime,value,quality
2023-01-01 00:00:00,5.2,good
2023-01-01 01:00:00,3.8,good
2023-01-01 02:00:00,,missing
2023-01-01 03:00:00,7.1,good
```
- **Required Columns**: `datetime` (or auto-detected), numeric data column
- **Optional Columns**: `quality`, `flags`, station-specific fields
- **Effect**: Provides actual time-series measurements
- **Format**: CSV with various datetime formats supported

**Satellite Data Files** (NetCDF/GeoTIFF - Satellite Mode):
- **Format**: NetCDF4, GeoTIFF time series
- **Structure**: Time-varying raster data
- **Effect**: Gridded time-series coverage
- **Requirements**: Proper georeferencing and time dimension

### Interactive Prompts
- **Data Directory Selection**:
  - **Prompt**: "Select directory containing data files"
  - **Options**: Lists available subdirectories in input folder
  - **Effect**: Determines source of time-series data

- **Data Files Selection**:
  - **Prompt**: "Select data files"
  - **Options**: Lists all supported files in directory
  - **Effect**: Identifies specific datasets to import
  - **Multiple Selection**: Supported for batch processing

- **Gauge Info File Selection** (Station Mode):
  - **Prompt**: "Select gauge info file"
  - **Options**: Lists CSV files in directory
  - **Effect**: Links data files to station locations

- **Column Renaming** (if `--rename_csv_data_columns`):
  - **Prompt**: "Enter new labels for datetime and numeric columns"
  - **Options**: Shows current column names
  - **Effect**: Standardizes column naming across datasets

- **File-Station Association** (Station Mode):
  - **Prompt**: "Associate CSV files with stations"
  - **Options**: Lists stations and available files
  - **Effect**: Creates mapping between data files and stations

## Outputs

### Primary Output
**`{source_type}_{source_subtype}_vars.pkl`** - Dictionary containing:

#### Time-Series Data Structure
```python
{
    'datetimes': pd.DatetimeIndex,  # Unified timestamp index
    'data': {
        'station_name': pd.DataFrame  # Time-series data per station
    },
    'stations': pd.DataFrame,  # Station metadata
    'metadata': {
        'source_type': str,
        'source_subtype': str,
        'source_mode': str,
        'processing_params': dict,
        'quality_flags': dict
    }
}
```

#### Detailed Structure Components
- **datetimes**: Unified time index across all stations
- **data**: Dictionary of DataFrames, one per station/source
- **stations**: Station locations, elevations, network info
- **metadata**: Processing parameters and quality information

### Effect on Downstream Scripts
- **m05b**: Provides time-series data for statistical analysis
- **m07a**: Enables time-sensitive alert calculations
- **Analysis**: Foundation for temporal landslide triggering analysis

### Configuration Updates
- **Input Files Registry**: All data files registered in environment
- **Settings Storage**: Processing parameters saved for reproducibility
- **File Tracking**: Maintains data provenance

## Sample CLI Usage

### Basic Rainfall Import
```bash
python m04c_import_time_sensitive_data.py --base_dir /path/to/case1 --source_type rain --source_subtype recordings
```

### Temperature Data with Custom Parameters
```bash
python m04c_import_time_sensitive_data.py --base_dir /path/to/case1 --source_type temperature --delta_time_hours 6 --fill_method_single linear
```

### High-Resolution Rainfall
```bash
python m04c_import_time_sensitive_data.py --base_dir /path/to/case1 --source_type rain --round_datetimes_to_nearest_minute 5 --delta_time_hours 1
```

### Satellite Data Import
```bash
python m04c_import_time_sensitive_data.py --base_dir /path/to/case1 --source_type rain --source_mode satellite
```

### Limited Time Period
```bash
python m04c_import_time_sensitive_data.py --base_dir /path/to/case1 --source_type rain --last_date 2023-12-31
```

## Detailed Effects of Parameter Choices

### Source Type Impact

#### Rainfall Data (`--source_type rain`)
- **Validation Range**: 0 to ∞ (negative values invalid)
- **Default Fill Method**: `zero` (no rain = 0)
- **Default Aggregation**: `sum` (cumulative rainfall)
- **Temporal Patterns**: Event-based, intermittent
- **Spatial Variability**: High (convective vs stratiform)
- **Use Case**: Landslide triggering analysis

#### Temperature Data (`--source_type temperature`)
- **Validation Range**: -50°C to 50°C
- **Default Fill Method**: `linear` (gradual changes)
- **Default Aggregation**: `mean` (average temperature)
- **Temporal Patterns**: Diurnal cycles, seasonal trends
- **Spatial Variability**: Moderate (elevation-dependent)
- **Use Case**: Soil moisture, freeze-thaw cycles

### Source Mode Impact

#### Station Mode (`--source_mode station`)
- **Data Structure**: Point measurements at discrete locations
- **Spatial Coverage**: Limited to station network
- **Temporal Resolution**: Typically high (hourly or better)
- **Data Quality**: Generally high (calibrated instruments)
- **Interpolation Required**: Yes, for spatial coverage
- **Use Case**: Dense station networks, local studies

#### Satellite Mode (`--source_mode satellite`)
- **Data Structure**: Gridded raster data
- **Spatial Coverage**: Complete area coverage
- **Temporal Resolution**: Variable (hourly to daily)
- **Data Quality**: May have gaps (cloud cover, sensor issues)
- **Interpolation Required**: Temporal gap filling only
- **Use Case**: Remote areas, large study regions

### Temporal Resolution Effects

#### High Resolution (`--delta_time_hours 1`)
- **Effect**: Captures short-duration events
- **Benefit**: Detailed temporal analysis
- **Cost**: More data, longer processing
- **Use Case**: Flash flood analysis, intense rainfall

#### Medium Resolution (`--delta_time_hours 6`)
- **Effect**: Balances detail and noise reduction
- **Benefit**: Good for most landslide studies
- **Cost**: Moderate processing requirements
- **Use Case**: General landslide triggering

#### Low Resolution (`--delta_time_hours 24`)
- **Effect**: Daily aggregates, smooths variability
- **Benefit**: Reduces noise, faster processing
- **Cost**: May miss short-duration triggers
- **Use Case**: Long-term trend analysis

### Gap Filling Strategy Impact

#### Zero Fill (`--fill_method_single zero`)
- **Effect**: Assumes no precipitation during gaps
- **Best For**: Rainfall data
- **Risk**: May underestimate cumulative rainfall
- **Use Case**: Conservative estimates

#### Linear Fill (`--fill_method_single linear`)
- **Effect**: Gradual interpolation between known values
- **Best For**: Temperature, gradual changes
- **Risk**: May create artificial trends
- **Use Case**: Continuous variables

#### Nearest Fill (`--fill_method_single nearest`)
- **Effect**: Uses temporally closest available value
- **Best For**: Stable conditions
- **Risk**: May propagate errors
- **Use Case**: Short gaps in consistent data

#### Previous/Next Fill (`--fill_method_single previous`)
- **Effect**: Forward or backward fill
- **Best For**: Persistent conditions
- **Risk**: May extend anomalies
- **Use Case**: Instrument failures

### Spatial Interpolation Effects

#### Nearest Neighbor (`--fill_method_multiple nearest`)
- **Effect**: Uses closest station value
- **Best For**: Dense networks, similar conditions
- **Risk**: Discontinuous spatial patterns
- **Computation**: Fast

#### Linear Interpolation (`--fill_method_multiple linear`)
- **Effect**: Weighted average of nearby stations
- **Best For**: Moderate station density
- **Risk**: May smooth local features
- **Computation**: Moderate

#### Inverse Distance Weighting (`--fill_method_multiple idw`)
- **Effect**: Distance-weighted average
- **Best For**: Variable station density
- **Risk**: Bullseye effect near stations
- **Computation**: Moderate

#### Radial Basis Function (`--fill_method_multiple rbf`)
- **Effect**: Smooth spatial interpolation
- **Best For**: Sparse networks
- **Risk**: May create artifacts
- **Computation**: Slow

## Code Architecture

### Key Functions
- `main()`: Primary execution and coordination
- `clean_station_name()`: Filesystem-safe station names
- `get_numeric_data_ranges()`: Data type-specific validation ranges
- `get_fill_and_aggregation_methods()`: Automatic method selection
- `rename_csv_data_headers()`: Column standardization
- `associate_csv_files_with_gauges()`: File-station linking

### Data Flow
```mermaid
graph TD
    A[Study Area] --> B[Select Source Mode]
    B -->|Station| C[Load Gauge Info]
    B -->|Satellite| D[Load Raster Data]
    C --> E[Select Data Files]
    D --> E
    E --> F[Load Time-Series Data]
    F --> G[Validate & Clean]
    G --> H[Fill Temporal Gaps]
    H --> I{Multiple Stations?}
    I -->|Yes| J[Spatial Interpolation]
    I -->|No| K[Single Station Processing]
    J --> L[Temporal Alignment]
    K --> L
    L --> M[Aggregate if Needed]
    M --> N[Assemble Data Structure]
    N --> O[Save {type}_vars.pkl]
```

### Processing Pipeline
1. **Data Discovery**: Identify available data sources
2. **Quality Control**: Validate and clean raw data
3. **Gap Filling**: Handle missing values appropriately
4. **Spatial Processing**: Interpolate between locations
5. **Temporal Alignment**: Ensure consistent time indexing
6. **Aggregation**: Summarize if needed
7. **Organization**: Structure for analysis

### Error Handling
- **Missing Files**: Clear error messages with file suggestions
- **Invalid Data**: Range checking and outlier detection
- **Format Issues**: Flexible datetime parsing
- **Spatial Errors**: Coordinate validation
- **Memory Issues**: Progress reporting and monitoring

## Integration with P-SLIP Workflow

### Dependencies
- **Requires**: m01a (study area definition)
- **Required by**: m05b (time-series analysis), m07a (time-sensitive alerts)

### Data Flow Chain
1. m01a: Define study area → Spatial extent
2. m04c: Import time-series data ← **This script**
3. m05b: Analyze temporal patterns
4. m07a: Calculate time-dependent alerts

### Configuration Updates
- **Input Files**: All data files registered
- **Processing Parameters**: Saved for reproducibility
- **Quality Metadata**: Tracked for analysis

## Performance Considerations

### Computation Time
- **Data Loading**: Fast (CSV parsing)
- **Gap Filling**: Moderate (interpolation calculations)
- **Spatial Interpolation**: Variable (depends on method and station count)
- **Typical Dataset**: 10 stations × 1 year hourly = 5-15 minutes

### Memory Usage
- **Time-Series Data**: ~100 bytes per observation
- **Typical Dataset**: 10 stations × 8760 hours = ~8 MB
- **Spatial Interpolation**: Additional memory for interpolation matrices
- **Optimization**: Processes one station at a time when possible

### Scalability
- **Station Count**: Linear scaling with number of stations
- **Time Period**: Linear scaling with duration
- **Temporal Resolution**: Linear scaling with sampling frequency
- **Optimization Tips**:
  1. Limit analysis period to relevant time windows
  2. Use appropriate temporal resolution for study goals
  3. Process large datasets in batches
  4. Use efficient interpolation methods for large station networks

## Quality Control

### Data Validation
- **Range Checking**: Ensures physically plausible values
- **Temporal Consistency**: Validates datetime sequences
- **Spatial Consistency**: Checks station coordinate validity
- **Completeness**: Reports data coverage statistics

### Gap Analysis
- **Gap Detection**: Identifies missing data periods
- **Gap Statistics**: Reports gap frequency and duration
- **Gap Distribution**: Analyzes temporal patterns of gaps
- **Quality Flags**: Marks interpolated vs measured values

### Interpolation Quality
- **Cross-Validation**: Tests interpolation accuracy
- **Error Estimation**: Provides uncertainty measures
- **Sensitivity Analysis**: Evaluates method selection impact
- **Visual Inspection**: Plots interpolated vs observed values

## Common Use Cases

### Rainfall Triggering Analysis
```bash
# Import high-resolution rainfall for landslide triggering
python m04c_import_time_sensitive_data.py --base_dir ./case1 --source_type rain --delta_time_hours 1 --fill_method_single zero
```

### Long-Term Climate Analysis
```bash
# Import temperature for seasonal analysis
python m04c_import_time_sensitive_data.py --base_dir ./case1 --source_type temperature --delta_time_hours 24 --fill_method_single linear
```

### Multi-Station Network
```bash
# Process dense station network with spatial interpolation
python m04c_import_time_sensitive_data.py --base_dir ./case1 --source_type rain --fill_method_multiple idw
```

### Satellite Rainfall Product
```bash
# Import satellite-based rainfall estimates
python m04c_import_time_sensitive_data.py --base_dir ./case1 --source_type rain --source_mode satellite
```

[← m04b_morphological_grids](m04b_morphological_grids.md) | [m04d_landslides_paths →](m04d_landslides_paths.md)