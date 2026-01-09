# Analysis Environment Object

The `AnalysisEnvironment` object is the central data structure in P-SLIP, defined in `src/config/analysis_init.py`. It serves as the core manager for an analysis session, storing metadata, folder paths, configuration, and dynamic variables. It is initialized or loaded via `m00a_env_init.py` and updated sequentially by other scripts throughout the workflow.

## Overview

The `AnalysisEnvironment` is a Python dataclass that provides a persistent, hierarchical structure for managing all aspects of a P-SLIP analysis. It acts as the "brain" of the analysis, tracking:

- **Project metadata**: Case name, creator information, timestamps, version
- **Folder structure**: Complete directory tree for inputs, outputs, and intermediate files
- **Configuration**: Analysis parameters, settings, and options
- **Data variables**: References to all saved analysis data (DTM, parameters, paths, etc.)
- **System information**: Hardware specs and user tracking

## Purpose and Design Philosophy

The `AnalysisEnvironment` follows several key design principles:

1. **Persistence**: All state is saved to `environment.json` and can be reloaded
2. **Modularity**: Each script updates specific aspects without affecting others
3. **Traceability**: Complete history of who did what, when, and on what system
4. **Flexibility**: Supports both automated and manual configuration
5. **Version control**: Tracks P-SLIP version for compatibility

## Structure

The object is organized into fixed dataclass attributes and dynamic attributes added during analysis:

### Fixed Attributes (Defined in Dataclass)

These attributes are always present and defined in the class structure:

| Attribute | Type | Description | Default |
|-----------|------|-------------|---------|
| `case_name` | str | Name of the analysis case (e.g., "MyStudyArea") | "Not_Defined_Standalone" |
| `creator_user` | str | Username of the creator | Auto-retrieved |
| `creator_specs` | dict | System specs at creation (CPU, RAM, etc.) | Auto-retrieved |
| `creation_time` | str | ISO timestamp of creation | Auto-generated |
| `last_user` | str | Username of last modifier | Auto-updated |
| `last_user_specs` | dict | System specs at last update | Auto-updated |
| `last_update` | str | ISO timestamp of last update | Auto-updated |
| `app_version` | str | P-SLIP version from `version.txt` | Auto-retrieved |
| `folders` | dict | Nested dictionary of all folder paths | From `default_params.py` |
| `config` | dict | Analysis configuration and settings | From `default_params.py` |

### Dynamic Attributes

Added post-initialization by analysis scripts using `setattr()`:

- **`study_area_vars`**: Study area polygon and metadata (from m01a)
- **`dtm_vars`**: Digital terrain model data (from m03a)
- **`parameter_vars`**: Parameter grids and associations (from m04a)
- **`morphology_vars`**: Slope, aspect, curvature grids (from m04b)
- **`rain_recordings_vars`**: Rainfall time-series data (from m04c)
- **`landslide_paths_vars`**: Landslide path data (from m04d)
- **`alert_vars`**: Alert system data (from m07a)
- **Any other script-specific variables**

## Core Methods

### Initialization and Setup

#### `__post_init__()`
- **Purpose**: Post-initialization setup after dataclass creation
- **Logic**: 
  - Ensures `folders` and `config` are dictionaries
  - Deep copies defaults from `default_params.py` if missing
  - Sets base path to current directory if not provided
  - Validates types and structure
- **Effects**: Initializes core structure; raises TypeError if structure invalid
- **Called by**: Automatic dataclass mechanism

#### `create_folder_structure(base_dir: str)`
- **Purpose**: Creates complete directory tree for analysis
- **Logic**:
  1. Validates `base_dir` exists
  2. Creates nested folders based on `ANALYSIS_FOLDER_STRUCTURE` from `default_params.py`
  3. Updates `folders` dict with absolute paths
  4. Creates `input_files.csv` template
  5. Saves environment to JSON
- **Inputs**:
  - `base_dir`: Existing base directory (required)
- **Effects**: 
  - Builds complete analysis directory tree
  - Raises ValueError if base_dir doesn't exist
  - Creates `user_control/` with default CSV templates
- **Called by**: `create_analysis_environment()` in `m00a_env_init.py`

### Serialization and Persistence

#### `to_json(file_path: str)`
- **Purpose**: Serializes object to JSON for persistence
- **Logic**:
  1. Updates metadata (user, specs, timestamp, version)
  2. Converts `__dict__` to JSON with proper formatting
  3. Handles datetime objects and complex types
  4. Writes to specified file path
- **Inputs**:
  - `file_path`: Absolute path to save JSON (typically `environment.json`)
- **Effects**:
  - Overwrites existing file if present
  - Preserves all attributes including dynamic ones
  - Used by scripts after updates (e.g., m01a after study area definition)
- **Example**: `env.to_json('/path/to/analysis/environment.json')`

#### `from_json(cls, file_path: str) -> AnalysisEnvironment` (classmethod)
- **Purpose**: Reconstructs object from JSON file
- **Logic**:
  1. Loads JSON from file
  2. Maps JSON keys to dataclass fields
  3. Sets dynamic attributes using `setattr()`
  4. Validates structure and types
- **Inputs**:
  - `file_path`: Path to JSON file (required)
- **Effects**:
  - Returns reconstructed `AnalysisEnvironment` object
  - Used in `get_analysis_environment()` to load existing analyses
  - Handles missing keys gracefully with warnings
- **Example**: `env = AnalysisEnvironment.from_json('/path/to/environment.json')`

### Variable Management

#### `save_variable(variable_to_save: dict[str, object], variable_filename: str, environment_filename: str = 'environment.json', compression: str = None)`
- **Purpose**: Saves analysis variables to PKL files with metadata tracking
- **Logic**:
  1. Saves dictionary to PKL file in `variables/` folder
  2. Applies compression if specified (gzip)
  3. Updates `config['variables']` with labels and compression info
  4. Saves updated environment to JSON
- **Inputs**:
  - `variable_to_save`: Dictionary to save (required)
  - `variable_filename`: PKL filename (e.g., 'study_area_vars.pkl')
  - `environment_filename`: JSON filename (default: 'environment.json')
  - `compression`: 'gzip' or None (default: None for small files)
- **Effects**:
  - Persists variables for use by subsequent scripts
  - Enables data sharing between analysis steps
  - Compression recommended for large datasets (>10MB)
- **Example**: 
  ```python
  env.save_variable(
      variable_to_save={'study_area': poly, 'metadata': info},
      variable_filename='study_area_vars.pkl',
      compression='gzip'
  )
  ```

#### `load_variable(variable_filename: str) -> dict[str, object]`
- **Purpose**: Loads analysis variables from PKL files
- **Logic**:
  1. Checks `config['variables']` for compression info
  2. Loads from PKL file handling compression if specified
  3. Returns dictionary with all saved data
- **Inputs**:
  - `variable_filename`: PKL filename to load (required)
- **Effects**:
  - Returns complete dictionary with all saved variables
  - Raises FileNotFoundError if file doesn't exist
  - Handles compression automatically based on metadata
- **Example**: `study_data = env.load_variable('study_area_vars.pkl')`

### File Management

#### `add_input_file(file_path: str, file_type: str, file_subtype: str = None, force_add: bool = True) -> tuple[bool, str]`
- **Purpose**: Adds input file to tracking CSV
- **Logic**:
  1. Validates file exists and is readable
  2. Checks for duplicates using file path
  3. Adds row to `input_files.csv` with metadata
  4. Returns success status and file ID
- **Inputs**:
  - `file_path`: Absolute path to input file (required)
  - `file_type`: Type (e.g., 'dtm', 'land_use', 'rain')
  - `file_subtype`: Optional subtype (e.g., 'recordings', 'forecast')
  - `force_add`: Overwrite duplicates if True (default: True)
- **Effects**:
  - Updates `input_files.csv` in `user_control/` folder
  - Returns tuple: (success: bool, file_id: str)
  - Used by data import scripts to track sources
- **Example**:
  ```python
  success, file_id = env.add_input_file(
      file_path='/data/rainfall.csv',
      file_type='rain',
      file_subtype='recordings'
  )
  ```

#### `collect_input_files(file_type: list[str] = None, file_subtype: list[str] = None, file_custom_id: list[str] = None, multi_extension: bool = False)`
- **Purpose**: Copies input files from sources to analysis folder
- **Logic**:
  1. Reads `input_files.csv` and filters by criteria
  2. Copies files to appropriate `inputs/` subfolders
  3. Updates CSV with internal paths
  4. Handles multi-extension files (e.g., .shp with .shx, .dbf)
- **Inputs**:
  - `file_type`: List of types to collect (optional, all if None)
  - `file_subtype`: List of subtypes (optional)
  - `file_custom_id`: List of custom IDs (optional)
  - `multi_extension`: Handle multi-extension files (default: False)
- **Effects**:
  - Populates `inputs/` folder with data files
  - Updates CSV with copied file locations
  - Called by import scripts (e.g., m02a1)
- **Example**:
  ```python
  env.collect_input_files(
      file_type=['land_use', 'soil'],
      multi_extension=True
  )
  ```

### Configuration Management

#### `generate_default_csv()`
- **Purpose**: Creates default configuration CSV templates
- **Logic**:
  1. Reads default templates from `default_params.py`
  2. Creates CSV files in `user_control/` folder
  3. Includes: `parameter_classes.csv`, `analysis_parameters.csv`, etc.
- **Effects**:
  - Provides editable templates for user configuration
  - Enables customization without code changes
  - Called during environment creation
- **Example**: `env.generate_default_csv()`

## JSON Storage Format

The object is stored as `environment.json` in the base folder. This is a flat dictionary containing all attributes (fixed + dynamic):

```json
{
  "case_name": "MyStudyArea",
  "creator_user": "john_doe",
  "creator_specs": {
    "cpu": "Intel i7-10700K",
    "ram_gb": 32,
    "os": "Windows 10"
  },
  "creation_time": "2025-01-15T10:30:00",
  "last_update": "2025-01-20T14:45:00",
  "app_version": "1.2.3",
  "folders": {
    "base": {"path": "/path/to/analysis"},
    "inputs": {
      "path": "/path/to/analysis/inputs",
      "vectors": {"path": "/path/to/analysis/inputs/vectors"},
      "rasters": {"path": "/path/to/analysis/inputs/rasters"}
    },
    "outputs": {
      "path": "/path/to/analysis/outputs",
      "tables": {"path": "/path/to/analysis/outputs/tables"},
      "plots": {"path": "/path/to/analysis/outputs/plots"}
    },
    "variables": {"path": "/path/to/analysis/variables"},
    "user_control": {"path": "/path/to/analysis/user_control"},
    "logs": {"path": "/path/to/analysis/logs"},
    "temp": {"path": "/path/to/analysis/temp"}
  },
  "config": {
    "variables": {
      "study_area_vars.pkl": {
        "labels": ["study_area", "metadata"],
        "compression": "gzip"
      }
    },
    "plotting": {
      "dpi": 300,
      "format": "png"
    }
  },
  "study_area_vars": "/path/to/study_area_vars.pkl",
  "dtm_vars": "/path/to/dtm_vars.pkl"
}
```

### Manual Editing Guidelines

**What you can safely view/edit:**
- `case_name`: Change analysis name
- `folders`: Update paths if moved (use absolute paths)
- `config`: Modify settings (plotting, analysis parameters)
- Metadata fields: Usually best left auto-updated

**What to avoid editing:**
- Dynamic variable references (let scripts manage these)
- `creator_*` and timestamp fields (auto-maintained)
- `app_version` (should reflect actual P-SLIP version)

**Best practices:**
- Always backup `environment.json` before manual edits
- Validate JSON syntax after editing
- Reload environment and run validation scripts
- Document any manual changes in logs

## Usage Patterns

### Creation and Loading

```python
from config.analysis_init import create_analysis_environment, get_analysis_environment

# Create new environment
env = create_analysis_environment(
    base_dir="/path/to/new/analysis",
    case_name="MyLandslideStudy"
)

# Load existing environment
env = get_analysis_environment(base_dir="/path/to/existing/analysis")
```

### Variable Workflow

```python
# Script 1: Save data
study_data = {'polygon': study_poly, 'crs': epsg_code}
env.save_variable(
    variable_to_save=study_data,
    variable_filename='study_area_vars.pkl'
)

# Script 2: Load data
study_data = env.load_variable('study_area_vars.pkl')
study_poly = study_data['polygon']
```

### File Management Workflow

```python
# Add input file
success, file_id = env.add_input_file(
    file_path="/external/data/land_use.shp",
    file_type="land_use",
    file_subtype="top"
)

# Collect all land use files
env.collect_input_files(file_type=["land_use"], multi_extension=True)
```

## Integration with Analysis Workflow

The `AnalysisEnvironment` is the central hub connecting all P-SLIP scripts:

1. **m00a**: Creates/loads environment
2. **m01a**: Adds study area variables
3. **m02a1/m02a2**: Manages input files and property data
4. **m03a**: Saves DTM data
5. **m04a-m04d**: Store parameter, morphology, time-series, and path data
6. **m05a-m05b**: Load and analyze reference points and time-series
7. **m07a**: Load paths and save alert data

Each script typically:
- Loads environment at start
- Loads required variables from previous steps
- Performs analysis
- Saves results as new variables
- Updates environment configuration

## Troubleshooting

### Common Issues

1. **"Environment file not found"**:
   - Solution: Run `m00a_env_init.py` to create environment
   - Check base directory path is correct

2. **"Variable not found"**:
   - Solution: Ensure prerequisite scripts have been run
   - Check `config['variables']` for variable metadata

3. **"Invalid JSON format"**:
   - Solution: Check JSON syntax, restore from backup
   - Avoid manual edits to complex structures

4. **"Permission denied"**:
   - Solution: Check write permissions in analysis folder
   - Ensure files aren't open in other applications

### Debug Tips

- Use `print(env)` to see complete object state
- Check `logs/` folder for detailed error messages
- Validate folder structure with `os.path.exists()`
- Monitor `config['variables']` for data tracking issues

## Related Documentation

- [Folder Structure](folder_structure.md): Detailed directory organization
- [Configuration Guide](../config_guide.md): Setting up analysis parameters
- [default_params.py](../../src/config/default_params.py): Default settings
- [analysis_init.py](../../src/config/analysis_init.py): Implementation details
- [m00a_env_init.md](../scripts/mains/m00a_env_init.md): Environment initialization

This object ensures modular, persistent analyses while integrating seamlessly with `psliptools` for comprehensive data handling.
