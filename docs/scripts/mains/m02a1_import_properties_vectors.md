# m02a1_import_properties_vectors.py

## Purpose
Import and process vector-based property data (land use, soil types, vegetation, infrastructure) by loading vector files, clipping to study area boundaries, and preparing data for parameter association in subsequent analysis steps.

## Detailed Script Logic

### Core Workflow
1. **Environment Loading**: Loads analysis environment and study area boundaries
2. **Source Type Determination**: Identifies property type (land_use, soil, vegetation, etc.)
3. **File Selection**: Prompts user to select vector files from appropriate input directory
4. **Attribute Selection**: Allows user to choose classification field from vector attributes
5. **Geometry Loading**: Imports vector data with proper georeferencing and validation
6. **Spatial Clipping**: Clips geometries to study area polygon
7. **Data Simplification**: Reduces vertex count for complex polygons (performance optimization)
8. **Association File Creation**: Generates CSV template for parameter mapping
9. **Data Persistence**: Saves processed geometries for downstream processing

### Detailed Processing Steps

#### Step 1: Source Type Configuration
- **Logic**: Uses `--source_type` to determine input subfolder and output filename
- **Supported Types**: `land_use`, `soil`, `vegetation`, `infrastructures`, `landslides`
- **Effect**: Organizes inputs by type, enables type-specific processing
- **Subtype Support**: Optional `--source_subtype` for type variants (e.g., soil_top, soil_bottom)

#### Step 2: Vector File Selection
- **Function**: `select_file_prompt()` from psliptools
- **Location**: `inputs/{source_type}/` directory
- **Supported Formats**: Shapefile (.shp), GeoPackage (.gpkg), GeoJSON (.geojson)
- **Interactive**: Lists available files, prompts user selection
- **Effect**: Determines which property dataset to import

#### Step 3: Attribute Field Selection
- **Function**: `get_geo_file_fields()` extracts all attribute fields
- **Logic**: Displays field names and types, prompts user to select classification field
- **Criteria**: Typically categorical fields (land use codes, soil types, etc.)
- **Effect**: Determines how polygons will be classified and grouped

#### Step 4: Geometry Loading and Validation
- **Function**: `load_vectorial_file_geometry()` from psliptools.geometries
- **Parameters**:
  - `file_path`: Selected vector file
  - `field`: User-selected attribute field
  - `poly_bound_geo`: Study area polygon for clipping
  - `mask_out_poly`: True (clips to study area)
  - `convert_to_geo`: True (ensures geographic coordinates)
  - `points_lim`: Maximum vertices per polygon (default 80000)
  - `allow_only_polygons`: True (validates geometry type)
- **Validation**: Checks for valid polygons, proper topology
- **Effect**: Loads only relevant geometries, reduces data volume

#### Step 5: Data Frame Enhancement
- **Adds Columns**:
  - `label`: Copy of class names for user reference
  - `standard_class`: Empty (to be filled in m02a2)
  - `parameter_class`: Empty (to be filled in m02a2)
  - `info`: Empty (for additional metadata)
- **Effect**: Prepares structure for parameter association workflow

#### Step 6: Association CSV Generation
- **Logic**: Creates CSV file excluding geometry column
- **Filename**: `{source_type}_{subtype}_association.csv`
- **Location**: `user_control/` directory
- **Purpose**: Template for user to map property classes to parameters
- **Effect**: Enables manual parameter assignment in next step

#### Step 7: Environment Configuration Update
- **Function**: `env.add_input_file()` registers file in analysis
- **Settings Stored**:
  - `source_mode`: 'shapefile'
  - `source_field`: Selected attribute field
  - `source_subtype`: Subtype if specified
  - `association_filename`: Generated CSV filename
- **Effect**: Maintains analysis provenance and file tracking

## Prerequisites
- **Required**: `m01a_study_area.py` (provides study area polygon for clipping)
- **Files**: Vector files in appropriate `inputs/{source_type}/` directory
- **Environment**: Analysis environment must be initialized

## Inputs / Parameters

### CLI Arguments
- `--base_dir` (string, required):
  - **Options**: Valid directory path containing analysis environment
  - **Effect**: Loads environment and determines input/output paths
  - **Default**: None (prompts interactively)

- `--gui_mode` (boolean flag):
  - **Options**: True/False
  - **Effect**: Reserved for future GUI integration
  - **Default**: False

- `--source_type` (string):
  - **Options**: `land_use`, `soil`, `vegetation`, `infrastructures`, `landslides`
  - **Effect**: Determines input subfolder and output filename structure
  - **Default**: `land_use`
  - **Logic Influence**:
    - Controls which `inputs/` subdirectory to search
    - Determines output PKL filename: `{source_type}_vars.pkl`
    - Affects configuration storage in environment

- `--source_subtype` (string, optional):
  - **Options**: Any user-defined string (e.g., 'top', 'bottom', '2018', '2023')
  - **Effect**: Enables multiple datasets of same type
  - **Default**: None
  - **Logic Influence**:
    - Appends to output filename: `{source_type}_{subtype}_vars.pkl`
    - Allows separate processing of related datasets
    - Example: `soil_top_vars.pkl` vs `soil_bottom_vars.pkl`

- `--points_limit` (integer):
  - **Options**: Any positive integer
  - **Effect**: Maximum vertices per polygon before simplification
  - **Default**: 80000
  - **Logic Influence**:
    - Larger values = more detailed geometries but higher memory usage
    - Smaller values = faster processing but potential loss of detail
    - Warnings issued if polygons exceed limit

### Input Files
**Vector Files** (in `inputs/{source_type}/`):
- **Required Format**: Shapefile (.shp + .dbf + .shx), GeoPackage (.gpkg), GeoJSON (.geojson)
- **Geometry Type**: Polygons only (validated by script)
- **Coordinate System**: Should be compatible with study area CRS
- **Attribute Table**: Must contain at least one categorical field
- **Effect**: Provides property classification data for analysis

### Interactive Prompts
- **Vector File Selection**:
  - **Prompt**: "Name or full path of the {source_type} shapefile"
  - **Options**: Lists all supported files in input directory
  - **Effect**: Determines which dataset to process
  - **Example**: "land_use.shp" or full path

- **Attribute Field Selection**:
  - **Prompt**: "Select the attribute"
  - **Options**: Lists all attribute fields with data types
  - **Effect**: Determines polygon classification scheme
  - **Example**: "LU_CODE" (land use code), "SOIL_TYPE"

## Outputs

### Primary Output
**`{source_type}_vars.pkl`** - Dictionary containing:
```python
{
    'prop_df': gpd.GeoDataFrame  # Clipped and classified property polygons
}
```

**GeoDataFrame Structure**:
- `class_name`: Original attribute values from selected field
- `label`: Copy of class_name for user reference
- `standard_class`: Empty (reserved for m02a2)
- `parameter_class`: Empty (reserved for m02a2)
- `info`: Empty (reserved for additional metadata)
- `geometry`: Shapely polygon geometries (clipped to study area)

### Secondary Output
**`{source_type}_{subtype}_association.csv`** (in `user_control/`):
- **Purpose**: Template for parameter association
- **Columns**: `class_name`, `label`, `standard_class`, `parameter_class`, `info`
- **Usage**: User manually fills in standard_class and parameter_class mappings
- **Effect**: Enables parameter assignment in m02a2

### Effect on Downstream Scripts
- **m02a2**: Uses association CSV to map classes to parameters
- **m04a**: Provides property polygons for parameter indexing to grid
- **m05a**: Property information at reference points
- **m07a**: Property context for alert calculations

### Configuration Updates
- **Input Files Registry**: Adds entry to `env.config['inputs'][source_type]`
- **File Tracking**: Assigns unique custom_id to each input file
- **Settings Storage**: Preserves processing parameters for reproducibility

## Sample CLI Usage

### Basic Land Use Import
```bash
python m02a1_import_properties_vectors.py --base_dir /path/to/case1 --source_type land_use
```

### Soil with Subtype
```bash
python m02a1_import_properties_vectors.py --base_dir /path/to/case1 --source_type soil --source_subtype top
```

### High Detail Vegetation
```bash
python m02a1_import_properties_vectors.py --base_dir /path/to/case1 --source_type vegetation --points_limit 150000
```

### Memory Optimized
```bash
python m02a1_import_properties_vectors.py --base_dir /path/to/case1 --source_type land_use --points_limit 40000
```

## Detailed Effects of Parameter Choices

### Source Type Impact
- **land_use**:
  - Effect: Imports land use/land cover classifications
  - Use Case: Vegetation cover analysis, land use change
  - Parameters: Typically vegetation type, cover density
  - Common Fields: "LU_CODE", "CLASS", "CATEGORY"

- **soil**:
  - Effect: Imports soil type and property data
  - Use Case: Geotechnical analysis, hydrology
  - Parameters: Soil strength, permeability, texture
  - Common Fields: "SOIL_TYPE", "TEXTURE", "GEOLOGY"

- **vegetation**:
  - Effect: Imports vegetation characteristics
  - Use Case: Root reinforcement, evapotranspiration
  - Parameters: Root strength, vegetation density
  - Common Fields: "VEG_TYPE", "COVER", "HEIGHT"

- **infrastructures**:
  - Effect: Imports infrastructure locations
  - Use Case: Risk assessment, vulnerability analysis
  - Parameters: Infrastructure type, condition
  - Common Fields: "INFRA_TYPE", "STATUS", "OWNER"

- **landslides**:
  - Effect: Imports landslide inventory data
  - Use Case: Susceptibility validation, training data
  - Parameters: Landslide type, date, magnitude
  - Common Fields: "LS_TYPE", "DATE", "AREA"

### Subtype Strategy
- **No Subtype** (default):
  - Effect: Single dataset per source type
  - Use Case: Simple analyses, single time period
  - Output: `land_use_vars.pkl`

- **With Subtype**:
  - Effect: Multiple related datasets
  - Use Case: Temporal comparison, layered analysis
  - Examples:
    - `soil_top_vars.pkl` + `soil_bottom_vars.pkl`
    - `land_use_2018_vars.pkl` + `land_use_2023_vars.pkl`
  - Benefit: Enables comparison and change detection

### Points Limit Effects
- **Low Limit (20,000-40,000)**:
  - Effect: Aggressive simplification, fast processing
  - Use Case: Large study areas, preliminary analysis
  - Trade-off: May lose small features, simplified boundaries
  - Memory: Low (suitable for limited RAM)

- **Medium Limit (40,000-80,000)**:
  - Effect: Balanced detail and performance
  - Use Case: Most standard analyses
  - Trade-off: Minimal detail loss, reasonable memory
  - Memory: Moderate (standard workstations)

- **High Limit (80,000-150,000+)**:
  - Effect: Maximum detail preservation
  - Use Case: High-resolution studies, critical features
  - Trade-off: Higher memory usage, slower processing
  - Memory: High (requires sufficient RAM)
  - Risk: May cause memory issues with many polygons

### Attribute Field Selection Impact
- **Categorical Fields** (recommended):
  - Effect: Clear class boundaries, easy parameter mapping
  - Examples: Land use codes, soil types, vegetation classes
  - Benefit: Straightforward parameter association

- **Numeric Fields** (possible but complex):
  - Effect: Continuous values, may need binning
  - Examples: Elevation ranges, age classes
  - Challenge: Requires user decision on binning strategy

- **Text Fields** (use with caution):
  - Effect: Descriptive but may need standardization
  - Examples: Names, descriptions
  - Challenge: Manual cleanup often required

## Code Architecture

### Key Functions
- `main()`: Primary execution function
  - Validates inputs and environment
  - Coordinates file selection and processing
  - Generates outputs and updates configuration

### Data Flow
```mermaid
graph TD
    A[Study Area Polygon] --> B[Select Source Type]
    B --> C[Choose Vector File]
    C --> D[Select Attribute Field]
    D --> E[Load Geometry]
    E --> F[Clip to Study Area]
    F --> G{Simplify?}
    G -->|Points > Limit| H[Simplify Polygons]
    G -->|Within Limit| I[Keep Original]
    H --> J[Add Metadata Columns]
    I --> J
    J --> K[Generate Association CSV]
    K --> L[Save {type}_vars.pkl]
    L --> M[Update Environment Config]
```

### Error Handling
- **Invalid Source Type**: Validates against `KNOWN_OPTIONAL_STATIC_INPUT_TYPES`
- **Missing Files**: Clear error messages with directory suggestions
- **Geometry Errors**: Validates polygon topology, repairs if possible
- **Attribute Issues**: Checks field existence and data types
- **Memory Warnings**: Alerts if polygons exceed point limits

## Integration with P-SLIP Workflow

### Dependencies
- **Requires**: m01a (study area definition)
- **Required by**: m02a2 (parameter association)
- **Feeds into**: m04a (parameter indexing), m05a (reference points)

### Data Persistence
- Saves to `variables/{source_type}_vars.pkl`
- Association CSV in `user_control/`
- Persistent across analysis sessions

### Configuration Chain
1. m00a: Creates environment structure
2. m01a: Defines spatial extent
3. m02a1: Registers property datasets
4. m02a2: Maps properties to parameters
5. m04a: Indexes parameters to grid

## Performance Considerations

### Memory Usage
- **Per Polygon**: ~100 bytes per vertex (geometry + attributes)
- **Typical Dataset**: 1000 polygons × 1000 vertices = ~100 MB
- **With Simplification**: Can reduce by 50-90%

### Computation Time
- **Loading**: Fast (disk I/O limited)
- **Clipping**: Moderate (spatial operations)
- **Simplification**: Variable (depends on complexity)
- **Typical Dataset**: < 30 seconds for 1000 polygons

### Optimization Tips
1. Use subtypes to split large datasets
2. Apply appropriate points_limit for your hardware
3. Pre-clip source data to study area when possible
4. Use simplified geometries for preliminary analysis
5. Process different source types sequentially

## Common Use Cases

### Multi-Layer Soil Analysis
```bash
# Import topsoil
python m02a1_import_properties_vectors.py --base_dir ./case1 --source_type soil --source_subtype top

# Import subsoil
python m02a1_import_properties_vectors.py --base_dir ./case1 --source_type soil --source_subtype bottom
```

### Temporal Land Use Comparison
```bash
# Historical land use
python m02a1_import_properties_vectors.py --base_dir ./case1 --source_type land_use --source_subtype 1990

# Current land use
python m02a1_import_properties_vectors.py --base_dir ./case1 --source_type land_use --source_subtype 2020
```

### High-Resolution Vegetation Mapping
```bash
python m02a1_import_properties_vectors.py --base_dir ./case1 --source_type vegetation --points_limit 200000
```

[← m01a_study_area](m01a_study_area.md) | [m02a2 →](m02a2_read_properties_association.md)