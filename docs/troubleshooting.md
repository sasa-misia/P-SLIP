# 🐛 Troubleshooting Guide

Common issues and solutions for P-SLIP. Check logs in `logs/` directory for detailed error traces and debugging information.

## 🚨 Quick Problem Solver

| Issue Category | Quick Fix | Detailed Section |
|----------------|-----------|------------------|
| **Installation errors** | Use conda for GDAL: `conda install -c conda-forge gdal` | [Installation Issues](#installation-issues) |
| **Missing files** | Run scripts in order, check `variables/` folder | [File & Path Issues](#file--path-issues) |
| **CRS/projection errors** | Reproject inputs to study area CRS | [CRS & Coordinate Issues](#crs--coordinate-issues) |
| **Memory problems** | Reduce grid resolution, increase RAM | [Memory & Performance Issues](#memory--performance-issues) |
| **Script failures** | Check prerequisites, validate inputs | [Script Execution Issues](#script-execution-issues) |
| **Data quality issues** | Validate file formats, check logs | [Data Quality Issues](#data-quality-issues) |

## 🔧 Installation Issues

### Issue 1: GDAL Installation Fails

**🐛 Symptoms**: 
```
ERROR: Could not find a version that satisfies the requirement gdal
ERROR: No module named '_gdal'
```

**✅ Solutions**:

**Method 1: Conda (Recommended)**
```bash
conda create -n pslip python=3.10 -y
conda activate pslip
conda install -c conda-forge gdal geopandas rasterio -y
pip install -r requirements.txt
```

**Method 2: System Packages (Linux)**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal

# Fedora/RHEL
sudo dnf install -y gdal gdal-devel python3-gdal

# Then install Python packages
pip install -r requirements.txt
```

**🔍 Verification**:
```bash
python -c "import rasterio; print(f'Rasterio version: {rasterio.__version__}')"
python -c "import geopandas; print(f'GeoPandas version: {geopandas.__version__}')"
gdalinfo --version
```

### Issue 2: Mayavi Installation Issues

**🐛 Symptoms**:
```
ERROR: Failed building wheel for mayavi
ERROR: No module named 'vtk'
```

**✅ Solutions**:

```bash
# Install VTK first (mayavi dependency)
conda install -c conda-forge vtk

# Then install mayavi
pip install mayavi

# Alternative: Install everything via conda
conda install -c conda-forge mayavi
```

### Issue 3: PROJ/CRS Errors

**🐛 Symptoms**:
```
CRSError: Invalid CRS
pyproj.exceptions.CRSError: Invalid CRS
```

**✅ Solutions**:

```bash
# Update PROJ and pyproj
conda update -c conda-forge proj pyproj

# Verify PROJ version
python -c "import pyproj; print(pyproj.proj_version_str)"

# Reinstall if needed
pip install --upgrade pyproj
```

## 📁 File & Path Issues

### Issue 4: Missing `analysis_environment.json` or `vars.pkl`

**🐛 Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'analysis_environment.json'
FileNotFoundError: variables/xxx_vars.pkl not found
```

**✅ Solutions**:

**Check Script Execution Order**:
```bash
# Always run scripts sequentially
cd src/scripts

# 1. Initialize environment first
python m00a_env_init.py --base_dir /path/to/analysis

# 2. Check that environment was created
ls /path/to/analysis/analysis_environment.json

# 3. Run subsequent scripts in order
python m01a_study_area.py --base_dir /path/to/analysis
python m02a1_import_properties_vectors.py --base_dir /path/to/analysis
# ... continue sequence
```

**Verify Folder Structure**:
```bash
# Check analysis directory structure
ls -la /path/to/analysis/

# Expected structure:
# analysis_environment.json
# inputs/
# variables/
# modeling/
# user_control/
# outputs/
# logs/
```

**Recover Missing PKL Files**:
```python
# Check what PKL files exist
import os
variables_dir = '/path/to/analysis/variables/'
pkl_files = [f for f in os.listdir(variables_dir) if f.endswith('.pkl')]
print(f"Found PKL files: {pkl_files}")

# If missing, re-run the script that should create it
# Example: If morphology_vars.pkl missing, re-run m04b
```

### Issue 5: FileNotFoundError or Missing Columns

**🐛 Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'inputs/xxx/file.xxx'
ValueError: Missing required columns: ['lon', 'lat', 'id']
```

**✅ Solutions**:

**Validate Input File Structure**:
```bash
# Check input_files.csv for registered files
cat /path/to/analysis/inputs/input_files.csv

# Verify file paths exist
while IFS=, read -r custom_id path type subtype internal; do
    if [ -f "$path" ]; then
        echo "✓ $custom_id: $path"
    else
        echo "✗ $custom_id: $path (MISSING)"
    fi
done < /path/to/analysis/inputs/input_files.csv
```

**Check CSV Column Requirements**:

**Reference Points CSV** (must have: `lon`, `lat`, `id`):
```csv
lon,lat,id,date,info
12.4924,41.8902,PT001,2023-06-15,"Monitoring point 1"
13.3615,43.7311,PT002,2023-06-16,"Monitoring point 2"
```

**Input Files CSV** (must have: `custom_id`, `path`, `type`, `subtype`, `internal`):
```csv
custom_id,path,type,subtype,internal
my_dtm,/path/to/dtm.tif,dtm,,False
rain_data,/path/to/rain.csv,rain,recordings,False
```

**Fix Missing Files**:
```python
from src.config.analysis_init import get_analysis_environment

env = get_analysis_environment('/path/to/analysis')

# Re-scan inputs directory
env.collect_input_files()

# Or manually add file
env.add_input_file(
    custom_id='my_file',
    path='/correct/path/to/file.xxx',
    type='file_type',
    subtype='',
    internal=False,
    force_add=True
)
```

## 🌐 CRS & Coordinate Issues

### Issue 6: CRS/Projection Mismatch

**🐛 Symptoms**:
```
ValueError: CRS mismatch between geometries
rasterio._err.CPLE_AppDefinedError: Cannot find coordinate operations
```

**✅ Solutions**:

**Check Current CRS**:
```python
import geopandas as gpd
import rasterio

# Check vector CRS
gdf = gpd.read_file('input.shp')
print(f"Vector CRS: {gdf.crs}")

# Check raster CRS
with rasterio.open('input.tif') as src:
    print(f"Raster CRS: {src.crs}")

# Check study area CRS
from src.config.analysis_init import get_analysis_environment
env = get_analysis_environment('/path/to/analysis')
study_area_crs = env.config['study_area']['crs']
print(f"Study area CRS: {study_area_crs}")
```

**Reproject to Study Area CRS**:
```python
import geopandas as gpd

# Load and reproject vector data
gdf = gpd.read_file('input.shp')
target_crs = 'EPSG:32633'  # UTM Zone 33N (example)
gdf_projected = gdf.to_crs(target_crs)

# Save reprojected data
gdf_projected.to_file('input_reprojected.shp')

# Update input_files.csv with new path
```

**Batch Reproject Multiple Files**:
```python
import geopandas as gpd
import os

target_crs = 'EPSG:32633'
input_dir = 'inputs/'
output_dir = 'inputs/reprojected/'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.shp'):
        gdf = gpd.read_file(os.path.join(input_dir, filename))
        gdf_projected = gdf.to_crs(target_crs)
        gdf_projected.to_file(os.path.join(output_dir, filename))
        print(f"✓ Reprojected {filename}")
```

### Issue 7: Coordinate Shape Mismatch

**🐛 Symptoms**:
```
ValueError: shapes do not match
ValueError: operands could not be broadcast together
```

**✅ Solutions**:

**Verify Grid Consistency**:
```python
import numpy as np

# Check DTM grid shapes
print(f"DTM shape: {dtm.shape}")
print(f"X grid shape: {x_grid.shape}")
print(f"Y grid shape: {y_grid.shape}")

# Ensure consistency
assert dtm.shape == x_grid.shape == y_grid.shape, "Grid shapes must match"

# If mismatch, regenerate grids
from psliptools.rasters import create_grid_from_bbox
x_grid_new, y_grid_new, profile = create_grid_from_bbox(bbox, resolution)
```

## 💾 Memory & Performance Issues

### Issue 8: Out of Memory / RAM Limit

**🐛 Symptoms**:
```
MemoryError: Unable to allocate array
Killed process (out of memory)
```

**✅ Solutions**:

**Check Current Memory Usage**:
```python
import psutil

# Check available RAM
ram_gb = psutil.virtual_memory().total / (1024**3)
available_gb = psutil.virtual_memory().available / (1024**3)
print(f"Total RAM: {ram_gb:.1f} GB")
print(f"Available RAM: {available_gb:.1f} GB")

# Estimate memory requirements
dtm_pixels = dtm.shape[0] * dtm.shape[1]
estimated_mb = (dtm_pixels * 8 * 10) / (1024**2)  # 10 arrays * 8 bytes/double
print(f"Estimated memory needed: {estimated_mb:.1f} MB")
```

**Reduce Memory Usage**:

**Option 1: Reduce Grid Resolution**
```python
from psliptools.rasters import resample_raster

# Resample to lower resolution
dtm_lowres = resample_raster(
    in_raster=dtm,
    in_profile=profile,
    new_size=[dtm.shape[0]//2, dtm.shape[1]//2],  # Half resolution
    resample_method='bilinear'
)
```

**Option 2: Process in Chunks**
```python
# Process large rasters in chunks
chunk_size = 1000  # rows per chunk

for i in range(0, dtm.shape[0], chunk_size):
    chunk = dtm[i:i+chunk_size, :]
    # Process chunk
    process_chunk(chunk)
```

**Option 3: Use Memory-Efficient Settings**
```python
# Limit path computation
max_paths = 500000  # Reduce from default

# Use compression for PKL files
env.save_variable(data, 'vars.pkl', compress=True)
```

**Monitor Memory Usage**:
```bash
# Monitor RAM usage during execution
while true; do
    ps aux | grep python | grep -v grep | awk '{print $6/1024 " MB"}'
    sleep 5
done
```

## 🚀 Script Execution Issues

### Issue 9: Too Many Polygon Points (>80k)

**🐛 Symptoms**:
```
Warning: Polygon has >80,000 points, may cause performance issues
```

**✅ Solutions**:

**Simplify Polygons**:
```python
from shapely.geometry import Polygon
from shapely.ops import simplify

# Simplify polygon while preserving shape
tolerance = 1.0  # Adjust based on coordinate system units
simplified_polygon = simplify(complex_polygon, tolerance=tolerance)

print(f"Original points: {len(complex_polygon.exterior.coords)}")
print(f"Simplified points: {len(simplified_polygon.exterior.coords)}")
```

**Clip to Study Area First**:
```python
from psliptools.geometries import intersect_polygons

# Clip complex polygon to study area
clipped = intersect_polygons(complex_polygon, study_area)

# Resulting polygon will be smaller and have fewer points
```

### Issue 10: Polygon No Intersection

**🐛 Symptoms**:
```
Warning: No intersection between polygons
ValueError: Study area polygon is empty
```

**✅ Solutions**:

**Check Polygon Extents**:
```python
from psliptools.geometries import get_polygon_extremes

# Get bounds of each polygon
bounds1 = get_polygon_extremes(polygon1)
bounds2 = get_polygon_extremes(polygon2)

print(f"Polygon 1 bounds: {bounds1}")
print(f"Polygon 2 bounds: {bounds2}")

# Check for overlap
if bounds1[1] < bounds2[0] or bounds1[0] > bounds2[1]:  # X bounds
    print("⚠️  No overlap in X direction")
if bounds1[3] < bounds2[2] or bounds1[2] > bounds2[3]:  # Y bounds
    print("⚠️  No overlap in Y direction")
```

**Expand Study Area**:
```python
from psliptools.geometries import add_buffer_to_polygons

# Add buffer to ensure intersection
expanded_study_area = add_buffer_to_polygons(
    [study_area],
    buffer_distance=1000  # 1km buffer (adjust units)
)[0]
```

## 📊 Data Quality Issues

### Issue 11: Invalid NoData/Dtype in Rasters

**🐛 Symptoms**:
```
ValueError: Invalid NoData value
TypeError: Cannot handle raster with dtype
```

**✅ Solutions**:

**Validate and Fix Raster**:
```python
import rasterio
import numpy as np

# Check raster properties
with rasterio.open('input.tif') as src:
    print(f"Data type: {src.dtypes[0]}")
    print(f"NoData value: {src.nodata}")
    print(f"CRS: {src.crs}")
    data = src.read(1)

# Fix NoData values
data_fixed = np.where(data == old_nodata, new_nodata, data)

# Fix data type
data_float = data.astype(np.float32)

# Save corrected raster
profile = src.profile
profile.update(dtype=rasterio.float32, nodata=new_nodata)

with rasterio.open('input_fixed.tif', 'w', **profile) as dst:
    dst.write(data_float, 1)
```

### Issue 12: Missing or Invalid Data in CSV Files

**🐛 Symptoms**:
```
ValueError: Could not parse datetime
KeyError: 'required_column'
```

**✅ Solutions**:

**Validate CSV Structure**:
```python
import pandas as pd
from psliptools.utilities import read_generic_csv

# Read with automatic encoding detection
df = read_generic_csv('input.csv')

# Check required columns
required_cols = ['lon', 'lat', 'id']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    print(f"Available columns: {df.columns.tolist()}")

# Check for NaN values
nan_counts = df.isna().sum()
if nan_counts.any():
    print(f"⚠️  NaN values found:\n{nan_counts[nan_counts > 0]}")
```

**Fix Common CSV Issues**:
```python
# Fix encoding issues
with open('input.csv', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix delimiter issues
df = pd.read_csv('input.csv', sep=';')  # or ',' or '\t'

# Fix decimal separators
df['value'] = df['value'].str.replace(',', '.').astype(float)

# Save corrected CSV
df.to_csv('input_fixed.csv', index=False)
```

## 🔍 Debugging & Log Analysis

### Understanding Log Files

**Log Location**: `logs/user_{case_name}_session_{timestamp}.log`

**Common Log Patterns**:

```log
# Successful execution
INFO - === Module m01a_study_area.py ===
INFO - Loading study area from inputs/study_area/boundary.shp
INFO - Study area loaded successfully: 1250.5 km²
INFO - CRS: EPSG:32633
INFO - Saving study_area_vars.pkl

# Error patterns
ERROR - FileNotFoundError: inputs/study_area/boundary.shp not found
WARNING - CRS mismatch: input EPSG:4326, study area EPSG:32633
ERROR - MemoryError: Unable to allocate 15.2 GiB
```

**Analyze Logs for Issues**:
```bash
# Search for errors
grep -i "error\|exception" logs/*.log

# Search for warnings
grep -i "warning" logs/*.log

# Check specific script execution
grep -A 10 "m01a_study_area.py" logs/*.log

# Monitor log in real-time
tail -f logs/user_*.log
```

### Performance Monitoring

**Monitor System Resources**:
```python
import psutil
import time

def monitor_resources(interval=10):
    """Monitor RAM and CPU usage"""
    while True:
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        print(f"RAM: {ram.percent}% used ({ram.used/(1024**3):.1f} GB)")
        print(f"CPU: {cpu}% used")
        print("-" * 40)
        
        time.sleep(interval)

# Run in separate thread
import threading
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()
```

## 🆘 Getting Help

### Self-Service Troubleshooting

1. **Check Documentation**
   - [Installation Guide](installation.md) - Setup issues
   - [Scripts Guide](scripts_guide.md) - Workflow issues
   - [Configuration Guide](config_guide.md) - Environment issues

2. **Verify System Requirements**
   ```bash
   # Check Python version
   python --version  # Should be 3.10+
   
   # Check key dependencies
   python -c "import geopandas, rasterio, psliptools; print('✅ Dependencies OK')"
   
   # Check available RAM
   python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total/(1024**3):.1f} GB')"
   ```

3. **Validate Installation**
   ```bash
   # Run verification script
   python -c "import sys; sys.path.insert(0, 'src'); import psliptools; print('✅ P-SLIP OK')"
   ```

### Contact Support

If issues persist:
1. **Collect Debug Information**:
   ```bash
   # System info
   python -c "import platform; print(platform.platform())"
   
   # Package versions
   pip list | grep -E "geopandas|rasterio|shapely|gdal"
   
   # Log files
   tail -n 50 logs/*.log
   ```

2. **Report Issues**:
   - Include complete error messages
   - Attach relevant log files
   - Provide system information
   - Describe steps to reproduce

---

**🎯 Quick Links**:
- 📖 [Installation Guide](installation.md) - Setup troubleshooting
- 🔧 [Configuration Guide](config_guide.md) - Environment issues
- 🚀 [Scripts Guide](scripts_guide.md) - Workflow troubleshooting

[← Back to Index](../index.md) | [🚀 Next: Main README →](../README.md)