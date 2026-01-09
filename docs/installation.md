# 🔧 Installation Guide

Follow this comprehensive guide to set up P-SLIP on your system. We provide multiple installation methods to suit different environments and use cases.

## 📋 System Requirements

### Minimum Requirements
| Component | Requirement | Notes |
|-----------|-------------|-------|
| **OS** | Windows 10+, macOS 10.14+, Linux (Ubuntu 20.04+) | Cross-platform support |
| **Python** | 3.10 or higher | Critical for library compatibility |
| **RAM** | 8 GB minimum | 20+ GB recommended for large analyses |
| **Storage** | 5 GB free space | For P-SLIP + dependencies + sample data |
| **CPU** | Multi-core processor | 4+ cores recommended for parallel processing |

### Recommended Specifications
- **RAM**: 32 GB for handling large DTMs and complex flow routing
- **CPU**: 8+ cores for faster morphological computations
- **GPU**: NVIDIA GPU with CUDA support (optional, for ML features)
- **Storage**: SSD for faster I/O operations with large rasters

## 🚀 Quick Installation (Recommended)

### Method 1: Conda Installation (Easiest)

**Best for**: Most users, especially on Windows and macOS where GDAL can be challenging to install.

```bash
# 1. Clone the repository
git clone https://github.com/sasa-misia/P-SLIP.git
cd P-SLIP

# 2. Create and activate conda environment
conda create -n pslip python=3.10 -y
conda activate pslip

# 3. Install GDAL via conda (handles complex dependencies)
conda install -c conda-forge gdal geopandas rasterio -y

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install optional dependencies (recommended)
pip install -r requirements_opt.txt

# 6. Verify installation
python -c "import geopandas, rasterio, mayavi; print('✅ P-SLIP installation successful!')"
```

### Method 2: pip/venv Installation

**Best for**: Linux users or those preferring lightweight virtual environments.

```bash
# 1. Clone the repository
git clone https://github.com/sasa-misia/P-SLIP.git
cd P-SLIP

# 2. Create virtual environment
python -m venv pslip_env

# 3. Activate environment
# On Windows:
pslip_env\Scripts\activate
# On Linux/Mac:
source pslip_env/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install system dependencies (Linux)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal
# Fedora/RHEL:
sudo dnf install -y gdal gdal-devel python3-gdal

# 6. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements_opt.txt

# 7. Verify installation
python -c "import geopandas, rasterio; print('✅ P-SLIP installation successful!')"
```

### Method 3: Docker Installation

**Best for**: Consistent environments, containerized deployments, or avoiding local installation issues.

```dockerfile
# Dockerfile
FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone P-SLIP
RUN git clone https://github.com/sasa-misia/P-SLIP.git /opt/pslip

# Set working directory
WORKDIR /opt/pslip

# Create conda environment
RUN conda create -n pslip python=3.10 -y && \
    conda install -c conda-forge gdal geopandas rasterio -y

# Install Python dependencies
RUN /opt/conda/envs/pslip/bin/pip install -r requirements.txt && \
    /opt/conda/envs/pslip/bin/pip install -r requirements_opt.txt

# Set conda environment as default
RUN echo "conda activate pslip" >> ~/.bashrc
ENV PATH /opt/conda/envs/pslip/bin:$PATH

# Default command
CMD ["/bin/bash"]
```

**Build and run**:
```bash
docker build -t pslip:latest .
docker run -it -v /path/to/your/data:/data pslip:latest
```

## 📦 Dependencies Overview

### Core Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | Latest | Numerical computations, array operations |
| **scipy** | Latest | Scientific computing, interpolation, optimization |
| **matplotlib** | Latest | 2D plotting and visualization |
| **shapely** | Latest | Geometric operations, polygon handling |
| **pandas** | Latest | Data manipulation, CSV handling |
| **geopandas** | Latest | Geospatial data handling, GeoDataFrames |
| **fiona** | Latest | Vector file I/O (shapefiles, GeoJSON) |
| **gdal** | Latest | Geospatial data abstraction, raster/vector support |
| **rasterio** | Latest | Clean and fast raster I/O |
| **scikit-image** | Latest | Image processing, morphological operations |
| **scikit-learn** | Latest | Machine learning algorithms |
| **pyproj** | Latest | CRS transformations, coordinate conversions |
| **mayavi** | Latest | 3D visualization, isometric plots |
| **pydantic** | Latest | Data validation, configuration management |
| **chardet** | Latest | Character encoding detection for CSV files |

### Optional Dependencies (`requirements_opt.txt`)

| Package | Purpose | When to Install |
|---------|---------|-----------------|
| **torch** | Deep learning models | For ML-based susceptibility mapping |
| **tensorflow** | Neural network training | For advanced ML workflows |
| **xarray** | Multi-dimensional data arrays | For climate data processing |
| **opencv-python** | Computer vision | For advanced image analysis |
| **open3d** | 3D data processing | For point cloud analysis |
| **joblib** | Parallel processing | For performance optimization |

## 🔍 Detailed Installation Steps

### Step 1: System Preparation

#### Windows
```powershell
# Install Git for Windows
# Download from: https://git-scm.com/download/win

# Install Conda (if not already installed)
# Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# Verify installations
git --version
conda --version
python --version  # Should be 3.10+
```

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git and Conda
brew install git
brew install --cask miniconda

# Verify installations
git --version
conda --version
python --version
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y git build-essential

# Install Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

# Verify installations
git --version
conda --version
python --version
```

### Step 2: Environment Setup

#### Creating Isolated Environment

**Why use isolated environments?**
- Prevents conflicts with existing Python packages
- Ensures reproducible installations
- Makes cleanup and updates easier

```bash
# Create environment with specific Python version
conda create -n pslip python=3.10 -y

# Activate environment
conda activate pslip

# Verify Python version in environment
python --version  # Should show Python 3.10.x
```

### Step 3: Geospatial Dependencies

Geospatial libraries (GDAL, GEOS, PROJ) have complex native dependencies. Conda handles these automatically:

```bash
# Install geospatial stack via conda-forge
conda install -c conda-forge \
    gdal \
    geopandas \
    rasterio \
    fiona \
    shapely \
    pyproj \
    -y

# Verify geospatial installation
python -c "import geopandas; print(f'GeoPandas version: {geopandas.__version__}')"
python -c "import rasterio; print(f'Rasterio version: {rasterio.__version__}')"
```

### Step 4: Core Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Verify core packages
python -c "import numpy, scipy, pandas, matplotlib, sklearn; print('✅ Core packages OK')"
```

### Step 5: Optional Dependencies

Install if you plan to use machine learning features or advanced visualization:

```bash
# Install optional packages
pip install -r requirements_opt.txt

# Note: PyTorch and TensorFlow are large packages
# Consider installing CPU-only versions if no GPU available:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 6: Verification

Run comprehensive verification to ensure all components work correctly:

```python
# verification_test.py
import sys
print(f"Python version: {sys.version}")

# Core imports
try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    import shapely
    import matplotlib.pyplot as plt
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Core import failed: {e}")
    sys.exit(1)

# Geospatial functionality test
try:
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame({'geometry': [Point(0, 0)]})
    print("✅ Geospatial functionality OK")
except Exception as e:
    print(f"❌ Geospatial test failed: {e}")

# Optional imports
try:
    import torch
    import tensorflow as tf
    print("✅ Optional ML packages available")
except ImportError:
    print("⚠️  Optional ML packages not installed (OK if not needed)")

print("\n🎉 P-SLIP installation verification complete!")
```

## 🐛 Troubleshooting Installation Issues

### Common Issues and Solutions

#### Issue 1: GDAL Installation Fails

**Symptoms**: `ERROR: Could not find a version that satisfies the requirement gdal`

**Solutions**:
```bash
# Windows: Use conda
conda install -c conda-forge gdal

# Linux: Install system packages first
sudo apt-get install gdal-bin libgdal-dev python3-gdal

# macOS: Use Homebrew
brew install gdal
```

#### Issue 2: Mayavi Installation Issues

**Symptoms**: `ERROR: Failed building wheel for mayavi`

**Solutions**:
```bash
# Install VTK first (mayavi dependency)
conda install -c conda-forge vtk

# Then install mayavi
pip install mayavi

# Alternative: Use conda for everything
conda install -c conda-forge mayavi
```

#### Issue 3: PROJ/CRS Errors

**Symptoms**: `CRSError: Invalid CRS` or PROJ-related warnings

**Solutions**:
```bash
# Update PROJ and pyproj
conda update -c conda-forge proj pyproj

# Verify PROJ version
python -c "import pyproj; print(pyproj.proj_version_str)"
```

#### Issue 4: Memory Issues During Installation

**Symptoms**: Installation crashes or hangs

**Solutions**:
```bash
# Use pip with memory optimization
pip install --no-cache-dir -r requirements.txt

# Install packages individually
for package in $(cat requirements.txt); do
    pip install $package
done
```

#### Issue 5: Permission Errors

**Symptoms**: `Permission denied` during pip install

**Solutions**:
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER ~/.local
```

## 🎯 Post-Installation Setup

### First-Time Configuration

```bash
# Navigate to P-SLIP directory
cd P-SLIP

# Initialize your first analysis
cd src/scripts
python m00a_env_init.py --base_dir /path/to/your/first_analysis

# This creates the analysis folder structure:
# first_analysis/
# ├── analysis_environment.json
# ├── inputs/
# ├── variables/
# ├── modeling/
# ├── user_control/
# ├── outputs/
# └── logs/
```

### Environment Variables (Optional)

Set these for convenience:

```bash
# Add to ~/.bashrc or ~/.zshrc (Linux/Mac)
export PSLIP_HOME="/path/to/P-SLIP"
export PATH="$PSLIP_HOME/src/scripts:$PATH"

# Windows: Add to Environment Variables in System Properties
PSLIP_HOME=C:\path\to\P-SLIP
PATH=%PSLIP_HOME%\src\scripts;%PATH%
```

### IDE Configuration

#### VS Code
```json
// .vscode/settings.json
{
    "python.pythonPath": "/path/to/miniconda3/envs/pslip/bin/python",
    "python.linting.enabled": true,
    "python.formatting.provider": "autopep8"
}
```

#### PyCharm
1. Open Settings → Project → Python Interpreter
2. Add interpreter → Conda Environment
3. Select existing environment: `pslip`

## 📊 Performance Optimization

### Hardware-Specific Tuning

```python
# performance_config.py
import os
import psutil

# Optimize for your hardware
ram_gb = psutil.virtual_memory().total / (1024**3)
cpu_count = os.cpu_count()

print(f"System RAM: {ram_gb:.1f} GB")
print(f"CPU Cores: {cpu_count}")

# Set environment variables for optimization
if ram_gb >= 32:
    os.environ['P-SLIP_MAX_PATHS'] = '5000000'  # 5M paths
elif ram_gb >= 16:
    os.environ['P-SLIP_MAX_PATHS'] = '2000000'  # 2M paths
else:
    os.environ['P-SLIP_MAX_PATHS'] = '500000'   # 500K paths
```

### GPU Acceleration (Optional)

```bash
# Install CUDA-enabled PyTorch (if you have NVIDIA GPU)
conda install -c conda-forge pytorch-gpu

# Verify GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## 🔄 Updating P-SLIP

```bash
# Pull latest changes
cd P-SLIP
git pull origin main

# Update dependencies
conda activate pslip
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements_opt.txt

# Verify update
python -c "import sys; sys.path.insert(0, 'src'); from config.version_writer import get_version; print(f'P-SLIP version: {get_version()}')"
```

## 🧪 Testing Your Installation

Run the comprehensive test suite:

```bash
cd P-SLIP
python -m pytest tests/ -v

# Or run individual module tests
python -c "import sys; sys.path.insert(0, 'src'); import psliptools; print('✅ psliptools imported successfully')"
```

---

**🎉 Congratulations! You've successfully installed P-SLIP.**

**Next Steps**:
- 📖 Read the [Quick Start Guide](../README.md#quick-start)
- 🚀 Try the [Scripts Guide](scripts_guide.md) for workflow overview
- 🗺️ Explore [Analysis Structure](analysis_structure/) for data organization

[← Back to Index](../index.md) | [🚀 Next: psliptools Guide →](psliptools_guide.md)