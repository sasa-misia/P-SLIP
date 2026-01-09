# 🏔️ P-SLIP: Python Landslide Susceptibility Platform

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen)](docs/index.md)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](README.md)

## 📋 Overview

**P-SLIP** (Python Soil Landslide Investigation Platform) is a complete, fast, and modern **landslide susceptibility analysis tool** built entirely in Python. It processes geospatial data—including Digital Terrain Models (DTMs), vector polygons (study areas, land use, soil properties), and scattered time-sensitive data (e.g., rainfall)—to compute morphological parameters (slopes, curvatures, flow paths), attribute parameters, generate datasets, and perform analyses such as SLIP factor-of-safety calculations, landslide paths, and attention pixel alerts.

This tool supports forecasting soil slips and landslides through a modular, workflow-driven approach with comprehensive GIS capabilities.

## ✨ Key Features

### 🗺️ Geospatial Processing
- **Modular Library**: `psliptools` library with dedicated modules for geometries, rasters, utilities, and scattered data
- **Multi-format Support**: Handles rasters (GeoTIFF, ASC), vectors (Shapefile, GeoJSON), and CSV data
- **CRS Flexibility**: Supports projected/geographic EPSG codes with automatic reprojection
- **Spatial Operations**: Buffering, intersections, clipping, and complex geometric computations

### 🔄 Workflow Management
- **Sequential Scripts**: Modular workflow from environment setup (`m00a`) to alerts (`m07a`)
- **Data Persistence**: PKL-based variable storage for efficient state management
- **Configuration System**: JSON-based analysis environment with flexible parameter control
- **Dependency Tracking**: Automatic validation of script prerequisites

### 📊 Analysis Capabilities
- **Morphological Analysis**: Slope, aspect, curvature, flow routing computations
- **Time-Sensitive Processing**: Rainfall interpolation, mobile averages, event detection
- **Path Simulation**: Landslide path generation with customizable parameters
- **Alert Systems**: Attention pixel identification and risk assessment
- **Parameter Attribution**: Soil and vegetation property assignment

### 💻 Performance & Usability
- **Hardware-Aware**: Monitors RAM/CPU usage, limits computations based on available resources
- **Memory Management**: Efficient chunked processing for large datasets
- **Interactive/CLI**: Both prompt-based and command-line interface options
- **Comprehensive Logging**: Detailed session logs for debugging and monitoring

## 🔗 Predecessor

P-SLIP is an enhanced Python port of the [M-SLIP MATLAB app](https://github.com/sasa-misia/M-SLIP), providing:
- 🚀 **Better Performance**: Optimized Python implementation with NumPy/GeoPandas
- 🔧 **Enhanced Modularity**: Clean separation of concerns with dedicated libraries
- 🌐 **Python Ecosystem**: Integration with standard GIS and scientific Python tools
- 📚 **Comprehensive Documentation**: Detailed guides and troubleshooting

It is designed to eventually replace M-SLIP while maintaining compatibility with existing workflows.

## 🚀 Quick Installation

### Prerequisites
- **Python**: 3.10 or higher
- **RAM**: 8+ GB recommended (16+ GB for large datasets)
- **Storage**: 5+ GB free space for dependencies and analysis outputs

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sasa-misia/P-SLIP.git
   cd P-SLIP
   ```

2. **Create Conda Environment** (Recommended for GDAL/GeoPandas):
   ```bash
   conda create -n pslip python=3.10 -y
   conda activate pslip
   ```

3. **Install Dependencies**:
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # Optional dependencies (for advanced features)
   pip install -r requirements_opt.txt
   ```

4. **Verify Installation**:
   ```bash
   python -c "import geopandas, rasterio, psliptools; print('✅ P-SLIP installation successful!')"
   ```

📖 **Full installation guide**: [docs/installation.md](docs/installation.md)

## 🎯 Quick Start

### Basic Workflow

1. **Initialize Analysis Environment**:
   ```bash
   cd src/scripts
   python m00a_env_init.py --base_dir /path/to/your/analysis/case
   ```

2. **Define Study Area**:
   ```bash
   python m01a_study_area.py --base_dir /path/to/your/analysis/case --source_mode reference_points
   ```

3. **Continue with Sequential Processing**:
   ```bash
   # Import properties and DTM
   python m02a1_import_properties_vectors.py --base_dir /path/to/your/analysis/case
   python m03a_dtm_base_grid.py --base_dir /path/to/your/analysis/case
   
   # Compute parameters and morphology
   python m04a_parameter_indexing.py --base_dir /path/to/your/analysis/case
   python m04b_morphological_grids.py --base_dir /path/to/your/analysis/case
   
   # Generate outputs
   python m07a_attention_pixels_alert.py --base_dir /path/to/your/analysis/case
   ```

### High-Level Workflow

```mermaid
graph TD
    A[🏗️ m00a: Environment Init<br/>📁 Creates analysis structure<br/>💾 Saves: env.pkl] --> B[🗺️ m01a: Study Area<br/>📍 Defines analysis bounds<br/>💾 Saves: study_area_vars.pkl]
    B --> C[🏞️ m02a1/2: Properties<br/>📦 Imports vectors/rasters<br/>💾 Saves: {source_type}_vars.pkl]
    B --> D[🏔️ m03a: DTM Grid<br/>📐 Creates base grids<br/>💾 Saves: dtm_vars.pkl]
    D --> E[📊 m04a: Parameter Indexing<br/>🔢 Assigns soil/veg params<br/>💾 Saves: parameter_vars.pkl]
    D --> F[⛰️ m04b: Morphology<br/>📈 Computes slopes/curvatures<br/>💾 Saves: morphology_vars.pkl]
    D --> G[⏰ m04c: Time-Sensitive<br/>🌧️ Processes rainfall data<br/>💾 Saves: {ts_type}_vars.pkl]
    D --> H[🛤️ m04d: Paths<br/>🧭 Generates flow paths<br/>💾 Saves: landslide_paths_vars.pkl]
    E --> I[📍 m05a: Reference Points<br/>🎯 Creates monitoring points<br/>📄 Outputs: ref_points.csv]
    G --> J[📅 m05b: Time Analysis<br/>📊 Event detection<br/>💾 Updates: {ts_type}_vars.pkl]
    H --> K[🚨 m07a: Alerts<br/>⚠️ Attention pixels<br/>📄 Outputs: alerts.csv]
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style K fill:#ffebee,stroke:#c62828,stroke-width:2px
```

📖 **Detailed workflow guide**: [docs/scripts_guide.md](docs/scripts_guide.md)

## 📚 Full Documentation

Explore the comprehensive documentation in the [docs folder](docs/):

### 🚀 Getting Started
- 📖 [**Index & Overview**](docs/index.md) - Complete platform introduction and navigation
- 🔧 [**Installation Guide**](docs/installation.md) - Detailed setup instructions and requirements
- 🎯 [**Quick Start Guide**](docs/scripts_guide.md) - Workflow overview and execution order

### 🔍 Technical Documentation
- ⚙️ [**Configuration Guide**](docs/config_guide.md) - Analysis environment and parameter configuration
- 🏗️ [**Analysis Structure**](docs/analysis_structure/) - Folder structure and object documentation
  - [Analysis Environment](docs/analysis_structure/analysis_environment.md) - Object structure and methods
  - [Folder Structure](docs/analysis_structure/folder_structure.md) - Directory organization
- 🛠️ [**psliptools Library**](docs/psliptools_guide.md) - Complete library reference

### 📋 Script Documentation
- 📝 [**Script Details**](docs/scripts/) - Comprehensive documentation for each script
  - [Main Scripts](docs/scripts/mains/) - Core workflow scripts (m00a-m07a)
  - [Optional Scripts](docs/scripts/optional/) - Additional analysis tools

### 🐛 Support
- 🐛 [**Troubleshooting Guide**](docs/troubleshooting.md) - Common issues and solutions
- ❓ [**FAQ**](docs/troubleshooting.md) - Frequently asked questions

## 🏗️ Architecture

### Core Components

```
P-SLIP/
├── 📁 src/
│   ├── 🔧 config/           # Configuration management
│   ├── 🛠️ psliptools/       # Core processing library
│   │   ├── geometries/      # Vector operations
│   │   ├── rasters/         # Raster processing
│   │   ├── scattered/       # Point cloud/time-series
│   │   └── utilities/       # Helper functions
│   └── 📜 scripts/          # Workflow scripts
│       ├── mains/           # Core workflow (m00a-m07a)
│       └── optional/        # Additional tools
├── 📚 docs/                 # Comprehensive documentation
└── 📋 requirements.txt      # Dependencies
```

### Key Technologies
- **GIS Processing**: GeoPandas, Rasterio, Shapely, GDAL
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Visualization**: Matplotlib, Mayavi (optional)
- **Configuration**: JSON, PKL serialization
- **CLI Interface**: argparse, click

## 🔬 Use Cases

### 🎯 Primary Applications
- **Landslide Risk Assessment**: Susceptibility mapping and hazard evaluation
- **Early Warning Systems**: Real-time monitoring and alert generation
- **Research & Education**: Academic research and teaching geospatial analysis
- **Infrastructure Planning**: Risk evaluation for construction projects

### 🌍 Real-World Scenarios
- **Post-Fire Debris Flows**: Assessing landslide risk after wildfires
- **Heavy Rainfall Events**: Monitoring and predicting rainfall-induced landslides
- **Seismic Hazard Analysis**: Evaluating earthquake-triggered landslide potential
- **Land Use Planning**: Incorporating landslide risk into development decisions

## 🤝 Development & Contributions

P-SLIP is under active development. We welcome contributions from the community!

### How to Contribute
1. **Report Issues**: Found a bug? Have a feature request? [Open an issue](https://github.com/sasa-misia/P-SLIP/issues)
2. **Submit Pull Requests**: Improvements to code, documentation, or examples
3. **Share Use Cases**: Tell us how you're using P-SLIP in your work
4. **Improve Documentation**: Help make P-SLIP more accessible to users

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/P-SLIP.git
cd P-SLIP

# Create development environment
conda create -n pslip-dev python=3.10 -y
conda activate pslip-dev
pip install -r requirements.txt
pip install -r requirements_opt.txt

# Install development tools
pip install pytest pytest-cov black flake8

# Run tests (when available)
pytest tests/

# Format code
black src/
```

### Contact & Support
- **Issues**: [GitHub Issues](https://github.com/sasa-misia/P-SLIP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sasa-misia/P-SLIP/discussions)
- **Email**: Contact the maintainer for collaboration opportunities

## 📄 License

P-SLIP is released under the MIT License.

## 🙏 Acknowledgments

- **M-SLIP**: The original MATLAB implementation that inspired this project
- **Research Community**: For feedback, testing, and use cases
- **Open Source Projects**: GeoPandas, Rasterio, NumPy, and the broader Python geospatial ecosystem

---

**Thank you for your interest in P-SLIP!** 🎉

*Built with ❤️ for the geospatial analysis community*

[← Back to Documentation](docs/index.md) | [🚀 Quick Start →](docs/scripts_guide.md)