# ZIM-UAV Point Cloud Processing Pipeline (Docker)

A Docker-based pipeline for processing UAV and ALS point cloud data, featuring ground classification, segmentation, roof extraction, street axis detection, curb mapping, and data fusion capabilities.

## 🎯 Overview

This project provides a complete, containerized workflow for processing airborne LiDAR data from both UAV and ALS sources. The pipeline includes:

- **Ground Classification**: CSF (Cloth Simulation Filter) for ground/non-ground separation
- **Machine Learning Segmentation**: RF/XGBoost models for ground, roof, and street classification
- **Roof Extraction**: RANSAC-based plane segmentation with geometric analysis
- **Street Axis Extraction**: Automated centerline extraction with elevation data
- **Curb Detection**: ML-based curb identification and mapping
- **Data Fusion**: UAV-ALS data merging with ICP registration and Gaussian weighting
- **Object Extraction**: Automated extraction of specific features from point clouds

## 📋 Requirements

### System Requirements
- **OS**: Ubuntu 22.04 or compatible Linux distribution
- **RAM**: 64 GB minimum (recommended for large datasets)
- **CPU**: 10+ cores recommended
- **Storage**: 50+ GB free space for data and models
- **Docker**: Docker Engine 20.10+ and Docker Compose v2.0+

### Key Dependencies (Handled by Docker)
- Python 3.10
- PDAL, Open3D, Laspy
- NumPy, SciPy, Pandas
- Scikit-learn, XGBoost, LightGBM
- Geopandas, Rasterio, Shapely
- Jakteristics (point cloud features)
- CSF (Cloth Simulation Filter)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
cd /path/to/workspace
git clone <repository-url>
cd zim_results_docker
git checkout zim-docker
```

### 2. Build and Start Container
```bash
# Build the Docker image and start the container
./start.sh
```

This script will:
- Build the Docker image with all dependencies
- Start the container in detached mode
- Verify all packages are installed

### 3. Access the Container
```bash
# Enter the container shell
./shell.sh
```

### 4. Run Processing Scripts
```bash
# Inside the container
python scripts/csf.py configs/csf_config.yml
```

## 📁 Project Structure

```
zim_results_docker/
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── start.sh                    # Build and start container
├── shell.sh                    # Access container shell
├── stop.sh                     # Stop container
│
├── input/                      # Raw input data
│   ├── *.las                   # Point cloud files
│   └── object_extractions_*/   # Shapefiles
│
├── output/                     # Processing results
│   ├── *_ground.las
│   ├── *_non_ground.las
│   ├── *_roofs.shp
│   ├── *_street.shp
│   └── segmentation/
│
├── configs/                    # YAML configuration files
│   ├── csf_config.yml
│   ├── batch_infe_config.yml
│   ├── roof_extractor.yml
│   ├── curb_inference.yml
│   └── ...
│
├── scripts/                    # Processing modules
│   ├── csf.py                 # Ground classification
│   ├── batch_infe.py          # Segmentation inference
│   ├── roof_extractor.py      # Roof plane extraction
│   ├── axis_extractor.py      # Street axis extraction
│   ├── curb_inference.py      # Curb detection
│   ├── converter.py           # LAS format conversion
│   ├── mesh.py                # DEM generation
│   ├── distance.py            # Point-to-DEM distance
│   ├── features.py            # Feature extraction
│   ├── model_train.py         # Model training
│   └── ...
│
├── shell/                      # Automation scripts
│   ├── run_pipeline.sh        # Complete pipeline
│   ├── run_csf.sh
│   ├── run_segmentation.sh
│   ├── run_roof.sh
│   └── ...
│
├── models/                     # Trained ML models
└── log/                        # Processing logs
```

## 🔧 Configuration

All scripts use YAML configuration files in the `configs/` directory. Modify these files to customize processing parameters.

### Example: CSF Configuration (`configs/csf_config.yml`)
```yaml
pcd_path: "input/point_cloud.las"
cloth_resolution: 1
max_interations: 500
class_threshold: 0.5
sloopsmooth: true
path_out: "output/csf/"
filename: "result"
```

### Example: Segmentation Configuration (`configs/batch_infe_config.yml`)
```yaml
input_las_path: "output/csf/result_ground.las"
model_path: "models/rf_3c_pkl"
output_path: "output/segmented.las"
voxel_downsample: 0.5
probability_threshold: 0.7
```

### Example: Roof Extraction Configuration (`configs/roof_extractor.yml`)
```yaml
las_filename: "output/segmented_roof.las"
roof_shapefile: "input/roofs.shp"
output_dir: "output/roofs/"
min_points_per_roof: 2000
segment_distance_threshold: 0.15
cluster_eps: 0.5
filter_polygon_min_area: 1.0
```

## 🔄 Processing Workflows

### Complete End-to-End Pipeline

Use the automated pipeline script:
```bash
./shell/run_pipeline.sh
```

This runs all steps sequentially:
1. LAS format conversion
2. Ground classification (CSF)
3. DEM generation
4. Feature extraction
5. Segmentation inference
6. Roof extraction
7. Street axis extraction

### Manual Step-by-Step Processing

#### 1. Data Preparation
```bash
# Convert to LAS 1.2 format
docker-compose exec zimuav python scripts/converter.py configs/converter_config.yml

# Optional: Crop to area of interest
docker-compose exec zimuav python scripts/crop_point_cloud.py configs/crop_point_cloud_config.yml
```

#### 2. Ground Classification
```bash
# Separate ground and non-ground points
docker-compose exec zimuav python scripts/csf.py configs/csf_config.yml
```
**Output**: `*_ground.las`, `*_non_ground.las`, `*_hag.las`

#### 3. DEM Generation
```bash
# Generate Digital Elevation Model
docker-compose exec zimuav python scripts/mesh.py configs/mesh_config.yml

# Calculate point-to-DEM distances
docker-compose exec zimuav python scripts/distance.py configs/distance_config.yml
```

#### 4. Segmentation
```bash
# Run ML inference for classification
docker-compose exec zimuav python scripts/batch_infe.py configs/batch_infe_config.yml
```
**Output**: Point cloud with classification (ground=2, roof=6, street=10)

#### 5. Object-Specific Processing

**Roof Extraction**:
```bash
# Split roof points by building
docker-compose exec zimuav python scripts/roof_pcd.py configs/roof_pcd_split.yml

# Extract roof planes with geometry
docker-compose exec zimuav python scripts/roof_extractor.py configs/roof_extractor.yml
```
**Output**: Shapefile with roof planes, inclination, orientation, area

**Street Axis Extraction**:
```bash
docker-compose exec zimuav python scripts/axis_extractor.py configs/axis_config.yml
```
**Output**: Shapefile with street centerlines and elevation

**Curb Detection**:
```bash
docker-compose exec zimuav python scripts/curb_inference.py configs/curb_inference.yml
```
**Output**: Shapefile and LAS with detected curbs

#### 6. Data Fusion (Optional)

For merging UAV and ALS datasets:
```bash
# Point cloud registration
docker-compose exec zimuav python scripts/Simple_ICP.py configs/icp_config.yml

# Identify stable/changed areas
docker-compose exec zimuav python scripts/Stablearea_detection.py configs/stable_area_config.yml

# Gaussian-weighted fusion
docker-compose exec zimuav python scripts/Fusion_gaussian_automatic.py configs/fusion_config.yml
```

## 🐳 Docker Commands

### Container Management
```bash
# Start container
docker-compose up -d

# Stop container
docker-compose down

# Rebuild container (after code changes)
docker-compose build --no-cache

# View container logs
docker-compose logs -f

# Check container status
docker-compose ps
```

### Running Scripts
```bash
# Execute script in container
docker-compose exec zimuav python scripts/script_name.py configs/config.yml

# Run with custom Python command
docker-compose exec zimuav python -c "import pdal; print(pdal.__version__)"

# Run bash script
docker-compose exec zimuav bash shell/run_pipeline.sh
```

### File Transfer
```bash
# Copy file TO container
docker cp local_file.las zimuav:/workspace/input/

# Copy file FROM container
docker cp zimuav:/workspace/output/result.las ./results/
```

## 📊 Processing Modules

### Ground Classification (`csf.py`)
- **Algorithm**: Cloth Simulation Filter
- **Input**: Raw point cloud (.las)
- **Output**: Ground points, non-ground points, height above ground
- **Parameters**: cloth_resolution, class_threshold, slope_smooth

### Segmentation (`batch_infe.py`)
- **Algorithm**: Random Forest / XGBoost
- **Input**: Ground points with features
- **Output**: Classified point cloud (ground/roof/street)
- **Classes**: 2=Ground, 6=Roof, 10=Street

### Roof Extraction (`roof_extractor.py`)
- **Algorithm**: RANSAC + Alpha-shape
- **Input**: Roof points + building shapefile
- **Output**: Individual roof planes with attributes
- **Attributes**: Inclination, orientation, area, point count, geometry

### Street Axis (`axis_extractor.py`)
- **Algorithm**: Skeleton extraction
- **Input**: Street points
- **Output**: Centerlines with elevation profile

### Curb Detection (`curb_inference.py`)
- **Algorithm**: Machine Learning (trained model)
- **Input**: Street/ground points
- **Output**: Curb locations and polylines

### Data Fusion (`Fusion_gaussian_automatic.py`)
- **Algorithm**: Gaussian-weighted blending
- **Input**: Registered UAV + ALS point clouds
- **Output**: Fused point cloud with optimal detail

## 📦 Model Training

### Segmentation Model
```bash
# Extract features from labeled data
docker-compose exec zimuav python scripts/features.py configs/feature_config.yml

# Train classification model
docker-compose exec zimuav python scripts/model_train.py configs/training_config.yml
```

### Curb Detection Model
```bash
docker-compose exec zimuav python scripts/train_curb.py configs/curb_training.yml
```

Place trained models in `models/` directory.

## 📝 Logging

All scripts generate timestamped logs in the `log/` directory:
- Format: `{config_name}.log`
- Level: INFO (includes timing, statistics, errors)
- Console + file output

View logs:
```bash
tail -f log/csf_config.log
```

## 🔍 Troubleshooting

### Container won't start
```bash
# Check Docker service
sudo systemctl status docker

# View build logs
docker-compose build

# Remove old containers
docker-compose down -v
docker system prune
```

### Memory issues
```bash
# Increase Docker memory limit
# Edit: /etc/docker/daemon.json
{
  "default-runtime": "runc",
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Soft": -1
    }
  }
}
```

### Import errors
```bash
# Verify package installation
docker-compose exec zimuav python -c "import laspy; import pdal; import open3d"

# Rebuild container
docker-compose build --no-cache
```

### PROJ database issues
The container includes PROJ data. If errors occur:
```bash
export PROJ_LIB=/opt/conda/share/proj
```

## 📂 Data Requirements

### Input Format
- **Point Clouds**: LAS/LAZ format (preferably LAS 1.2)
- **Shapefiles**: EPSG projection matching point cloud
- **Coordinate System**: Any projected CRS (UTM recommended)

### Sample Data Structure
```
input/
├── site_name_pointcloud.las          # Main point cloud
└── object_extractions_site/
    ├── buildings.shp                  # Building footprints
    ├── streets.shp                    # Street polygons
    └── inlets.shp                     # Drainage features
```

## 🎓 Example Datasets

The pipeline has been tested on:
- **Maibach**: Urban area with buildings, streets, and vegetation
- **Poppenhausen**: Mixed residential area
- **Custom datasets**: Various UAV and ALS combinations

## 📜 License

[Add your license information]

## 👥 Contributors

[Add contributor information]

## 📧 Contact

For questions, issues, or contributions, please contact the development team.

## 🔗 Additional Resources

- Docker Documentation: https://docs.docker.com/
- PDAL Documentation: https://pdal.io/
- Open3D Documentation: http://www.open3d.org/
- Laspy Documentation: https://laspy.readthedocs.io/

---

**Version**: zim-docker branch  
**Last Updated**: January 2026  
**Python Version**: 3.10  
**Docker Base Image**: Ubuntu 22.04
