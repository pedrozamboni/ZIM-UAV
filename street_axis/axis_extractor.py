import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from skimage.morphology import skeletonize, binary_erosion, disk
from shapely.geometry import LineString
import logging
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import laspy
from tqdm import tqdm
import argparse
import yaml
import os
import sys


def get_log_filename(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return f'{base}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(get_log_filename(sys.argv[1])),
        logging.StreamHandler()
    ]
)


def parse_args()-> dict:
    parser = argparse.ArgumentParser(description ='CSF filter')
    parser.add_argument('config_file')

    args = parser.parse_args()
    return args

def read_config(file_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Parameters:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The configuration data loaded from the YAML file.
    """
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            logging.error(exc)

# Load the point cloud
def load_point_cloud(las_path):
    """Load LAS/LAZ file and return coordinates array"""
    las = laspy.read(las_path)
    return np.vstack((las.x, las.y, las.z)).transpose()


def shp_skeleton(geometry: gpd.GeoDataFrame, resolution: float, erode_radius: int = 30) -> gpd.GeoDataFrame:
    logging.info("Starting skeleton creation")
    ### create skeleton
    bounds = geometry.total_bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)

    logging.info(f"Rasterizing geometry with dimensions {width}x{height}")
    raster = rasterize(
        [(geom, 1) for geom in geometry.geometry],
        out_shape=(height, width),
        transform=(resolution, 0, bounds[0], 0, -resolution, bounds[3]),
        fill=0,
        dtype=np.uint8
    )

    logging.info(f"Performing binary erosion with radius {erode_radius}")
    eroded = binary_erosion(raster, footprint=disk(erode_radius))  

    logging.info("Creating skeleton")
    skeleton = skeletonize(eroded > 0)
    
    logging.info("Converting skeleton to lines")
    lines = []
    for y in range(height):
        for x in range(width):
            if skeleton[y, x]:
                x_coord = bounds[0] + x * resolution
                y_coord = bounds[3] - y * resolution
                lines.append(LineString([(x_coord, y_coord), (x_coord + resolution, y_coord)]))

    logging.info(f"Created {len(lines)} line segments")
    skeleton_gdf = gpd.GeoDataFrame(geometry=lines, crs=geometry.crs)
    return skeleton_gdf, lines

### old filter function -> complexity n**2, too slow
# def filter_points_by_distance(gdf, min_distance=1.0):
#     logging.info(f"Filtering points with minimum distance of {min_distance}")
#     # Convert to list for easier manipulation
#     points = list(gdf.geometry)
#     filtered_points = []
    
#     # Add first point
#     filtered_points.append(points[0])
    
#     logging.info(f"Processing {len(points)} points")
#     # Check each point against the filtered points
#     for point in points[1:]:
#         # Check if point is far enough from all filtered points
#         distances = [point.distance(fp) for fp in filtered_points]
#         if all(d >= min_distance for d in distances):
#             filtered_points.append(point)
    
#     logging.info(f"Filtered down to {len(filtered_points)} points")
#     # Create new GeoDataFrame with filtered points
#     filtered_gdf = gpd.GeoDataFrame(geometry=filtered_points, crs=gdf.crs)
#     return filtered_gdf

### new func - complexity almost O(n log n) - faster
def filter_points_by_distance(gdf, min_distance=1.0):
    """
    Filter points to ensure minimum distance between them using KD-tree
    
    Args:
        gdf (GeoDataFrame): Input points
        min_distance (float): Minimum distance between points
        
    Returns:
        GeoDataFrame: Filtered points maintaining minimum distance
    """
    logging.info(f"Filtering points with minimum distance of {min_distance}")
    
    # Convert points to numpy array for faster processing
    points = np.array([[p.x, p.y] for p in gdf.geometry])
    
    # Initialize mask for valid points
    mask = np.ones(len(points), dtype=bool)
    filtered_indices = []
    
    # Add first point
    filtered_indices.append(0)
    mask[0] = False
    
    # Create KD-tree with remaining points
    tree = cKDTree(points)
    
    logging.info(f"Processing {len(points)} points")
    
    # While we still have points to process
    while True:
        # Get remaining points
        remaining_points = points[mask]
        if len(remaining_points) == 0:
            break
            
        # Take the first remaining point
        current_point = remaining_points[0]
        current_idx = np.where(mask)[0][0]
        filtered_indices.append(current_idx)
        
        # Find all points within min_distance
        nearby_idx = tree.query_ball_point(current_point, min_distance)
        # Update mask to remove nearby points
        mask[nearby_idx] = False
    
    logging.info(f"Filtered down to {len(filtered_indices)} points")
    
    # Create new GeoDataFrame with filtered points
    filtered_gdf = gpd.GeoDataFrame(
        geometry=gdf.geometry.iloc[filtered_indices].tolist(),
        crs=gdf.crs
    )
    
    return filtered_gdf

# Function to find mean elevation of k nearest points
def get_mean_elevation_of_nearest(point_coords: np.array, point_cloud_coords: laspy.LasData, k:int=10)->np.array:
    logging.debug(f"Getting mean elevation for point {point_coords[:2]} with k={k}")
   
    if not hasattr(get_mean_elevation_of_nearest, 'tree'):
        logging.debug("Creating new KD-tree for point cloud")
        get_mean_elevation_of_nearest.tree = cKDTree(point_cloud_coords[:, :2])
    
    # Query k nearest neighbors
    logging.debug("Querying k nearest neighbors")
    distances, indices = get_mean_elevation_of_nearest.tree.query(
        point_coords[:2].reshape(1, -1),
        k=k,
        workers=-1  # Use all available CPU cores
    )
    
    mean_elevation = np.mean(point_cloud_coords[indices, 2])
    logging.debug(f"Calculated mean elevation: {mean_elevation}")
    return mean_elevation

def get_point_elevation(point_gpd: gpd.GeoDataFrame, pcd: np.array, batch_size: int = 300, k_nearest: int = 10)->gpd.GeoDataFrame:
    logging.info(f"Getting elevations for {len(point_gpd)} points with batch size {batch_size}")
    points_array = np.array([[p.x, p.y] for p in point_gpd.geometry])
    elevations = []
    
    logging.info("Processing points in batches")
    for i in tqdm(range(0, len(points_array), batch_size), desc="Computing elevations"):
        batch = points_array[i:i + batch_size]
        logging.debug(f"Processing batch {i//batch_size + 1} with {len(batch)} points")
        batch_elevations = [get_mean_elevation_of_nearest(p, pcd, k=10) for p in batch]
        elevations.extend(batch_elevations)

    logging.info("Adding elevation data to GeoDataFrame")
    point_gpd['elevation'] = elevations
    return point_gpd

def smooth_elevation_outliers(points_gdf: gpd.GeoDataFrame, threshold: float=2.0, n_neighbors: int=3)->gpd.GeoDataFrame:
    logging.info(f"Starting elevation smoothing with threshold={threshold} and n_neighbors={n_neighbors}")
   
    logging.info("Converting point coordinates to array")
    coords = np.array([[p.x, p.y] for p in points_gdf.geometry])
    
    logging.info("Building KDTree for nearest neighbor search")
    tree = cKDTree(coords)
    
    logging.info("Getting elevation values")
    elevations = points_gdf['elevation'].values
    new_elevations = elevations.copy()
    
    logging.info(f"Processing {len(coords)} points")
    for i in tqdm(range(len(coords)), desc="Smoothing elevations"):
        distances, indices = tree.query(coords[i], k=n_neighbors+1)
        neighbor_elevations = elevations[indices[1:]]
        mean_neighbor_elevation = np.mean(neighbor_elevations)
        
        if abs(elevations[i] - mean_neighbor_elevation) > threshold:
            logging.debug(f"Point {i} elevation adjusted from {elevations[i]} to {mean_neighbor_elevation}")
            new_elevations[i] = mean_neighbor_elevation
    
    logging.info("Elevation smoothing completed")
    return new_elevations


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting main processing")
    try:
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")

        logging.info("Loading input shapefile")
        gdf = gpd.read_file(config['input_shapefile'])
        logging.debug(f"Loaded {len(gdf)} features from shapefile")
        
        logging.info("Loading point cloud data")
        pcd = load_point_cloud(config['point_cloud_path'])
        logging.debug(f"Loaded point cloud with {len(pcd)} points")

        output_dir = config['output_directory']
        logging.info(f"Output directory set to: {output_dir}")

        logging.info("Filtering street features")
        filtered_gdf = gdf[gdf['type'].str.lower().isin(['street_paved','street_unpaved'])]
        logging.debug(f"Filtered to {len(filtered_gdf)} street features")
        
        logging.info("Merging geometries")
        merged_geom = unary_union(filtered_gdf.geometry)
        merged_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=gdf.crs)
        
        logging.info("Creating skeleton")
        skeleton_gdf,lines = shp_skeleton(merged_gdf, resolution=config['skeleton_resolution'], erode_radius=config['erode_radius'])
        logging.debug(f"Generated {len(lines)} line segments")

        logging.info("Calculating midpoints")
        midpoints = [line.interpolate(0.5, normalized=True) for line in lines]
        unique_coords = list({(pt.x, pt.y): pt for pt in midpoints}.values())
        points_gdf = gpd.GeoDataFrame(geometry=unique_coords, crs=merged_gdf.crs)
        logging.debug(f"Generated {len(points_gdf)} unique midpoints")

        logging.info("Filtering points by distance")
        filtered_points_gdf = filter_points_by_distance(points_gdf, min_distance=config['min_distance'])
        logging.info(f"Original points: {len(points_gdf)}")
        logging.info(f"Filtered points: {len(filtered_points_gdf)}")
        
        logging.info("Getting point elevations")
        points_with_elevation = get_point_elevation(filtered_points_gdf, pcd, batch_size=config['batch_size'], k_nearest=config['k_nearest'])

        logging.info("Smoothing elevations")
        smoothed_elevations = smooth_elevation_outliers(
            points_with_elevation,
            threshold=0.5, 
            n_neighbors=20)
        
        logging.info(f"Points before smoothing: {len(points_with_elevation)}")
        logging.info(f"Points after smoothing: {len(smoothed_elevations)}")

        logging.info("Saving results to file")
        points_with_elevation['elevation_smoothed'] = smoothed_elevations
        points_with_elevation.to_file(f"{output_dir}/{config['filename']}.shp")
        logging.info("Processing completed")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()