import pdal
import json
import geopandas as gpd
import logging
from shapely.geometry import MultiPolygon
import numpy as np 
from shapely.geometry import MultiPolygon
import open3d as o3d
import copy
import alphashape
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
import pandas as pd
import sys
import os 
import argparse
import yaml

def get_log_filename(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return os.path.join('log', f'{base}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('roof_extractor_v2.log'),
        logging.StreamHandler()
    ]
)

def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize points to range [0,1]
    
    Args:
        points: Nx2 array of points
        
    Returns:
        tuple containing:
            - normalized_points (np.ndarray): Nx2 array of normalized points 
            - mins (np.ndarray): array of minimum values for each dimension
            - ranges (np.ndarray): array of ranges (max-min) for each dimension
    """
    
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    
    normalized_points = (points - mins) / ranges 
    return normalized_points, mins, ranges


def process_roof_polygons(las_path: str, roof_gdf: gpd.GeoDataFrame) -> dict:
    """
    Process each roof polygon and return clipped point clouds
    
    Args:
        las_path: Path to input LAS file
        roof_gdf: GeoDataFrame containing roof polygons
    
    Returns:
        dict: Dictionary with polygon index as key and clipped point cloud array as value
    """
    results = {}
    
    # Process each polygon
    for idx, row in roof_gdf.iterrows():
        print(f"Processing roof {idx+1}/{len(roof_gdf)}")
        
        # Create single polygon GeoDataFrame
        polygon_gdf = gpd.GeoDataFrame(geometry=[row.geometry], crs=roof_gdf.crs)
        
        try:
            # Clip point cloud
            pipeline = pdal.Pipeline(json.dumps({
                "pipeline": [
                    {
                        "type": "readers.las",
                        "filename": las_path
                    },
                    {
                        "type": "filters.crop",
                        "polygon": polygon_gdf.geometry[0].wkt
                    }
                ]
            }))
            
            pipeline.execute()
            points = pipeline.arrays[0]
            
            if len(points) > 0:
                results[idx] = points
                logging.info(f"Found {len(points)} points")
            else:
                logging.info(f"No points found for roof {idx}")

        except Exception as e:
            logging.error(f"Error processing roof {idx}: {str(e)}")
    
    return results

def compute_plane_inclination(plane_model: np.ndarray) -> tuple[float, float]:
    """
    Compute inclination and aspect angles of a plane from its normal vector.
    
    Args:
        plane_model: numpy array containing plane equation coefficients [a,b,c,d]
        where ax + by + cz + d = 0
        
    Returns:
        tuple: (inclination_degrees, aspect_degrees)
            - inclination_degrees: slope angle in degrees (0-90°)
            - aspect_degrees: aspect angle in degrees (0-360° clockwise from North)
    """
    try:
        logging.info(f"Computing plane orientation for model: {plane_model}")
        
        a, b, c, d = plane_model  # Extract normal components
        norm = np.sqrt(a**2 + b**2 + c**2)
        a, b, c = a/norm, b/norm, c/norm

        # Slope (radians)
        slope = np.arccos(c)
        
        # Aspect (radians) - downslope direction
        aspect = np.arctan2(-b, -a)
        
        # Convert to degrees
        inclination_degrees = np.degrees(slope)
        aspect_degrees = np.degrees(aspect) % 360

        logging.info(f"Computed inclination: {inclination_degrees:.2f}°, aspect: {aspect_degrees:.2f}°")
        return inclination_degrees, aspect_degrees
        
    except Exception as e:
        logging.error(f"Error computing plane inclination: {str(e)}")
        return 0.0, 0.0  # Return default values on error


def segment_planes_with_ids(
    pcd: np.array,
    distance_threshold: np.float32 = 0.15,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_points: int = 1000
) -> tuple[list, list, np.ndarray, int]:
    """
    Segment a point cloud into multiple planes using RANSAC algorithm.
    
    This function iteratively segments planes from a point cloud using RANSAC plane fitting.
    For each plane found, it computes the inclination and aspect angles, and assigns 
    unique IDs to points belonging to each plane.

    Args:
        pcd (np.array): Input point cloud in Open3D format
        distance_threshold (np.float32, optional): Maximum distance a point can be from the plane model. Defaults to 0.15.
        ransac_n (int, optional): Number of initial points to estimate a plane model. Defaults to 3.
        num_iterations (int, optional): Number of RANSAC iterations. Defaults to 1000.
        min_points (int, optional): Minimum number of points required for a valid plane. Defaults to 1000.

    Returns:
        tuple: Contains:
            - list: Inclination angles for each plane in degrees
            - list: Aspect angles for each plane in degrees 
            - np.ndarray: Array of plane IDs for each point (-1 for unassigned points)
            - int: Total number of planes found
    """
    try:
        logging.info(f"Starting plane segmentation with {len(pcd.points)} points")
        
        n_points = len(pcd.points)
        plane_ids = np.full(n_points, -1, dtype=int)
        remaining_pcd = copy.deepcopy(pcd)
        remaining_indices = np.arange(n_points)

        plane_index = 0
        inclinations = []
        aspects = []
        
        while True:
            try:
                if len(remaining_pcd.points) < min_points:
                    logging.info(f"Stopping: remaining points ({len(remaining_pcd.points)}) below threshold")
                    break

                plane_model, inliers = remaining_pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations
                )
                
                if len(inliers) < min_points:
                    logging.info("Stopping: insufficient inliers for new plane")
                    break

                inclination, aspect = compute_plane_inclination(plane_model)
                inclinations.append(inclination)
                aspects.append(aspect)

                plane_ids[remaining_indices[inliers]] = plane_index
                logging.info(f"Found plane {plane_index}: {len(inliers)} points, "
                            f"inclination={inclination:.2f}°, aspect={aspect:.2f}°")

                outliers = np.setdiff1d(np.arange(len(remaining_pcd.points)), inliers)
                remaining_indices = remaining_indices[outliers]
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

                plane_index += 1

            except Exception as e:
                logging.error(f"Error processing plane {plane_index}: {str(e)}")
                break

        logging.info(f"Plane segmentation complete: found {plane_index} planes")
        return inclinations, aspects, plane_ids, plane_index

    except Exception as e:
        logging.error(f"Fatal error in plane segmentation: {str(e)}")
        return [], [], np.full(n_points, -1), 0


def cluster_plane_points(filtered_arr: np.ndarray, eps: float = 0.5, min_samples: int = 100) -> np.ndarray:
    """
    Cluster points within each plane using DBSCAN algorithm.
    
    Args:
        filtered_arr: np.ndarray of shape (N, 4) containing x,y,z coordinates and plane IDs
        eps: float, maximum distance between points in same cluster
        min_samples: int, minimum number of points to form a cluster
        
    Returns:
        np.ndarray: Array of shape (N, 5) containing original points, plane IDs and cluster labels
    """
    try:
        logging.info(f"Starting clustering with eps={eps}, min_samples={min_samples}")
        results = []
        
        # Process each plane
        unique_planes = np.unique(filtered_arr[:, 3])
        logging.info(f"Processing {len(unique_planes)} unique planes")
        
        for plane_id in unique_planes:
            try:
                # Get points for this plane
                plane_mask = filtered_arr[:, 3] == plane_id
                plane_points = filtered_arr[plane_mask]
                logging.info(f"Processing plane {plane_id} with {len(plane_points)} points")
                
                # Run DBSCAN
                clustering = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=-1
                ).fit(plane_points[:, :3])  # Use x,y,z coordinates
                
                # Add cluster labels
                cluster_labels = clustering.labels_
                valid_clusters = cluster_labels != -1
                n_clusters = len(np.unique(cluster_labels[valid_clusters]))
                
                # Add results with plane ID and cluster label
                plane_results = np.column_stack([
                    plane_points,  # Original points with plane ID
                    cluster_labels  # New cluster labels
                ])
                
                results.append(plane_results)
                logging.info(f"Plane {plane_id}: found {n_clusters} clusters")
                
            except Exception as e:
                logging.error(f"Error processing plane {plane_id}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No valid clusters found in any plane")
            
        final_array = np.vstack(results)
        logging.info(f"Clustering complete.")
        return final_array
        
    except Exception as e:
        logging.error(f"Fatal error in clustering: {str(e)}")
        return np.array([])  # Return empty array on error

def denormalize_points(normalized_points: np.ndarray, mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """
    Denormalize points back to original scale
    
    Args:
        normalized_points: Nx2 array of normalized points
        mins: array of minimum values for each dimension 
        ranges: array of ranges (max-min) for each dimension
        
    Returns:
        np.ndarray: Nx2 array of points in original scale
    """
    denormalized_points = (normalized_points * ranges) + mins
    return denormalized_points

def process_alpha_shape(
    alpha_shape: Polygon | MultiPolygon | None, 
    mins: np.ndarray, 
    ranges: np.ndarray
) -> list[Polygon]:
    """
    Process alpha shape result and handle MultiPolygon cases
    
    Args:
        alpha_shape: Shapely polygon or multipolygon from alphashape
        mins: array of minimum values for each dimension
        ranges: array of ranges (max-min) for each dimension
        
    Returns:
        list[Polygon]: List of denormalized Shapely Polygons
    """
    logging.info("Processing alpha shape result")
    polygons = []
    
    if alpha_shape is None:
        logging.info("No alpha shape found")
        return polygons
        
    # Handle MultiPolygon case
    if alpha_shape.geom_type == 'MultiPolygon':
        logging.info(f"Processing MultiPolygon with {len(alpha_shape.geoms)} parts")
        for poly in alpha_shape.geoms:
            denormalized_coords = denormalize_points(
                np.array(poly.exterior.coords),
                mins,
                ranges
            )
            polygons.append(Polygon(denormalized_coords))
    # Handle single Polygon case  
    elif alpha_shape.geom_type == 'Polygon':
        logging.info("Processing single Polygon")
        denormalized_coords = denormalize_points(
            np.array(alpha_shape.exterior.coords),
            mins,
            ranges
        )
        polygons.append(Polygon(denormalized_coords))
        
    logging.info(f"Processed {len(polygons)} polygons")
    return polygons

def filter_polygons(points_2d: np.ndarray, polygon: Polygon, min_area: float = 1.0, min_points: int = 50) -> bool:
    """
    Filter polygons based on area and number of contained points.
    
    Args:
        points_2d: Nx2 numpy array of original 2D points
        polygon: Shapely polygon to check
        min_area: Minimum area in square meters
        min_points: Minimum number of points that should be inside polygon
        
    Returns:
        bool: True if polygon passes filters (has sufficient area and contains enough points), 
              False otherwise
    """
    logging.info(f"Filtering polygon with area={polygon.area:.2f}, min_area={min_area}, min_points={min_points}")
    
    # Check area
    if polygon.area < min_area:
        logging.info(f"Polygon rejected: area {polygon.area:.2f} < minimum {min_area}")
        return False
        
    # Count points inside polygon
    points_inside = sum(1 for pt in points_2d if polygon.contains(Point(pt)))
    
    if points_inside < min_points:
        logging.info(f"Polygon rejected: contains {points_inside} points < minimum {min_points}")
        return False
    
    logging.info(f"Polygon accepted: area={polygon.area:.2f}, points_inside={points_inside}")    
    return True


def get_roofs(filtered_roof_points,
              segment_threshold=0.2,
              segment_ransac_n=5,
              segment_num_iterations=10000,
              segment_min_points=2000,
              filter_min_roof_points=2000,
              cluster_eps=0.5,
              cluster_min_samples=50,
              filter_polygon_min_area=1.0,
              filter_polygon_min_points=500,
              epsg="EPSG:25832"
              ):
    try:
        all_gpd = []
        processed = []
        errors = []

        for roof in list(filtered_roof_points.keys()):
            try:
                logging.info(f"Processing roof {roof}")
                roof_0 = filtered_roof_points[roof]  
                xyz_coords = np.vstack((roof_0['X'], roof_0['Y'], roof_0['Z'])).transpose()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_coords)

                logging.info('Segmenting planes')
                inclinations, aspects, plane_ids, num_planes = segment_planes_with_ids(
                    pcd,
                    distance_threshold=segment_threshold,
                    ransac_n=segment_ransac_n,
                    num_iterations=segment_num_iterations,
                    min_points=segment_min_points
                )

                logging.info(f'Found {num_planes} planes')
                logging.info('Filtering and clustering plane points')
                pointsss = np.hstack((np.asarray(pcd.points), plane_ids.reshape(-1, 1)))
                filtered_arr = np.delete(pointsss, np.where(pointsss[:, -1] == -1), axis=0)
                
                try:
                    for id in np.unique(filtered_arr[:, -1]):
                        if np.shape(np.where(filtered_arr[:, -1] == id))[1] < filter_min_roof_points:
                            filtered_arr = np.delete(filtered_arr, np.where(filtered_arr[:, -1] == id), axis=0)
                except Exception as e:
                    logging.error(f"Error filtering points for roof {roof}: {str(e)}")
                    continue

                logging.info(f'Remaining points after filtering: {filtered_arr.shape[0]}')
                
                try:
                    clustered_arr = cluster_plane_points(
                        filtered_arr,
                        eps=cluster_eps,
                        min_samples=cluster_min_samples
                    )
                except Exception as e:
                    logging.error(f"Error clustering points for roof {roof}: {str(e)}")
                    continue

                logging.info(f'Clustering complete. Total clustered points: {clustered_arr.shape[0]}')
                logging.info('Generating roof polygons from clusters')
                
                all_shpes = []
                cluster_inclinations = []
                cluster_aspects = []
                
                try:
                    for j in np.unique(clustered_arr[:, 3]):
                        plane_mask = clustered_arr[:, 3] == j
                        plane_points = clustered_arr[plane_mask]
                        unique_clusters = np.unique(plane_points[:, 4])

                        for i in unique_clusters:
                            if i == -1:
                                continue
                            try:
                                cluster_mask = plane_points[:, 4] == i
                                pts2d = plane_points[cluster_mask][:, 0:2]
                                normalized_pts2d, mins, ranges = normalize_points(pts2d)
                                alpha_shape = alphashape.alphashape(normalized_pts2d, 20.0)
                                polygons = process_alpha_shape(alpha_shape, mins, ranges)
                                filtered_polygons = [
                                    poly for poly in polygons if filter_polygons(pts2d, poly, min_area=filter_polygon_min_area, min_points=filter_polygon_min_points)
                                ]
                                for filtered_polygon in filtered_polygons:
                                    if filtered_polygon.is_valid:
                                        all_shpes.append(filtered_polygon)
                                        cluster_inclinations.append(inclinations[int(j)])
                                        cluster_aspects.append(aspects[int(j)])
                            except Exception as e:
                                logging.error(f"Error processing cluster {i} in plane {j} for roof {roof}: {str(e)}")
                                continue
                except Exception as e:
                    logging.error(f"Error processing planes for roof {roof}: {str(e)}")
                    continue

                try:
                    rp_gpd = gpd.GeoDataFrame(
                        {
                            'geometry': all_shpes,
                            'roof_id': range(len(all_shpes)),
                            'inclination': cluster_inclinations,
                            'aspect': cluster_aspects
                        },
                        crs=epsg
                    )
                    all_gpd.append(rp_gpd)
                    logging.info(f"Processed roof {roof} with {len(rp_gpd)} segments.")
                    processed.append(roof)
                except Exception as e:
                    logging.error(f"Error creating GeoDataFrame for roof {roof}: {str(e)}")
                    continue

            except Exception as e:
                logging.error(f"Error processing roof {roof}: {str(e)}")
                errors.append(roof)
                continue

        logging.info(f"Processing complete. Processed {len(processed)} roofs with {len(errors)} errors.")
        return all_gpd

    except Exception as e:
        logging.error(f"Fatal error in get_roofs: {str(e)}")
        return []

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

def main():
    logging.info('Starting script.')
    try:
        # Input file handling
        try:
            args = parse_args()
            config = read_config(args.config_file)
            logging.info(f"Configuration: {config}")

        except Exception as e:
            logging.error(f"Error loading config")
            raise

        # Process roof polygons
        try:
            roof_image = gpd.read_file(config['roof_shapefile'])
            logging.info(f"Successfully loaded input files: {roof_image}")
            roof_points = process_roof_polygons(
                las_path=config['las_filename'],
                roof_gdf=roof_image
            )
            logging.info(f"Successfully processed {len(roof_points)} roofs")
        except Exception as e:
            logging.error(f"Error in process_roof_polygons: {str(e)}")
            raise

        # Filter roof points
        try:
            filtered_roof_points = {
                idx: points for idx, points in roof_points.items() 
                if len(points) >= config['min_points_per_roof']
            }
            logging.info("\nFiltering summary:")
            logging.info(f"Original roofs: {len(roof_points)}")
            logging.info(f"Filtered roofs: {len(filtered_roof_points)}")
            logging.info(f"Removed {len(roof_points) - len(filtered_roof_points)} roofs with < {config['min_points_per_roof']} points")
        except Exception as e:
            logging.error(f"Error filtering roof points: {str(e)}")
            raise

        # Process roofs
        try:
            all_gpd = get_roofs(filtered_roof_points,
                segment_threshold=config['ext_segment_threshold'],
                segment_ransac_n=config['ext_segment_ransac_n'],
                segment_num_iterations=config['ext_segment_num_iterations'],
                segment_min_points=config['ext_segment_min_points'],
                filter_min_roof_points=config['ext_filter_min_roof_points'],
                cluster_eps=config['ext_cluster_eps'],
                cluster_min_samples=config['ext_cluster_min_samples'],
                filter_polygon_min_area=config['ext_filter_polygon_min_area'],
                filter_polygon_min_points=config['ext_filter_polygon_min_points']
            )
            if not all_gpd:
                raise ValueError("No valid roof segments found")
        except Exception as e:
            logging.error(f"Error in get_roofs processing: {str(e)}")
            raise

        # Process results and save
        try:
            for idx, gdf in enumerate(all_gpd):
                gdf['roof_id'] = idx

            single_gdf = gpd.GeoDataFrame(pd.concat(all_gpd, ignore_index=True), crs=all_gpd[0].crs)
            logging.info(f"Total roof segments: {len(single_gdf)}")
            
            
            single_gdf.to_file(config['output_path'])
            logging.info(f"Saved all roof segments to {config['output_path']}")
            logging.info("Roof extraction process completed successfully.")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

    except Exception as e:
        logging.error(f"Fatal error in main process: {str(e)}")
        exit(1)
if __name__ == '__main__':
    main()
