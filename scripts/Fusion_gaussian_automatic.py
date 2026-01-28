import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os
import logging
import argparse
import yaml
import sys

def get_log_filename(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return os.path.join('log', f'{base}.log')


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


def get_2d_bounding_box(point_cloud):
    """Compute the 2D bounding box of a point cloud, using only x and y coordinates."""
    points = np.asarray(point_cloud.points)
    min_bound_2d = points[:, :2].min(axis=0)  # Use only x and y coordinates
    max_bound_2d = points[:, :2].max(axis=0)  # Use only x and y coordinates
    return min_bound_2d, max_bound_2d


def compute_edge_heights(edge_points, uav_pcd, als_pcd):
    """Compute edge heights as the absolute elevation difference between UAV and ALS at the edge."""
    uav_points = np.asarray(uav_pcd.points)
    als_points = np.asarray(als_pcd.points)

    # Create KD-trees for fast nearest neighbor search
    uav_tree = cKDTree(uav_points[:, :2])  # Use only XY for searching
    als_tree = cKDTree(als_points[:, :2])

    # Find nearest UAV and ALS points for the given edge points
    _, uav_indices = uav_tree.query(edge_points[:, :2])  # Get nearest UAV indices
    _, als_indices = als_tree.query(edge_points[:, :2])  # Get nearest ALS indices

    # Extract the Z-coordinates (heights)
    uav_heights = uav_points[uav_indices, 2]  # UAV elevation
    als_heights = als_points[als_indices, 2]  # ALS elevation

    # Compute absolute height differences
    edge_heights = np.abs(uav_heights - als_heights)

    return edge_heights

def adaptive_blending_distance(height_diffs, d_min=5, d_max=15, h_max=2):
    """Maps edge height differences to blending distances."""
    return d_min + (d_max - d_min) * np.clip(height_diffs / h_max, 0, 1)


def gaussian_weight(distances, sigma):
    """Gaussian weight function."""
    return np.exp(- (distances ** 2) / (2 * sigma ** 2))


def weighted_smoothing(uav_pcd, als_pcd, default_distance=10.0, k_neighbors=5):
    """Apply weighted smoothing to the overlapping region of two point clouds based on edge-adaptive blending."""
    
    logging.info("Starting weighted smoothing process.")
    
    # Convert point clouds to numpy arrays
    uav_points = np.asarray(uav_pcd.points)
    als_points = np.asarray(als_pcd.points)
    logging.info(f"UAV points: {len(uav_points)}, ALS points: {len(als_points)}")

    # Compute 2D bounding box and distances to edge
    min_bound_2d, max_bound_2d = get_2d_bounding_box(uav_pcd)
    x_min, y_min = min_bound_2d
    x_max, y_max = max_bound_2d
    logging.info(f"2D bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")

    # Distance to nearest edge
    min_distance_to_edge = np.minimum.reduce([
        np.abs(uav_points[:, 0] - x_min),
        np.abs(uav_points[:, 0] - x_max),
        np.abs(uav_points[:, 1] - y_min),
        np.abs(uav_points[:, 1] - y_max)
    ])

    # Filter points within max distance
    mask = min_distance_to_edge <= default_distance
    uav_points_in_overlap = uav_points[mask]
    distances_in_overlap = min_distance_to_edge[mask]
    logging.info(f"Points in overlap region: {len(uav_points_in_overlap)}")

    # Compute edge heights
    edge_heights = compute_edge_heights(uav_points_in_overlap, uav_pcd, als_pcd)
    logging.info(f"Mean edge height difference: {np.mean(edge_heights):.4f}")

    # Determine adaptive blending zone
    blending_distances = adaptive_blending_distance(edge_heights)

    # Nearest neighbor search for ALS points
    als_tree = cKDTree(als_points)
    distances, indices = als_tree.query(uav_points_in_overlap, k=k_neighbors)
    logging.info(f"KD-tree query completed with k={k_neighbors} neighbors")

    # Interpolate ALS points
    als_points_interpolated = np.array([
        np.dot(1 / (neighbor_distances + 1e-8) / np.sum(1 / (neighbor_distances + 1e-8)), als_points[indices[i]])
        for i, neighbor_distances in enumerate(distances)
    ])

    # Apply Gaussian weighting with adaptive blending distances
    w2 = gaussian_weight(distances_in_overlap, sigma=blending_distances / 3)
    w1 = 1 - w2
    logging.info(f"Weights computed: mean w1={np.mean(w1):.4f}, mean w2={np.mean(w2):.4f}")

    # Weighted averaging
    smoothed_points = w1[:, np.newaxis] * uav_points_in_overlap + w2[:, np.newaxis] * als_points_interpolated

    # Combine with non-overlapping UAV points
    combined_points = np.vstack((smoothed_points, uav_points[~mask], als_points))
    logging.info(f"Final combined point cloud size: {len(combined_points)}")

    # Create final smoothed point cloud
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    logging.info("Weighted smoothing process completed successfully.")

    return final_pcd

def main():
    logging.info("Starting Gaussian adaptive smoothing process.")
    try:
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")

    except Exception as e:
        logging.error(f"Error loading config")
        raise e

    uav_pcd = o3d.io.read_point_cloud(config["uav_point_cloud"])
    als_pcd = o3d.io.read_point_cloud(config["als_point_cloud"])

    smoothed_pcd = weighted_smoothing(uav_pcd, als_pcd, default_distance=config["default_distance"], k_neighbors=5)

    o3d.io.write_point_cloud(config["output_point_cloud"], smoothed_pcd)
# Example usage
    # uav_pcd = o3d.io.read_point_cloud(
    #     r'C:\Users\Hanne Hendrickx\Documents\08_GSS_Projects\ZIM\DATA\WP5-1-2\5.3 edge fusion\UAV_ground_cut_transformed.ply')
    # als_pcd = o3d.io.read_point_cloud(
    #     r'C:\Users\Hanne Hendrickx\Documents\08_GSS_Projects\ZIM\DATA\WP5-1-2\5.3 edge fusion\ALS_ground_cut.ply')

    # smoothed_pcd = weighted_smoothing(uav_pcd, als_pcd, default_distance=10.0, k_neighbors=5)

    # o3d.io.write_point_cloud("smoothed_gaussian_adaptive2.ply", smoothed_pcd)
    # o3d.visualization.draw_geometries([smoothed_pcd])

if __name__ == "__main__":
    main()