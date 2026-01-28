import open3d as o3d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def get_2d_bounding_box(point_cloud):
    """Compute the 2D bounding box of a point cloud (x and y coordinates only)."""
    points = np.asarray(point_cloud.points)
    min_bound_2d = points[:, :2].min(axis=0)  # Use only x and y
    max_bound_2d = points[:, :2].max(axis=0)
    return min_bound_2d, max_bound_2d

def get_edge_points_in_bbox(points, min_bound, max_bound, edge_tolerance=0.5):
    """Extract only edge points from the bounding box (not all points inside)."""
    x_min, y_min = min_bound
    x_max, y_max = max_bound

    # Find points near the bounding box edges
    mask_x_edge = (np.abs(points[:, 0] - x_min) < edge_tolerance) | (np.abs(points[:, 0] - x_max) < edge_tolerance)
    mask_y_edge = (np.abs(points[:, 1] - y_min) < edge_tolerance) | (np.abs(points[:, 1] - y_max) < edge_tolerance)

    edge_mask = mask_x_edge | mask_y_edge  # Points near any bounding edge
    return points[edge_mask]

def compute_edge_heights(uav_pcd, als_pcd, edge_tolerance=0.5):
    """Compute height differences between UAV and ALS only at bounding box edges."""
    # Get bounding box from UAV point cloud
    min_bound_2d, max_bound_2d = get_2d_bounding_box(uav_pcd)

    # Convert to NumPy arrays
    uav_points = np.asarray(uav_pcd.points)
    als_points = np.asarray(als_pcd.points)

    # Get only edge points (not entire bbox)
    uav_edge_points = get_edge_points_in_bbox(uav_points, min_bound_2d, max_bound_2d, edge_tolerance)

    if len(uav_edge_points) == 0:
        print("No edge points found in bounding box. Check edge_tolerance.")
        return None, None

    # Build KDTree for ALS nearest neighbor search
    als_tree = cKDTree(als_points)

    # Find nearest ALS neighbor for each UAV edge point
    distances, indices = als_tree.query(uav_edge_points)

    # Compute height differences (Z-coordinates)
    uav_heights = uav_edge_points[:, 2]
    als_heights = als_points[indices, 2]
    edge_heights = np.abs(uav_heights - als_heights)

    return uav_edge_points, edge_heights

def plot_violin_edge_heights(edge_heights):
    """Plot a violin plot of edge height differences."""
    plt.figure(figsize=(8, 5))
    sns.violinplot(y=edge_heights, color="skyblue")
    plt.ylabel("Edge Height Difference (m)")
    plt.title("Violin Plot of Edge Heights at Bounding Box Edges")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def visualize_high_edge_heights(uav_edge_points, edge_heights, H_max=1.0, output_ply="high_edge_heights_bbox.ply"):
    """Visualize UAV bounding box edge points where height differences exceed H_max."""
    high_edge_mask = edge_heights > H_max
    high_edge_points = uav_edge_points[high_edge_mask]

    if len(high_edge_points) == 0:
        logging.warning("No high-edge points exceed H_max.")
        return

    # Create Open3D point cloud
    high_edge_pcd = o3d.geometry.PointCloud()
    high_edge_pcd.points = o3d.utility.Vector3dVector(high_edge_points)

    # Assign RED color for high edges
    colors = np.zeros((high_edge_points.shape[0], 3))  # Initialize as black
    colors[:, 0] = 1.0  # Set red channel to 1 (Red)
    high_edge_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save and visualize
    o3d.io.write_point_cloud(output_ply, high_edge_pcd)
    logging.info(f"High edge height points saved to {output_ply}")
    logging.info(f"Found {len(high_edge_points)} points exceeding H_max={H_max}m")
    o3d.visualization.draw_geometries([high_edge_pcd])


def main():
    try:
        logging.info("Starting main execution")
        args = parse_args()
        logging.info("Parsing configuration file")
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded: {config}")
    except Exception as e:
        logging.error(f"Error loading config")
        raise e
    
    # Load point clouds
    uav_pcd = o3d.io.read_point_cloud(config["uav_point_cloud"])
    als_pcd = o3d.io.read_point_cloud(config["als_point_cloud"])

    # Compute edge heights within the bounding box edges
    uav_edge_points, edge_heights = compute_edge_heights(uav_pcd, als_pcd,
                                                          edge_tolerance=config['edge_tolerance'])

    if uav_edge_points is not None and edge_heights is not None:
        # Plot violin plot
        plot_violin_edge_heights(edge_heights)

        # Visualize high-edge points in Open3D
        visualize_high_edge_heights(uav_edge_points, edge_heights, H_max=0.4)


