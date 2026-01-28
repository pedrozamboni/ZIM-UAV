import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
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

def calculate_distances(original_pcd, registered_pcd):
    """
    Calculate the distances between each point in the registered point cloud and the nearest point in the original point cloud.

    Parameters:
        original_pcd (open3d.geometry.PointCloud): The original point cloud.
        registered_pcd (open3d.geometry.PointCloud): The registered point cloud.

    Returns:
        np.array: Distances between points in registered_pcd and nearest points in original_pcd.
    """
    logging.info("Starting distance calculation...")
    points_original = np.asarray(original_pcd.points)
    points_registered = np.asarray(registered_pcd.points)

    logging.info(f"Original point cloud: {len(points_original)} points")
    logging.info(f"Registered point cloud: {len(points_registered)} points")

    tree_original = KDTree(points_original)
    distances, _ = tree_original.query(points_registered, k=1)

    logging.info(f"Distance calculation completed. Mean distance: {np.mean(distances):.4f}, Max distance: {np.max(distances):.4f}")

    return distances

def compute_voxel_statistics(points, distances, voxel_size):
    """
    Compute the mean and std. deviation of distances within each voxel.

    Parameters:
        points (np.array): Points of the point cloud.
        distances (np.array): Distances corresponding to each point.
        voxel_size (float): The voxel size for voxelization.

    Returns:
        dict: Dictionary of voxel coordinates as keys and (mean, std_dev) as values.
    """
    logging.info("Starting voxel statistics computation...")
    
    # Calculate voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    logging.info(f"Voxel grid created with size: {voxel_size}")

    # Organize distances by voxel
    voxel_distances = {}
    for i, voxel in enumerate(voxel_indices):
        voxel_coord = tuple(voxel)
        if voxel_coord not in voxel_distances:
            voxel_distances[voxel_coord] = []
        voxel_distances[voxel_coord].append(distances[i])

    # Compute mean and std. deviation for each voxel
    voxel_stats = {}
    for voxel, dists in voxel_distances.items():
        voxel_stats[voxel] = (np.mean(dists), np.std(dists))

    logging.info(f"Voxel statistics computed for {len(voxel_stats)} voxels")
    return voxel_stats

def create_colored_point_cloud(original_pcd, voxel_stats, voxel_size, stat_type='mean'):
    """
    Create a point cloud where each point's color represents the mean or std. deviation of distances in its voxel.

    Parameters:
        original_pcd (open3d.geometry.PointCloud): The original point cloud.
        voxel_stats (dict): Dictionary of voxel coordinates as keys and (mean, std_dev) as values.
        voxel_size (float): The size of the voxel grid.
        stat_type (str): 'mean' or 'std' to visualize the mean or standard deviation of distances.

    Returns:
        open3d.geometry.PointCloud: Point cloud with voxel statistics.
    """
    logging.info(f"Creating colored point cloud with stat_type: {stat_type}")
    
    points = np.asarray(original_pcd.points)
    voxel_indices = np.floor(points / voxel_size).astype(int)

    colors = []
    max_mean = max([v[0] for v in voxel_stats.values()])
    max_std = max([v[1] for v in voxel_stats.values()])

    logging.info(f"Max mean: {max_mean}, Max std: {max_std}")

    for voxel in voxel_indices:
        voxel_coord = tuple(voxel)
        if voxel_coord in voxel_stats:
            mean_dist, std_dev = voxel_stats[voxel_coord]
            if stat_type == 'mean':
                color_value = mean_dist / max_mean if max_mean != 0 else 0
                color = [color_value, 0, 0]
            else:
                color_value = std_dev / max_std if max_std != 0 else 0
                color = [0, color_value, 0]
        else:
            color = [0, 0, 0]
        colors.append(color)

    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    logging.info(f"Colored point cloud created with {len(points)} points")
    return colored_pcd

def add_scalar_field(original_pcd, voxel_stats, voxel_size, threshold):
    """
    Add a scalar field to the point cloud based on the standard deviation threshold.

    Parameters:
        original_pcd (open3d.geometry.PointCloud): The original point cloud.
        voxel_stats (dict): Dictionary of voxel coordinates as keys and (mean, std_dev) as values.
        voxel_size (float): The size of the voxel grid.
        threshold (float): The standard deviation threshold to separate stable and changed points.

    Returns:
        np.array: Scalar field indicating stable (0) and changed (1) points.
    """
    logging.info(f"Adding scalar field with threshold: {threshold}")
    
    points = np.asarray(original_pcd.points)
    voxel_indices = np.floor(points / voxel_size).astype(int)

    scalar_field = []
    changed_count = 0
    stable_count = 0
    
    for voxel in voxel_indices:
        voxel_coord = tuple(voxel)
        if voxel_coord in voxel_stats:
            _, std_dev = voxel_stats[voxel_coord]
            scalar_value = 1 if std_dev > threshold else 0
        else:
            scalar_value = 0
        scalar_field.append(scalar_value)
        
        if scalar_value == 1:
            changed_count += 1
        else:
            stable_count += 1

    logging.info(f"Scalar field created: {stable_count} stable points, {changed_count} changed points")
    return np.array(scalar_field)

def save_point_clouds(original_pcd, scalar_field, output_path_stable, output_path_changed):
    """
    Save the stable and changed point clouds based on the scalar field.

    Parameters:
        original_pcd (open3d.geometry.PointCloud): The original point cloud.
        scalar_field (np.array): Scalar field indicating stable (0) and changed (1) points.
        output_path_stable (str): Path to save the stable point cloud.
        output_path_changed (str): Path to save the changed point cloud.
    """
    logging.info("Starting to save point clouds...")
    
    points = np.asarray(original_pcd.points)
    colors = np.asarray(original_pcd.colors) if original_pcd.has_colors() else None

    stable_indices = np.where(scalar_field == 0)[0]
    changed_indices = np.where(scalar_field == 1)[0]

    logging.info(f"Stable points: {len(stable_indices)}, Changed points: {len(changed_indices)}")

    stable_pcd = o3d.geometry.PointCloud()
    stable_pcd.points = o3d.utility.Vector3dVector(points[stable_indices])
    if colors is not None:
        stable_pcd.colors = o3d.utility.Vector3dVector(colors[stable_indices])

    changed_pcd = o3d.geometry.PointCloud()
    changed_pcd.points = o3d.utility.Vector3dVector(points[changed_indices])
    if colors is not None:
        changed_pcd.colors = o3d.utility.Vector3dVector(colors[changed_indices])

    o3d.io.write_point_cloud(output_path_stable, stable_pcd)
    o3d.io.write_point_cloud(output_path_changed, changed_pcd)

    logging.info(f"Stable point cloud saved to {output_path_stable}")
    logging.info(f"Changed point cloud saved to {output_path_changed}")

def main():
    logging.info('Starting script.')
    try:
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")
    except Exception as e:
        logging.error(f"Error loading config")
        raise


    original_path = config['original_path']
    registered_path = config['registered_path']
    voxel_size = config['voxel_size']
    threshold = config['threshold']
    # Load the point clouds
    logging.info("Loading point clouds...")
    original_pcd = o3d.io.read_point_cloud(original_path)
    registered_pcd = o3d.io.read_point_cloud(registered_path)

    logging.info(f"Original point cloud loaded from {original_path}")
    logging.info(f"Registered point cloud loaded from {registered_path}")
    logging.info(f"Computing distances between point clouds...")
    # Compute the distance point cloud
    distances = calculate_distances(original_pcd, registered_pcd)
    logging.info(f"Distances computed.")
    logging.info(f"Distances: Mean={np.mean(distances):.4f}, Max={np.max(distances):.4f}")
    logging.info(f"Computing voxel statistics...")
    # Compute voxel statistics
    points_registered = np.asarray(registered_pcd.points)
    voxel_stats = compute_voxel_statistics(points_registered, distances, voxel_size)
    logging.info(f"Voxel statistics computed.")
    logging.info(f"Creating colored point cloud for mean distances...")
    # Add scalar field based on standard deviation threshold
    scalar_field = add_scalar_field(original_pcd, voxel_stats, voxel_size, threshold)
    logging.info(f"Scalar field added to point cloud.")
    logging.info(f"Saving stable and changed point clouds...")
    # Save stable and changed point clouds
    output_path_stable = config['output_path_stable']
    output_path_changed = config['output_path_changed']
    save_point_clouds(original_pcd, scalar_field, output_path_stable, output_path_changed)
    logging.info(f"Point clouds saved to {output_path_stable} and {output_path_changed}")
    logging.info("Script completed successfully.")
# Example usage
if __name__ == "__main__":
    main()