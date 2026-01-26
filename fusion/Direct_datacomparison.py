import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import logging
import argparse
import yaml
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


def compute_cloud_to_cloud_distance(source_pcd, target_pcd):
    """Compute signed cloud-to-cloud distance for X, Y, and Z components."""
    logging.info("Computing cloud-to-cloud distances...")
    
    source_points = np.asarray(source_pcd.points)  # UAV or reference cloud
    target_points = np.asarray(target_pcd.points)  # ALS or comparison cloud
    
    logging.debug(f"Source cloud points shape: {source_points.shape}")
    logging.debug(f"Target cloud points shape: {target_points.shape}")

    # Build KDTree for target cloud
    logging.debug("Building KDTree for target cloud...")
    target_tree = cKDTree(target_points)

    # Find nearest neighbor in target cloud for each point in source cloud
    logging.debug("Querying nearest neighbors...")
    _, indices = target_tree.query(source_points)

    # Compute per-axis signed differences (UAV - ALS)
    x_diff = source_points[:, 0] - target_points[indices, 0]
    y_diff = source_points[:, 1] - target_points[indices, 1]
    z_diff = source_points[:, 2] - target_points[indices, 2]
    
    logging.info(f"Cloud-to-cloud distance computed. X: mean={np.mean(x_diff):.4f}, Y: mean={np.mean(y_diff):.4f}, Z: mean={np.mean(z_diff):.4f}")

    return x_diff, y_diff, z_diff

def plot_c2c_histograms(x_diff, y_diff, z_diff, bins=50, out_path='c2c_histograms.png'):
    """Plot histograms of signed Cloud-to-Cloud distance for X, Y, and Z components with Mean & Std Dev."""
    logging.info(f"Plotting cloud-to-cloud distance histograms to {out_path}...")
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    components = [('X', x_diff, 'blue'), ('Y', y_diff, 'green'), ('Z', z_diff, 'red')]

    for i, (axis, data, color) in enumerate(components):
        mean_error = np.mean(data)
        std_dev = np.std(data)
        
        logging.debug(f"{axis} component - Mean: {mean_error:.4f}, Std Dev: {std_dev:.4f}")

        axs[i].hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black')
        axs[i].set_title(f"{axis} Distance")
        axs[i].set_xlabel(f"Signed {axis} Difference (m)")
        axs[i].set_ylabel("Frequency")

        # Plot Mean & Standard Deviation as vertical lines
        axs[i].axvline(mean_error, color='black', linestyle='dashed', linewidth=2, label=f"Mean = {mean_error:.4f}")
        axs[i].axvline(mean_error + std_dev, color='gray', linestyle='dotted', linewidth=2, label=f"Mean + 1σ = {mean_error + std_dev:.4f}")
        axs[i].axvline(mean_error - std_dev, color='gray', linestyle='dotted', linewidth=2, label=f"Mean - 1σ = {mean_error - std_dev:.4f}")

        # Display the values on the plot
        axs[i].text(mean_error, axs[i].get_ylim()[1] * 0.9, f"Mean: {mean_error:.4f} m", color='black', fontsize=10)
        axs[i].text(mean_error + std_dev, axs[i].get_ylim()[1] * 0.8, f"+1σ: {mean_error + std_dev:.4f} m", color='gray', fontsize=10)
        axs[i].text(mean_error - std_dev, axs[i].get_ylim()[1] * 0.7, f"-1σ: {mean_error - std_dev:.4f} m", color='gray', fontsize=10)

        axs[i].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logging.info(f"Histograms saved to {out_path}")
    plt.close()


def main():
    logging.info('Starting script.')
   
    try:
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")

    except Exception as e:
        logging.error(f"Error loading config")
        raise
    logging.info("Loading point clouds...")
    uav_pcd = o3d.io.read_point_cloud(config['uav_pcd_path'])
    logging.info(f"UAV point cloud loaded from {config['uav_pcd_path']}")
    
    als_pcd = o3d.io.read_point_cloud(config['als_pcd_path'])
    logging.info(f"ALS point cloud loaded from {config['als_pcd_path']}")

    # Compute cloud-to-cloud distances (signed)
    x_diff, y_diff, z_diff = compute_cloud_to_cloud_distance(uav_pcd, als_pcd)

    # Plot histograms with Mean & Standard Deviation
    logging.info(f"Generating histogram output to {config['histogram_output_path']}")
    plot_c2c_histograms(x_diff, y_diff, z_diff, out_path=config['histogram_output_path'])
    logging.info("Script completed successfully.")

if __name__ == "__main__":
    main()