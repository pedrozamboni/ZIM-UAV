import numpy as np
from scipy.spatial import ConvexHull
import laspy
import logging
import argparse
import yaml
import os


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('downsample.log'),
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
def load_point_cloud(las_path):
    """Load LAS/LAZ file and return coordinates array"""
    las = laspy.read(las_path)
    return np.vstack((las.x, las.y, las.z,las.red,las.green,las.blue)).transpose(), las.header

def calculate_point_density(points):
    """
    Calculate point density (points per square meter)
    Args:
        points: Nx3 numpy array of XYZ coordinates
    Returns:
        density: points per square meter
    """
    # Project points to 2D (XY plane) for area calculation
    points_2d = points[:, :2]
    
    # Calculate area using ConvexHull
    hull = ConvexHull(points_2d)
    area = hull.area
    
    # Calculate density
    num_points = points.shape[0]
    density = num_points / area
    
    return density

def subsample_to_density(points, target_density):
    """
    Randomly subsample points to match target density
    Args:
        points: Nx3 numpy array of XYZ coordinates
        target_density: desired points per square meter
    Returns:
        subsampled_points: randomly subsampled points matching target density
    """
    current_density = calculate_point_density(points)
    
    if current_density <= target_density:
        return points
    
    # Calculate how many points we need
    points_2d = points[:, :2]
    hull = ConvexHull(points_2d)
    area = hull.area
    target_points = int(target_density * area)
    
    # Randomly select points
    indices = np.random.choice(
        points.shape[0], 
        size=target_points, 
        replace=False
    )
    
    return indices

# Example usage:# Example usage
def match_point_cloud_densities(pcd1, pcd2):
    """
    Match the density of two point clouds by subsampling the denser one
    Args:
        pcd1: First point cloud (Nx3 numpy array)
        pcd2: Second point cloud (Mx3 numpy array)
    Returns:
        pcd1_matched, pcd2_matched: Point clouds with matched densities
    """
    density1 = calculate_point_density(pcd1)
    density2 = calculate_point_density(pcd2)
    
    print(f"Density of point cloud 1: {density1:.2f} points/m²")
    print(f"Density of point cloud 2: {density2:.2f} points/m²")
    
    # Match to the lower density
    target_density = min(density1, density2)
    
    pcd1_indices = subsample_to_density(pcd1, target_density)
    pcd2_indices = subsample_to_density(pcd2, target_density)
    
    return pcd1_indices, pcd2_indices


def save_point_cloud(points, header, output_path):
   
    with laspy.open(output_path, mode='w', header=header) as writer:
                            point_record = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
                            
                            point_record.x = points[:,0]
                            point_record.y = points[:,1]
                            point_record.z = points[:,2]
                            point_record.red = points[:,3]
                            point_record.green = points[:,4]
                            point_record.blue = points[:,5]
                            writer.write_points(point_record)


def main():
    args = parse_args()
    config = read_config(args.config_file)
    logging.info(f"Configuration: {config}")
    # In your main function:
    pcd1, header1 = load_point_cloud(config['point_cloud_1_path'])
    pcd2, header2 = load_point_cloud(config['point_cloud_2_path'])
    pcd1_indices, pcd2_indices = match_point_cloud_densities(
        pcd1[:, :3],  # Use only XYZ coordinates
        pcd2[:, :3]
    )

    logging.info(f"Original points - PCD1: {pcd1.shape[0]}, PCD2: {pcd2.shape[0]}")
    logging.info(f"After matching - PCD1: {pcd1_indices.shape[0]}, PCD2: {pcd2_indices.shape[0]}")
    # Save matched point clouds
    logging.info("Saving matched point clouds...")
    # Check and save only point clouds that changed
    if pcd1_indices.shape[0] < pcd1.shape[0]:
        input_basename = os.path.splitext(os.path.basename(config['point_cloud_1_path']))[0]
        output_path1 = os.path.join(config['output_output'], f"{input_basename}_downsampled.las")
        save_point_cloud(pcd1[pcd1_indices], header1, output_path1)
        logging.info(f"PCD1 downsampled and saved to {output_path1}")
    else:
        logging.info("PCD1 unchanged - no downsampling needed")
    
    if pcd2_indices.shape[0] < pcd2.shape[0]:
        input_basename = os.path.splitext(os.path.basename(config['point_cloud_2_path']))[0]
        output_path2 = os.path.join(config['output_output'], f"{input_basename}_downsampled.las")
        save_point_cloud(pcd2[pcd2_indices], header2, output_path2)
        logging.info(f"PCD2 downsampled and saved to {output_path2}")
    else:
        logging.info("PCD2 unchanged - no downsampling needed")

if __name__ == "__main__":
    main()