import numpy as np
import open3d as o3d
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

def downsample_pointcloud(input_path, output_path,filename,voxel_size=0.05 ):
    logging.info(f"Starting downsampling with voxel size: {voxel_size}")
    # Create an empty Open3D point cloud object for the final result
    final_pcd = o3d.geometry.PointCloud()
    
    # Keep track of total points and header for output
    total_original_points = 0
    las_header = None
    first_chunk = True
    all_points = []
    all_dims = {}
    
    # Read the point cloud from a .laz file using chunk iterator
    logging.info(f"Reading input file: {input_path}")
    with laspy.open(input_path) as input_las:
        las_header = input_las.header
        # Get all dimension names except x, y, z
        dim_names = [dim.name for dim in input_las.header.point_format.dimensions if dim.name not in ('X', 'Y', 'Z')]
        logging.info(f"Found dimensions: {dim_names}")
        
        for i, points in enumerate(input_las.chunk_iterator(100000000)):
            logging.info(f"Processing chunk {i}")
            # Convert chunk to numpy array for coordinates
            points_array = np.vstack((points.x, points.y, points.z)).transpose()
            poins_colors = np.vstack((points.red, points.green, points.blue)).transpose()

            chunk_original_size = len(points_array)
            total_original_points += chunk_original_size
            
            # Create temporary point cloud for this chunk
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(points_array)
            temp_pcd.colors = o3d.utility.Vector3dVector(poins_colors)  
            logging.info(f"Iteration {i}: Original chunk shape: {points_array.shape}")
            
            
            # Downsample this chunk
            temp_pcd = temp_pcd.voxel_down_sample(voxel_size=voxel_size)
            final_pcd += temp_pcd
            
            # Clean up memory
            del points_array
            del poins_colors
            del temp_pcd
            points = None
    
    # Get final size and calculate percentage
    final_size = len(final_pcd.points)
    percentage = (final_size / total_original_points) * 100
    logging.info(f"Final point cloud size: {final_size}")
    logging.info(f"Original point cloud size: {total_original_points}")
    logging.info(f"Percentage of points retained: {percentage:.2f}%")
    
    # Create output LAS file
    output_filename = os.path.join(output_path, filename+f"downsampled_voxel_{voxel_size}.las")
    points = np.asarray(final_pcd.points)
    colors = np.asarray(final_pcd.colors)

    logging.info("Creating output LAS file")
    output_las = laspy.create(point_format=las_header.point_format, file_version=las_header.version)
    output_las.header.offsets = las_header.offsets
    output_las.header.scales = las_header.scales
    
    # Write coordinates
    output_las.x = points[:, 0]
    output_las.y = points[:, 1]
    output_las.z = points[:, 2]
    output_las.red = colors[:, 0]
    output_las.green = colors[:, 1]
    output_las.blue = colors[:, 2]
    # Write all other dimensions
    for dim_name, dim_data in all_dims.items():
        setattr(output_las, dim_name.lower(), dim_data)
    
    output_las.write(output_filename)
    logging.info(f"Saved downsampled point cloud to {output_filename}")
    
    return final_pcd

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_file)
    logging.info(f"Configuration: {config}")
    
    # Downsample with specified voxel size
    downsampled = downsample_pointcloud(config['input_file'], config['output_path'], config['filename'], config['voxel_size'])
    logging.info("Downsampling completed.")