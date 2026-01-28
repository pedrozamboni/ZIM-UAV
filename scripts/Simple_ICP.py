import open3d as o3d
import numpy as np
import os
import laspy
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
def evaluate_registration(source, target, transformation):
    """
    Compute RMSE (Root Mean Square Error) between source and target point clouds after alignment.

    Parameters:
        source (o3d.geometry.PointCloud): Source point cloud.
        target (o3d.geometry.PointCloud): Target point cloud.
        transformation (numpy.ndarray): Transformation matrix applied to source to align with target.

    Returns:
        float: Root Mean Square Error (RMSE) between aligned source and target. --> use ME instead?
    """
    source.transform(transformation)
    distances = source.compute_point_cloud_distance(target)
    rmse = np.sqrt(np.mean(np.asarray(distances)))
    return rmse

def main(source_path, target_path, max_correspondence_distances, output_path):
    """
    Main function to register two point clouds using multi-scale ICP.

    Parameters:
        source_path (str): Path to the source point cloud file.
        target_path (str): Path to the target point cloud file.
        max_correspondence_distances (list): List of maximum correspondence distances for multi-scale ICP.
    """
    try:
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")

    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise e

    logging.info("Loading point clouds...")
    logging.info(f"Source path: {config['source_path']}")
    logging.info(f"Target path: {config['target_path']}")

    # Load source and target point clouds
    source_pcd = o3d.io.read_point_cloud(config['source_path'])
    target_pcd = o3d.io.read_point_cloud(config['target_path'])
    input_header = laspy.read(config['source_path']).header

    logging.info("Point clouds loaded successfully.")
    # # Visualize initial alignment (optional)
    # o3d.visualization.draw_geometries([source_pcd, target_pcd])

    msa = o3d.utility.DoubleVector([0.3, 0.14, 0.07])

    logging.info("Starting multi-scale ICP registration...")
    # Perform multi-scale ICP registration
    registration_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distances=msa)

    logging.info("Multi-scale ICP registration completed.")
    # Get transformation matrix
    transformation_matrix = registration_result.transformation
    logging.info(f"Transformation Matrix:\n{transformation_matrix}")
    logging.info(f"Fitness: {registration_result.fitness}")

    # Apply transformation matrix to source point cloud
    source_pcd.transform(transformation_matrix)
    logging.info("Applied transformation to source point cloud.")
    # Compute RMSE (Root Mean Square Error) between aligned source and target
    rmse = evaluate_registration(source_pcd, target_pcd, transformation_matrix)
    logging.info(f"RMSE after alignment: {rmse}")

    # Visualize aligned point clouds
    logging.info(f"Writing results to {config['output_path']}")
    
    # Save the aligned point cloud
    try:
        # Get points and colors from Open3D point cloud
        points = np.asarray(source_pcd.points)
        colors = np.asarray(source_pcd.colors)
        
        with laspy.open(config['output_path'], mode='w', header=input_header) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(len(points), header=input_header)
            
            # Correct way to access point cloud data
            point_record.x = points[: , 0]
            point_record.y = points[:, 1]
            point_record. z = points[:, 2]
            
            # Convert colors from [0, 1] float to [0, 65535] uint16 if needed
            if colors.max() <= 1.0:
                colors = (colors * 65535).astype(np.uint16)
            
            point_record.red = colors[:, 0]
            point_record.green = colors[: , 1]
            point_record.blue = colors[:, 2]
            
            writer. write_points(point_record)
        
        logging.info(f"Successfully wrote aligned point cloud to {config['output_path']}")
        
    except Exception as e:
        logging.error(f"Error writing output file: {e}")

    logging.info(f"Aligned point cloud saved to {config['output_path']}")

if __name__ == "__main__":
    main()
    # Input parameters
    # source_path = r"C:\Users\Hanne Hendrickx\Documents\08_GSS_Projects\ZIM\DATA\ICP\Transformed_ALS.ply"
    # target_path = r"C:\Users\Hanne Hendrickx\Documents\08_GSS_Projects\ZIM\DATA\ICP\Poppenhausen_example_UAV.ply"
    # max_correspondence_distances =[0.3, 0.14, 0.07]

    # # Call the main function
    # main(source_path, target_path, max_correspondence_distances)
