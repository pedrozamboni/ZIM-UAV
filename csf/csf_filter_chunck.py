import argparse
import CSF
import laspy
import os
import numpy as np
import logging
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

def csf_filter(pcd_path: str, cloth_resolution: int, max_interations: int, path_out: str, filename, class_threshold=0.5, sloopsmooth=True, chunk_size=100000000):
    logging.info(f"Starting CSF filtering on file: {pcd_path}")
    logging.info(f"Parameters: cloth_resolution={cloth_resolution}, max_iterations={max_interations}, class_threshold={class_threshold}")
    
    # Create output directories
    ground_dir = os.path.join(path_out, filename, 'ground')
    non_ground_dir = os.path.join(path_out, filename, 'nonground')
    os.makedirs(ground_dir, exist_ok=True)
    os.makedirs(non_ground_dir, exist_ok=True)
    
   
    # Read the input file
    with laspy.open(pcd_path) as input_las:
        input_header = input_las.header
        point_count = input_header.point_count
        logging.info(f"Total points to process: {point_count}")
        
        # Process chunks
        for chunk_id, points in enumerate(input_las.chunk_iterator(chunk_size)):
            logging.info(f"Processing chunk {chunk_id} with {len(points)} points")
            # # Extract coordinates
            xyz = np.vstack((points.x, points.y, points.z)).transpose()
            
            # Initialize CSF filter
            csf = CSF.CSF()
            
            # Parameter settings
            csf.params.bSloopSmooth = sloopsmooth
            csf.params.cloth_resolution = cloth_resolution
            csf.params.max_interations = max_interations
            csf.params.class_threshold = class_threshold
            
            # Set point cloud and filter
            csf.setPointCloud(xyz)
            ground = CSF.VecInt()
            non_ground = CSF.VecInt()
            csf.do_filtering(ground, non_ground)
            
            #Save ground points for this chunk
            ground_points = points[np.array(ground)]
            ground_path = os.path.join(ground_dir, f'chunk_{chunk_id}.las')
            ground_file = laspy.create(point_format=input_header.point_format, file_version=input_header.version)
            ground_file.header.offsets = input_header.offsets
            ground_file.header.scales = input_header.scales
            ground_file.points = ground_points
            ground_file.write(ground_path)
            
            # Save non-ground points for this chunk
            non_ground_points = points[np.array(non_ground)]
            non_ground_path = os.path.join(non_ground_dir, f'chunk_{chunk_id}.las')
            non_ground_file = laspy.create(point_format=input_header.point_format, file_version=input_header.version)
            non_ground_file.header.offsets = input_header.offsets
            non_ground_file.header.scales = input_header.scales
            non_ground_file.points = non_ground_points
            non_ground_file.write(non_ground_path)
            
            logging.info(f"Saved chunk {chunk_id}: {len(ground_points)} ground points, {len(non_ground_points)} non-ground points")
            
            # # Clean up memory
            del xyz, ground_points, non_ground_points
            del ground_file, non_ground_file
        
  

def main():


    logging.info('Starting script.')
    try:
        
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")

        csf_filter(config['pcd_path'],config['cloth_resolution'],config['max_interations'],config['path_out'],config['filename'],config['class_threshold']
                    ,config['sloopsmooth'])

        logging.info("CSF filtering completed successfully.")
        logging.info("Process finished")
        logging.info(f"Point cloud saved: {config['path_out']}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        exit(1)
        
if __name__ == '__main__':
   main()