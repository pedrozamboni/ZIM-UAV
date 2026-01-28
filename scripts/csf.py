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

def csf_filter(pcd_path: str, cloth_resolution: int, max_interations: int, path_out: str, filename, class_threshold=0.5, sloopsmooth=True):
    logging.info(f"Starting CSF filtering on file: {pcd_path}")
    logging.info(f"Parameters: cloth_resolution={cloth_resolution}, max_iterations={max_interations}, class_threshold={class_threshold}")
    
    try:
        # Read the input file
        with laspy.open(pcd_path) as input_las:
            las_data = input_las.read()
            input_header = input_las.header
            point_count = input_header.point_count
            logging.info(f"Total points to process: {point_count}")
            
            # Extract coordinates
            xyz = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
            
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
            
            # Get ground and non-ground points
            ground_points = las_data[np.array(ground)]
            non_ground_points = las_data[np.array(non_ground)]
            
            logging.info(f"Ground points: {len(ground_points)}, Non-ground points: {len(non_ground_points)}")
            
            # Create output directory if it doesn't exist
            os.makedirs(path_out, exist_ok=True)
            
            # Save ground points
            ground_path = os.path.join(path_out, filename + '_ground.las')
            groundFile = laspy.create(point_format=input_header.point_format, file_version=input_header.version)
            groundFile.header.scales = input_header.scales
            groundFile.header.offsets = input_header.offsets
            for dim in input_header.point_format.dimension_names:
                groundFile[dim] = ground_points[dim]
            groundFile.write(ground_path)
            logging.info(f"Ground points saved to: {ground_path}")
            
            # Save non-ground points
            non_ground_path = os.path.join(path_out, filename + '_non_ground.las')
            non_groundFile = laspy.create(point_format=input_header.point_format, file_version=input_header.version)
            non_groundFile.header.scales = input_header.scales
            non_groundFile.header.offsets = input_header.offsets
            for dim in input_header.point_format.dimension_names:
                non_groundFile[dim] = non_ground_points[dim]
            non_groundFile.write(non_ground_path)
            logging.info(f"Non-ground points saved to: {non_ground_path}")
            
    except Exception as e:
        logging.error(f"Error in CSF filtering: {str(e)}")
        raise

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