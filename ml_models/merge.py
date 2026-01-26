import os
import laspy
import numpy as np
from pathlib import Path
import logging
import argparse
import yaml 


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("merge.log"),
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

def merge_las_files(input_folder, output_file):
    logging.info(f"Starting merge process from folder: {input_folder}")
    
    # Get all .las files in the input folder
    las_files = list(Path(input_folder).glob('*.las'))
    logging.info(f"Found {len(las_files)} LAS files to merge")
    
    if not las_files:
        logging.warning("No LAS files found in the specified folder")
        return

    # Read the first file to get header information
    logging.info(f"Reading first file: {las_files[0]}")
    first_las = laspy.read(las_files[0])
    merged_las = laspy.create(point_format=first_las.header.point_format)
    
    # Initialize lists
    points_list = []
    red_list = []
    green_list = []
    blue_list = []
    classification_list = [] if hasattr(first_las, 'classification') else None
    
    # Process each LAS file
    total_points = 0
    for las_path in las_files:
        logging.info(f"Processing file: {las_path}")
        las = laspy.read(las_path)
        points_list.append(np.array([las.x, las.y, las.z]).transpose())
        total_points += len(las.x)
        
        # Append RGB values if they exist
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            red_list.append(las.red)
            green_list.append(las.green)
            blue_list.append(las.blue)
            logging.debug(f"RGB data found in file: {las_path}")
        
        # Append classification if it exists
        if  hasattr(las, 'classification'):
            print('classification found')
            classification_list.append(las.classification)
            logging.debug(f"Classification data found in file: {las_path}")
        # Delete las object to free memory
        del las
        logging.debug(f"Freed memory for file: {las_path}")

    logging.info(f"Total points to merge: {total_points}")
    
    # Concatenate all points
    logging.info("Merging point coordinates...")
    all_points = np.vstack(points_list)
    del points_list
    
    # Set points in the merged file
    merged_las.x = all_points[:, 0]
    merged_las.y = all_points[:, 1]
    merged_las.z = all_points[:, 2]
    del all_points
    
    # Set RGB values if they exist
    if red_list and green_list and blue_list:
        logging.info("Merging RGB values...")
        merged_las.red = np.concatenate(red_list)
        merged_las.green = np.concatenate(green_list)
        merged_las.blue = np.concatenate(blue_list)
        merged_las.classification = np.concatenate(classification_list) if classification_list else None
        del red_list, green_list, blue_list, classification_list
    
    # Write the merged file
    logging.info(f"Writing merged file to: {output_file}")
    merged_las.write(output_file)
    logging.info("Merge process completed successfully")

def main():
    args = parse_args()
    logging.info("Reading configuration file")
    config = read_config(args.config_file)
    input_folder = config.get("input_folder", "")
    output_file = config.get("output_file", "")
    
    if not input_folder or not output_file:
        logging.error("Input folder or output file not specified in config")
        return
        
    merge_las_files(input_folder, output_file)

if __name__ == "__main__":
    main()

