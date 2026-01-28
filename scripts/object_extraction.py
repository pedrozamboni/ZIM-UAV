import numpy as np
import argparse
import os
import yaml
import laspy
import logging
import sys
from matplotlib.colors import rgb_to_hsv
import jakteristics
import gc
import time 
import joblib

def read_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def parse_args()-> dict:
    parser = argparse.ArgumentParser(description ='Feature calculation')
    parser.add_argument('config_file')

    args = parser.parse_args()
    return args

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

def load_point_cloud(las_path):
    """Load LAS/LAZ file and return coordinates array"""
    las = laspy.read(las_path)
    return np.vstack((las.x, las.y, las.z,las.blue,las.green,las.blue,las.classification)).transpose(), las.header

def main():
    args = parse_args()
    config = read_config(args.config_file)

    las_path = config['input_las']
    output_path = config['output_features']

    logging.info(f"Loading point cloud from {las_path}")
    pcd, header = load_point_cloud(las_path)
    logging.info(f"Point cloud loaded with {pcd.shape[0]} points")

    classifications = pcd[:, 6].astype(np.int32)
    logging.info(f"Extracted classifications with unique values: {np.unique(classifications)}")
    classes_names = { 1: 'ground', 2: 'roof'}
    
    ### slicing point cloud per class
    for class_id, class_name in classes_names.items():
        logging.info(f"Processing class {class_name} (ID: {class_id})")
        mask = classifications == class_id
        pcd_class = pcd[mask][:, :6]  # Exclude classification column for feature computation
        logging.info(f"Class {class_name} has {pcd_class.shape[0]} points")
        if pcd_class.shape[0] == 0:
            logging.warning(f"No points found for class {class_name}, skipping.")
            continue
        
        # Save point cloud for this class
        output_class_path = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.splitext(os.path.basename(output_path))[0]}_{class_name}.las"
        )
        logging.info(f"Saving {class_name} point cloud to {output_class_path}")

        # Create new LAS file with filtered points
        las_class = laspy.LasData(header)
        las_class.x = pcd_class[:, 0]
        las_class.y = pcd_class[:, 1]
        las_class.z = pcd_class[:, 2]
        las_class.blue = pcd_class[:, 3].astype(np.uint16)
        las_class.green = pcd_class[:, 4].astype(np.uint16)
        las_class.red = pcd_class[:, 5].astype(np.uint16)
        las_class.classification = np.full(pcd_class.shape[0], class_id, dtype=np.uint8)

        las_class.write(output_class_path)
        logging.info(f"Successfully saved {pcd_class.shape[0]} points to {output_class_path}")