import logging
import laspy
import numpy as np
import sys
import os 
import argparse
import yaml

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

def main():
    args = parse_args()
    config = read_config(args.config_file)

    input_las_path = config['input_las']
    logging.info(f"Reading LAS file from {input_las_path}")
    input_las = laspy.read(input_las_path)

    logging.info("Reading points from LAS file")
    all_points = np.vstack((input_las.x, input_las.y, input_las.z, input_las.red, input_las.green, input_las.blue, input_las.classification)).transpose()
    logging.info(f"Points read with shape: {all_points.shape}")
    # ...existing code...
    # Split points by classification
    class_1_mask = all_points[:, 6] == 1
    class_2_mask = all_points[:, 6] == 2

    class_1_points = all_points[class_1_mask]
    class_2_points = all_points[class_2_mask]

    logging.info(f"Class 1 points: {class_1_points.shape[0]}")
    logging.info(f"Class 2 points (roof): {class_2_points.shape[0]}")

    # Create output LAS file for class 2 (roof)
    output_las_path = config['output_las']
    # Generate output path based on input path
    input_dir = os.path.dirname(input_las_path)
    input_filename = os.path.splitext(os.path.basename(input_las_path))[0]
    output_las_path = os.path.join(input_dir, f"{input_filename}_roof.las")
    logging.info(f"Writing roof points (class 2) to {output_las_path}")


    # Create new LAS file with class 2 points
    header = laspy.LasHeader(point_format=input_las.header.point_format, version=input_las.header.version)
    header.offsets = input_las.header.offsets
    header.scales = input_las.header.scales

    output_las = laspy.LasData(header)
    output_las.x = class_2_points[:, 0]
    output_las.y = class_2_points[:, 1]
    output_las.z = class_2_points[:, 2]
    output_las.red = class_2_points[:, 3].astype(np.uint16)
    output_las.green = class_2_points[:, 4].astype(np.uint16)
    output_las.blue = class_2_points[:, 5].astype(np.uint16)
    output_las.classification = class_2_points[:, 6].astype(np.uint8)

    output_las.write(output_las_path)
    logging.info(f"Successfully saved roof points to {output_las_path}")

if __name__ == "__main__":  
    main()