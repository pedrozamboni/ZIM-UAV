import laspy
import sys
import argparse
import logging
import argparse
import yaml
import sys
import os

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

def convert_las_to_v12(input_file, output_file=None):
    try:
        # Open the input LAS file
        las_data = laspy.read(input_file)
        
        # Check the version
        version = las_data.header.version
        version_float = version.major + version.minor / 10.0

        logging.info(f"LAS file version: {version_float}")
        
        # If version is higher than 1.2, convert to 1.2
        if version_float > 1.2:
            logging.info("Converting to version 1.2...")
            
            # Create a new LAS 1.2 file
            point_format_id = 3

            header = laspy.LasHeader(version="1.2", point_format=point_format_id)
            header.offsets = las_data.header.offsets
            header.scales = las_data.header.scales
            
            # Create new LAS with converted header
            new_las = laspy.LasData(header)
            
            # Copy point records
            compatible_dims = header.point_format.dimension_names
            for dim in compatible_dims:
                if hasattr(las_data, dim):
                    setattr(new_las, dim, getattr(las_data, dim))
                        # Save to output file
            output_path = output_file if output_file else input_file.replace('.las', '_v1.2.las')
            new_las.write(output_path)
            logging.info(f"Converted file saved as: {output_path}")
        else:
            logging.info("No conversion needed. File is already version 1.2 or lower.")
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_file)
    logging.info(f"Configuration: {config}")

    # parser = argparse.ArgumentParser(description='Convert LAS file to version 1.2')
    # parser.add_argument('input_file', help='Input LAS file path')
    # parser.add_argument('--output', help='Output file path (optional)')
    convert_las_to_v12(config['input_file'], config['output_file'])