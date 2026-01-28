import laspy
import numpy as np
import pdal
import logging
import argparse
import yaml
import json
import geopandas as gpd
from shapely.ops import unary_union
import os

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

# def argparse
def parse_args()-> dict:
    """Parse command line arguments.

    This function sets up an argument parser to handle command line inputs,
    specifically looking for a config file path.

    Returns:
        dict: Parsed command line arguments containing the config file path
    """
    parser = argparse.ArgumentParser(description ='Crop point cloud using shapefile')
    parser.add_argument('config_file')
    args = parser.parse_args()
    return args

## crop function
def clip_pointcloud_with_pdal(las_path: str, output_path: str, polygon_gdf: gpd.GeoDataFrame) -> int:
    """
    Clips a point cloud file using a polygon geometry with PDAL.

    Parameters:
        las_path (str): Path to the input LAS/LAZ point cloud file.
        output_path (str): Path where the cropped LAS/LAZ file will be saved.
        polygon_gdf (geopandas.GeoDataFrame): GeoDataFrame containing the polygon geometry for cropping.

    Returns:
        int: The number of points remaining after cropping, or 0 if an error occurs.
    """
    # Open the input LAS file
    las_data = laspy.read(las_path)
    
    # Check the version
    version = las_data.header.version
    version_float = version.major + version.minor / 10.0

    print(f"LAS file version: {version_float}")
    
    # If version is higher than 1.2, convert to 1.2
    if version_float > 1.2:
        logging.info("Convert to version 1.2...")
        logging.info("Use converter.py")
    else:
        logging.info("Crop initialization")

        wkt_polygon = polygon_gdf.geometry[0].wkt
        pipeline = {
            "pipeline": [
            {
                "type": "readers.las",
                "filename": las_path
            },
            {
                "type": "filters.crop",
                "polygon": wkt_polygon
            },
            {
                "type": "writers.las",
                "filename": output_path, "extra_dims": "all",
                "forward": "all",
                "minor_version": 4,
                "dataformat_id": 8  
            }
        ]
    }
 
    
    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline))
        pipeline.execute()
        logging.info("Cropping completed successfully.")
        return len(pipeline.arrays[0])
    
    except Exception as e:
        logging.error(f"Error executing PDAL pipeline: {e}")
        return 0

def main():
    """
    Main function to process and crop point cloud data using a shapefile.
    This function performs the following operations:
    1. Loads configuration from a specified file
    2. Reads and processes a shapefile to create a merged and buffered geometry
    3. Uses PDAL to clip a point cloud based on the processed geometry
    The function expects a configuration file with the following keys:
    - shapefile_path: Path to the input shapefile
    - polygon_type: List of polygon types to filter
    - point_cloud_path: Path to the input point cloud file
    - output_las: Path for the output cropped point cloud
    Raises:
        Exception: Any error during processing will be logged and cause the script to exit
    Returns:
        None
    Note:
        The function uses a fixed buffer distance of 0.7 units and applies a miter join style (2)
        for the geometry buffer operation.
    """
    logging.info("Starting script.")

    try:
        args = parse_args()
        config = read_config(args.config_file)  
        
        logging.info("Configuration file loaded successfully.")
        logging.info(f"Configuration: {config}")
        shp_gpd = gpd.read_file(config['shapefile_path'])
        class_list = config['polygon_type']
        filtered_gdf = shp_gpd[shp_gpd['type'].str.lower().isin(class_list)]
        merged_geom = unary_union(filtered_gdf.geometry)
        merged_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=shp_gpd.crs)

        buffer_distance = config['buffer_distance'] if 'buffer_distance' in config else 0.3
        buffered_geom = merged_gdf.geometry[0].buffer(buffer_distance, join_style=2)
        buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_geom], crs=merged_gdf.crs) 
        logging.info("Shapefile processed and geometry prepared.")
        
        points_remaining = clip_pointcloud_with_pdal(
            las_path = config['point_cloud_path'],
            output_path = config['output_las'],
            polygon_gdf = buffered_gdf,
)

        logging.info("Process finished")
        logging.info(f"Point cloud saved: {config['output_las']}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        exit(1)




if __name__ == "__main__":
    main()