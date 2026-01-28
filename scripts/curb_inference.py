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
from scipy.spatial import cKDTree
import geopandas as gpd
from shapely.geometry import Point
import open3d as o3d

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
    return np.vstack((las.x, las.y, las.z,las.blue,las.green,las.blue)).transpose(), las.header

def get_colors(rgb):
    rgb_cor = rgb*(256/np.max(rgb,axis=0))
    hsv = rgb_to_hsv(rgb_cor/256)

    return rgb_cor, hsv

def features_extraction(pcd,radius,features_list):
    features = []
    for r in radius:
        logging.info(f"Extracting features with radius: {r}")
        features.append(jakteristics.compute_features(pcd[:,0:3], search_radius=r, feature_names=features_list))

    rgb_features = pcd[:, 3:6]  
    rgb, hsv = get_colors(rgb_features)
    all_features = np.column_stack(features)
    return np.hstack((all_features, rgb, hsv))


def main():
    try:
        logging.info("Starting main execution")
        args = parse_args()
        logging.info("Parsing configuration file")
        config = read_config(args.config_file)
        logging.info(f"Configuration loaded: {config}")

        pcd, input_header = load_point_cloud(config['pcd_path'])
        logging.info(f"Point cloud loaded with shape: {pcd.shape}")
        features = features_extraction(pcd,[0.2,0.3,0.5,0.7],['verticality','omnivariance','surface_variation', 'linearity','nx', 'ny', 'nz'])
        valid_indices = ~np.isnan(features).any(axis=1)
        features = features[valid_indices]
        
        
        logging.info(f"Features extracted with shape: {features.shape}")
        model = joblib.load(config['model_path'])
        logging.info(f"Model loaded from {config['model_path']}")
        predictions = model.predict(features)
        logging.info(f"Predictions made with shape: {predictions.shape}")
        
        
        logging.info(f"Filtered predictions shape: {predictions.shape}")
        logging.info(f"Writing results to {config['output_path']}")
        hsv_values = features[:, -3:]
        try:
               # Extract HSV values from features (last 3 columns)
            
            
            # Add HSV as extra dimensions (scalar fields)
            input_header.add_extra_dim(laspy.ExtraBytesParams(name="hue", type=np.float64))
            input_header.add_extra_dim(laspy.ExtraBytesParams(name="saturation", type=np.float64))
            input_header.add_extra_dim(laspy.ExtraBytesParams(name="value", type=np.float64))
            with laspy.open(config['output_path'], mode='w', header=input_header) as writer:
                point_record = laspy.ScaleAwarePointRecord.zeros(len(predictions), header=input_header)
                
                point_record.x = pcd[valid_indices,0]
                point_record.y = pcd[valid_indices,1]
                point_record.z = pcd[valid_indices,2]
                point_record.red = pcd[valid_indices,3]
                point_record.green = pcd[valid_indices,4]
                point_record.blue = pcd[valid_indices,5]
                point_record.classification = np.array(predictions)
                point_record.hue = hsv_values[:, 0]
                point_record.saturation = hsv_values[:, 1]
                point_record.value = hsv_values[:, 2]
                
                writer.write_points(point_record)

        except Exception as e:
            logging.error(f"Error writing file to {config['output_path']}")
            raise
    
        # Filter points with classification == 1
        curb_mask = predictions == 1
        curb_points = pcd[valid_indices][curb_mask]
        logging.info(f"Filtered {np.sum(curb_mask)} points with classification 1")

        # Filter out points with saturation > 0.2
        curb_hsv = hsv_values[curb_mask]
        saturation_mask = curb_hsv[:, 1] <= config['saturation_filter']
        curb_points = curb_points[saturation_mask]
        
        logging.info(f"After saturation filter: {len(curb_points)} points remaining")
        # Downsample to 1 meter grid using Open3D
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(curb_points[:, :3])
        downsampled_pcd = pcd_o3d.voxel_down_sample(voxel_size=1.0)
        downsampled_points = np.asarray(downsampled_pcd.points)
        logging.info(f"Downsampled to {len(downsampled_points)} points")

        # Save as LAS
        curb_las_path = config['output_path'].replace('.las', '_curb_class1.las')
        curb_header = laspy.LasHeader(point_format=input_header.point_format, version=input_header.version)
        curb_header.offsets = input_header.offsets
        curb_header.scales = input_header.scales
        with laspy.open(curb_las_path, mode='w', header=curb_header) as writer:
            curb_record = laspy.ScaleAwarePointRecord.zeros(len(downsampled_points), header=curb_header)
            curb_record.x = downsampled_points[:, 0]
            curb_record.y = downsampled_points[:, 1]
            curb_record.z = downsampled_points[:, 2]
            curb_record.classification = np.ones(len(downsampled_points), dtype=np.uint8)
            writer.write_points(curb_record)
        logging.info(f"Saved curb points to {curb_las_path}")

        # Save as shapefile (EPSG:25832)
        geometry = [Point(xy) for xy in downsampled_points[:, :2]]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=config['crs'])
        gdf['z'] = downsampled_points[:, 2]
        gdf['class'] = 1
        shp_path = config['output_path'].replace('.las', '_curb_class1.shp')
        gdf.to_file(shp_path)
        logging.info(f"Saved shapefile to {shp_path}")

    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()