import numpy as np
import argparse
import os
import yaml
import laspy
import logging
import sys
from matplotlib.colors import rgb_to_hsv
from jakteristics import las_utils, compute_features, FEATURE_NAMES
import gc
import time 
import joblib
import open3d as o3d
from scipy.spatial import cKDTree

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
    return f'{base}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(get_log_filename(sys.argv[1])),
        logging.StreamHandler()
    ]
)


def get_colors(rgb: np.array) -> tuple[np.array, np.array]:
    """
    Normalize rgb values (0-1) and convert it to HSV (hue, saturation and value).

    Parameters
    ----------
    rgb : np.array with RGB value for each point

    Returns
    -------
    tuple[np.array, np.array]
        Normalized RGB values and HSV values
    """
    try:
        logging.info(f"Processing RGB array of shape {rgb.shape}")
        
        # normalize rgb values
        rgb_cor = rgb*(256/np.max(rgb,axis=0))
        logging.debug(f"Normalized RGB array shape: {rgb_cor.shape}, dtype: {rgb_cor.dtype}")
        
        # convert to hsv colors
        hsv = rgb_to_hsv(rgb_cor/256)
        logging.debug(f"Converted HSV array shape: {hsv.shape}, dtype: {hsv.dtype}")

        return rgb_cor, hsv
        
    except ValueError as e:
        logging.error(f"Error processing RGB values: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in color processing: {str(e)}")
        raise


def feature_calculation(xyz: np.array, search_raidus: float, features_name: list)->np.array:
    '''
    Calculate geometric features for point cloud

    Parameters
    ----------
    xyz : np.array with xyz from the point cloud 
    search_radius : radius of an elipsoid to get neighbors points
    features : features to use 

    for features: 
        if point cloud has classification:
                ['red','green','blue','distance','classification']
        if not:
                ['red','green','blue','distance','classification']
    '''
    try:
        logging.info(f'Computing features for radius {search_raidus}')
        logging.info(f'Features to compute: {features_name}')

        # calculate features for a given search radius 
        features = compute_features(xyz, search_raidus, feature_names=features_name)
        
        logging.info(f'Features computed with shape: {features.shape}')

        return features, features_name

    except ValueError as e:
        logging.error(f"Value error in feature calculation: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in feature calculation: {str(e)}")
        raise


def main():
    logging.info('Starting script.')
    try:
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")

        logging.info(f"Loading model from {config['model_path']}")
        model = joblib.load(config['model_path'])
        logging.info('Model loaded successfully')

        logging.info(f"Processing input file: {config['input_las_path']}")
        with laspy.open(config['input_las_path']) as input_las:
            input_header = input_las.header
            point_count = input_header.point_count
            logging.info(f"Total points in input file: {point_count}")
            
            logging.info("Reading points from LAS file")
            points = input_las.read()
            all_points = np.vstack((points.x, points.y, points.z, points.red, points.green, points.blue, points.distance)).transpose()
            logging.info(f"Points read with shape: {all_points.shape}")
            logging.info(f"Distance stats - max: {max(all_points[:,-1])}, min: {min(all_points[:,-1])}, mean: {np.mean(all_points[:,-1])}")

            if config['voxel_downsample']:
                    logging.info("Applying voxel downsampling")
                # Create Open3D point cloud from numpy array
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(all_points[:, 0:3])
                    pcd.colors = o3d.utility.Vector3dVector(all_points[:, 3:6] / 65535.0)  # Normalize to [0,1]

                    # Downsample the point cloud
                    #voxel_size = config.get('voxel_size', 0.1)  # Get from config or use default
                    pcd_down = pcd.voxel_down_sample(voxel_size=config['voxel_downsample'])

                    # Convert back to numpy array
                    downsampled_xyz = np.asarray(pcd_down.points)
                    downsampled_colors = np.asarray(pcd_down.colors) * 65535.0
                    
                    # Find nearest neighbor indices in original point cloud for each downsampled point
                    tree = cKDTree(all_points[:, 0:3])
                    distances, indices = tree.query(downsampled_xyz)
                    
                    # Extract distance values from original points using nearest neighbor indices
                    downsampled_distance = all_points[indices, -1].reshape(-1, 1)
                    
                    pt = np.hstack([downsampled_xyz, downsampled_colors, downsampled_distance])
                    logging.info(f"Downsampled point cloud shape: {pt.shape}")
                    logging.info(f"Distance stats - max: {max(pt[:,-1])}, min: {min(pt[:,-1])}, mean: {np.mean(pt[:,-1])}")

                    del pcd, pcd_down,all_points
                    gc.collect()
            else:
                logging.info("No downsampling applied")
                pt = all_points
                del all_points
                gc.collect()
                
            
            logging.info("Sorting points by coordinates")
            sorted_indices = np.lexsort((pt[:,1], pt[:,0]))
            pt = pt[sorted_indices]
            
            chunk_size = 10000000
            num_chunks = (len(pt) + chunk_size - 1) // chunk_size
            chunk_index = 0
            logging.info(f"Processing {num_chunks} chunks of {chunk_size} points each")

            for chunk_start in range(0, len(pt), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(pt))
                logging.info(f"Processing chunk {chunk_start//chunk_size + 1}/{num_chunks}, points {chunk_start} to {chunk_end}")
                pt_chunk = pt[chunk_start:chunk_end]
                rgb, hsv = get_colors(pt_chunk[:,3:6])
                features_list = []
                for radius in [0.125, 0.25, 0.5,0.75,1,2,3,5]:
                    start2 = time.time()
                    features, features_name = feature_calculation(pt_chunk[:,0:3], radius,['surface_variation',
                                                        'sphericity',
                                                        'verticality'])
                    features_list.append(features)
                    end2 = time.time()
                    logging.info(f'Features computed for radius {radius} in {end2 - start2:.2f} seconds')
                    del features
                    gc.collect()

                features = np.hstack((np.hstack(features_list), rgb, hsv, pt_chunk[:,-1].reshape(-1,1)))
                valid_indices = ~np.isnan(features).any(axis=1)
                feat = features[valid_indices]
                logging.info(f"Feature matrix shape: {np.shape(feat)}")

                logging.info("Performing prediction")
                pred = model.predict(feat)
                
                # logging.info("Applying classification filters")
                # #mask_class1 = (pred == 1) & (feat[:, -3] > 0.27)
                # #mask_class2 = (pred == 2) & (feat[:, -1] < 1)
                # #pred[mask_class1 | mask_class2] = 0
            
                output_path = os.path.join(config['output_dir'], f'inf_chunk2_{chunk_index}.las')
                logging.info(f"Writing results to {output_path}")
                try:
                    with laspy.open(output_path, mode='w', header=input_header) as writer:
                        point_record = laspy.ScaleAwarePointRecord.zeros(len(pred), header=input_header)
                        
                        point_record.x = pt_chunk[valid_indices,0]
                        point_record.y = pt_chunk[valid_indices,1]
                        point_record.z = pt_chunk[valid_indices,2]
                        point_record.red = rgb[valid_indices,0]
                        point_record.green = rgb[valid_indices,1]
                        point_record.blue = rgb[valid_indices,2]
                        point_record.classification = np.array(pred)
                        point_record.distance = pt_chunk[valid_indices,6]
                        writer.write_points(point_record)

                except Exception as e:
                    logging.error(f"Error writing chunk {chunk_index} to file: {str(e)}")
                    raise

                del feat, valid_indices, features_list, features, rgb, hsv, pt_chunk, pred
                gc.collect()
                logging.info(f"Completed processing chunk {chunk_index}")
                chunk_index += 1
        
        logging.info("Processing completed successfully")
    except Exception as e:
         logging.error(f"An error occurred: {str(e)}")
         raise
if __name__ == '__main__':
   main()
