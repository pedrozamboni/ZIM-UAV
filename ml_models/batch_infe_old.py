import numpy as np
import argparse
import os
import yaml
import laspy
import logging
import sys
from matplotlib.colors import rgb_to_hsv
from jakteristics import las_utils, compute_features, FEATURE_NAMES
import h5py
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



def get_colors(rgb:np.array):

    '''
    Normalize rgb values (0-1) and convert it to HSV (hue, saturation and value).

    Parameters
    ----------
    rgb : np.array with RGB value for each point

    '''
    # normalize rgb values
    rgb_cor = rgb*(256/np.max(rgb,axis=0))
    
    # convert to hsv colors
    hsv = rgb_to_hsv(rgb_cor/256)

    return rgb_cor, hsv


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
    print('Computing feature for the radius of ', search_raidus)
    print(features_name)

    # calculate features for a given search radius 
    features = compute_features(xyz, search_raidus, feature_names=features_name)

    return features, features_name


def main():

    model = joblib.load('/home/pedro/Documents/Documents/projects/pointcloud/zim_results/ml_models/trained_model/rf_3c_pkl')

    print('model loaded')
    print('extracting features and classifying for chunck of 10 million points')
    with laspy.open('/home/pedro/Documents/Documents/projects/pointcloud/zim_results/output/mesh/poppenhausen_ground_final.las') as input_las:
        input_header = input_las.header
        point_count = input_header.point_count
        
      
        for chunk_id, points in enumerate(input_las.chunk_iterator(3000000)):
            # # Extract coordinates
            start = time.time()
            pt = np.vstack((points.x, points.y, points.z,points.red,points.green,points.blue, points.distance)).transpose()
            rgb, hsv = get_colors(pt[:,3:6])

            features_list = []

            for radius in [0.125, 0.25, 0.5,0.75,1,2,3,5]:
                start2 = time.time()
                features, features_name = feature_calculation(pt[:,0:3], radius,['surface_variation',
                                                                                 'sphericity',
                                                                                 'verticality'])
                features_list.append(features)
                end2 = time.time()
                print('Time to compute features for radius ', radius, ' is ', end2 - start2)
                del features
            
            features = np.hstack((np.hstack(features_list), rgb, hsv, pt[:,-1].reshape(-1,1)))
            valid_indices = ~np.isnan(features).any(axis=1)
            feat = features[valid_indices]
            print(np.shape(feat))

            
            end = time.time()
            print('Time to compute features for chunk ', chunk_id, ' is ', end - start)

            pred = model.predict(feat)

            # Create new LAS file with the same header as input
            with laspy.open('/home/pedro/Documents/Documents/projects/pointcloud/zim_results/output/mesh/pp_pred_3.las', mode="w", header=input_header) as writer:
                # Create point record with valid points only
                point_record = laspy.ScaleAwarePointRecord.zeros(len(pred), header=input_header)
                
                # Assign coordinates for valid points
                point_record.x = pt[valid_indices,0]
                point_record.y = pt[valid_indices,1]
                point_record.z = pt[valid_indices,2]

                # Assign RGB values for valid points
                point_record.red = rgb[valid_indices,0]
                point_record.green = rgb[valid_indices,1]
                point_record.blue = rgb[valid_indices,2]
                
                # Add custom fields for prediction, distance and HSV
                point_record.classification = np.array(pred)  # Store prediction as classification
                point_record.distance = pt[valid_indices,6]
                point_record.hue = hsv[valid_indices,0]
                point_record.saturation = hsv[valid_indices,1]
                point_record.value = hsv[valid_indices,2]

                writer.write_points(point_record)
            del features_list, features
            del rgb
            del hsv
            del pt
            break
if __name__ == '__main__':
   main()
