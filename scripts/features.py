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


def load_data(data_path, plot_centre=None, plot_radius=0, plot_radius_buffer=0, silent=False, headers_of_interest=None, return_num_points=False):
   output_headers = []

   if headers_of_interest is None:
      headers_of_interest = []

   if data_path.endswith('.las') or data_path.endswith('.laz'):

      inFile = laspy.read(data_path)
      header_names = list(inFile.point_format.dimension_names)
      pointcloud = np.vstack((inFile.x, inFile.y, inFile.z))
      if len(headers_of_interest) != 0:
         for header in headers_of_interest:
            if header in header_names:
                  pointcloud = np.vstack((pointcloud, getattr(inFile, header)))
                  output_headers.append(header)
      pointcloud = pointcloud.transpose()

   original_num_points = pointcloud.shape[0]

   if plot_centre is None:
      mins = np.min(pointcloud[:, :2], axis=0)
      maxes = np.max(pointcloud[:, :2], axis=0)
      plot_centre = (maxes + mins) / 2

   if plot_radius > 0:
      distances = np.linalg.norm(pointcloud[:, :2] - plot_centre, axis=1)
      keep_points = distances < plot_radius + plot_radius_buffer
      pointcloud = pointcloud[keep_points]
  
 
   #final_pts = np.hstack(( pointcloud[:, :3],pointcloud[:, -1].reshape(np.shape(pointcloud[:, -1])[0],1)))
   
   print('Point cloud loaded from file : ',data_path)

   if return_num_points:
        return pointcloud, output_headers, original_num_points
   else:
      return pointcloud, output_headers

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


def feature_calculation(xyz: np.array, search_raidus: float, features_name: list, pointdensity: bool)->np.array:

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
    if pointdensity == True and 'number_of_neighbors' == features_name[-1]:
        point_density = (features[:,-1]+1)/((4/3)*np.pi*(search_raidus**3))

    if pointdensity:
        return features,point_density, features_name
    else:
        return features, features_name


def main():
     
    args = parse_args()
    config = read_config(args.config_file)
    print(config['search_raidus'])  
    print(config)

    # open point cloud with x,y,z, r,g,b and selected headers
    pt, headers = load_data(config['input_path'], 
                        plot_centre=None, plot_radius=0, plot_radius_buffer=0, 
                        silent=False, headers_of_interest = config['headers'], 
                            return_num_points=False)

    # normalize rgb color and compute hsv colors
    rgb, hsv = get_colors(pt[:,3:6])


    if config['pointdensity']:
        features, point_density, features_name = feature_calculation(pt[:,0:3], config['search_raidus'], config['features_name'], config['pointdensity'])
        
        # if classification in the point cloud
        if 'classification' in config['headers']: 
            features = np.hstack((features, point_density.reshape((len(point_density),1)),
                          rgb, hsv, pt[:,6].reshape((len(pt[:,6]),1)), pt[:,0:3], pt[:,-1].reshape((len(pt[:,-1]),1))))
            config['features_name'].extend(['point_density','red','blue','green','h','s','v','hag','x','y','z','class'])
        else:
            features = np.hstack((features, point_density.reshape((len(point_density),1)),
                          rgb, hsv, pt[:,-1].reshape((len(pt[:,-1]),1)), pt[:,0:3]))
            config['features_name'].extend(['point_density','red','blue','green','h','s','v','hag','x','y','z'])
    else:
        features, features_name = feature_calculation(pt[:,0:3], config['search_raidus'], config['features_name'], config['pointdensity'])
        
        # if classification in the point cloud
        if 'classification' in config['headers']:
            features = np.hstack((features, rgb, hsv, pt[:,6].reshape((len(pt[:,6]),1)), pt[:,0:3], pt[:,-1].reshape((len(pt[:,-1]),1))))
            config['features_name'].extend(['red','blue','green','h','s','v','hag','x','y','z','class'])
        else:
            features = np.hstack((features, rgb, hsv, pt[:,-1].reshape((len(pt[:,-1]),1)), pt[:,0:3]))
            config['features_name'].extend(['red','blue','green','h','s','v','hag','x','y','z'])

    # print array results size to check process 
    print("Feature array shape: ", np.shape(features))
    print("Feature name array shape: ", np.shape(config['features_name']))

    # create path to save files
    #output_path = os.path.join(config['path_out'],'features_'+ str(config['search_raidus']).replace('.','')+'.h5')
    output_features_path = os.path.join(config['path_out'],'features_names_'+ str(config['search_raidus']).replace('.','')+'.npy')
    # Save features as text file
    output_features_txt = os.path.join(config['path_out'],'features_'+ str(config['search_raidus']).replace('.','')+'.txt')
    np.savetxt(output_features_txt, features, delimiter=',', fmt='%.6f')
    logging.info(f"Features saved to {output_features_txt}")
    
    # Save feature names
    np.save(output_features_path, config['features_name'])
    logging.info(f"Feature names saved to {output_features_path}")
    # # Save arrays to HDF5
    # with h5py.File(output_path, 'w') as f:
    #     f.create_dataset('features', data=features, compression='gzip', compression_opts=9)
    # f.create_dataset('feature_names', data=config['features_name'])
    # print(f"Features saved to {output_path}")
   
if __name__ == '__main__':
   main()
