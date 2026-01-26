import laspy
import numpy as np
import jakteristics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os
import gc
import logging
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import joblib
import yaml
import sys
import argparse
import logging
from matplotlib.colors import rgb_to_hsv


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

def load_point_cloud(las_path):
    """Load LAS/LAZ file and return coordinates array"""
    las = laspy.read(las_path)
    return np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue)).transpose()

def get_colors(rgb):
    rgb_cor = rgb*(256/np.max(rgb,axis=0))
    hsv = rgb_to_hsv(rgb_cor/256)

    return rgb_cor, hsv

def get_folder_name(path):
    #logging.info(f"Reading folder names from path: {path}")
    return [os.path.join(path, f) for f in os.listdir(path)]

def get_file_name(path):
    #logging.info(f"Reading file names from folder: {path}")
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.las') or f.endswith('.laz')]

def training_data(path):
    folders = get_folder_name(path)
    logging.info(f"Found {len(folders)} folders with curb and non-curb data")
    all_data = []
    curbs = []
    non_curbs = []
    for f in folders:
        files = get_file_name(f)
        for file in files:
            if 'non' not in file:
                logging.info(f"Reading files : {file}")
                pcd = load_point_cloud(file)

                curbs.append(np.hstack((pcd, np.ones(np.shape(pcd)[0]).reshape(-1, 1))))
            else:
                logging.info(f"Reading files : {file}")
                pcd = load_point_cloud(file)
                non_curbs.append(np.hstack((pcd, np.zeros(np.shape(pcd)[0]).reshape(-1, 1))))

    for i in range(len(curbs)):
        all_data.append(np.vstack((curbs[i], non_curbs[i])))
    logging.info(f"Total datasets created: {len(all_data)}")
    return all_data

def features_extraction(pcd,radius,features_list):
    features = []
    for r in radius:
        logging.info(f"Extracting features with radius: {r}")
        features.append(jakteristics.compute_features(pcd[:,0:3], search_radius=r, feature_names=features_list))
    
    classification_pp = pcd[:, -1].reshape(-1, 1)
    all_features = np.column_stack(features)
    
    rgb_features = pcd[:, 3:6]  
    rgb, hsv = get_colors(rgb_features)# Extract RGB values
    return np.hstack((all_features, rgb, hsv, classification_pp))

def get_features(data,radius,features_list):
    all_features = []
    for i in range(len(data)):
        logging.info(f"Extracting features for dataset {i+1}/{len(data)}")
        feats = features_extraction(data[i],radius,features_list)
        all_features.append(feats)
    logging.info(f"Feature extraction completed for all datasets")
    return np.vstack(all_features)


def get_model(model_name: str) -> RandomForestClassifier | XGBClassifier:
    """
    Creates and returns a machine learning model based on the specified name.

    Args:
        model_name (str): Name of the model to create ('xgboost' or 'random_forest')

    Returns:
        Union[RandomForestClassifier, XGBClassifier]: The initialized model

    Raises:
        ValueError: If model_name is not supported
    """
    try:
        logging.info(f"Creating {model_name} model")
        if model_name == 'xgboost':
            model = XGBClassifier(
                max_depth=18, 
                n_estimators=100, 
                n_jobs=10,
                verbosity=2, 
                use_label_encoder=False
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                max_depth=18, 
                n_estimators=100, 
                n_jobs=10,
                verbose=100, 
                bootstrap=False
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        logging.info(f"Successfully created {model_name} model")
        return model
        
    except Exception as e:
        logging.error(f"Error creating model {model_name}: {str(e)}")
        raise

def main():

    try:
        logging.info("Starting main execution")
        
        args = parse_args()
        logging.info("Parsing configuration file")
        config = read_config(args.config_file)
        logging.info(f"Configuration loaded: {config}")

        #path =  '/home/pedro/Documents/Documents/projects/pointcloud/pp_shp/curb_rf/training_data/'
        all_data = training_data(config['data_path'])
        all_features = get_features(all_data,[0.2,0.3,0.5,0.7],['verticality','omnivariance','surface_variation', 'linearity','nx', 'ny', 'nz'])
        print(np.shape(all_features))
        mask = ~np.isnan(all_features).any(axis=1)
        cleaned_features = all_features[mask]

        del all_data, all_features
        gc.collect()

        class_0_indices = np.where(cleaned_features[:, -1] == 0)[0]
        class_1_indices = np.where(cleaned_features[:, -1] == 1)[0]

        np.random.seed(42)  
        sampled_class_0_indices = np.random.choice(
            class_0_indices, 
            size=int(0.60 * len(class_0_indices)), 
            replace=False
        )

        selected_indices = np.concatenate([sampled_class_0_indices, class_1_indices])
        balanced_features = cleaned_features[selected_indices]
        logging.info("Original class distribution:")
        logging.info(f"Class 0: {len(class_0_indices)}")
        logging.info(f"Class 1: {len(class_1_indices)}")
        logging.info("Balanced class distribution:")
        logging.info(f"Class 0: {len(sampled_class_0_indices)}")
        logging.info(f"Class 1: {len(class_1_indices)}")

        # Update X and y with balanced dataset
        X = balanced_features[:, :-1]  # All columns except last
        y = balanced_features[:, -1]   # Last column (classification)

        logging.info(f"Final dataset shape: {X.shape}, Labels shape: {y.shape}")
        logging.info("Splitting data into training and test sets")
        X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, 
        y,
        test_size=0.3,  
        stratify=y,     
        random_state=567)

        del balanced_features, cleaned_features, X, y
        gc.collect()

        model = get_model(config['model_name'])
        logging.info("Training model")
        model.fit(X_trainval, y_trainval)
        logging.info("Model training completed")
        logging.info("Making predictions")
        pred = model.predict(X_test)
        report = classification_report(y_test, pred)
        logging.info(f"Classification report:\n{report}")

        logging.info(f"Saving model to disk in folder {config['model_output_path']}")
        joblib.dump(model, config['model_output_path'])

    except FileNotFoundError as e:
        logging.error(f"File not found error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        raise

        
if __name__ == "__main__":
    main()
