import os
import numpy as np
import laspy 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import joblib
from xgboost import XGBClassifier
import joblib
import logging
import yaml
import sys
import argparse


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

def get_data(data_path: str, features_to_use: list, transdict: dict=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and process point cloud feature data for training.
    
    Args:
        data_path (str): Path to the directory containing feature files
        features_to_use (list): List of feature names to use for training
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) arrays for model training
    """
    try:
        features = []
        classes = []
        others = []

        features_path = ['train_features_0125.npy', 'train_features_025.npy', 'train_features_050.npy',
                        'train_features_075.npy', 'train_features_10.npy', 'train_features_20.npy',
                        'train_features_30.npy', 'train_features_50.npy']

        feature_dict = {
            'eigenvalue_sum': 4, 'omnivariance': 5, 'eigenentropy': 6,
            'anisotropy': 7, 'planarity': 8, 'linearity': 9,
            'surface_variation': 10, 'sphericity': 11, 'verticality': 12,
            'PCA1': 13, 'PCA2': 14, 'number_of_neighbors': 15,
            'point_density': 16,
        }

        logging.info(f"Loading feature files from {data_path}")
        feature_indices = [feature_dict[name] for name in features_to_use]
        
        for file_path in features_path:
            full_path = os.path.join(data_path, file_path)
            try:
                pc = np.load(full_path)
                features.append(pc[:, feature_indices])
                classes.append(pc[:,3])
                others.append(pc[:,17:])
                logging.debug(f"Loaded {file_path}")
            except FileNotFoundError:
                logging.error(f"File not found: {full_path}")
                raise
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
                raise

        feat = np.hstack(features)
        feat = np.hstack((feat,others[0]))
        valid_indices = ~np.isnan(feat).any(axis=1)
        cl = classes[0][valid_indices].astype(np.uint)
        feat = feat[valid_indices]
        logging.info(f"Feature array shape after stacking: {feat.shape}")
        #transdict = {0:0, 1:1, 2:0, 3:0, 4:2, 5:0, 6:0, 7:0, 8:0, 9:0}
        if transdict is not None:
            cl2 = np.array([transdict.get(i, i) for i in cl])

            logging.info(f"Total samples: {len(cl2)}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                feat, cl2,
                stratify=cl2, 
                test_size=0.25,
                shuffle=True,
                random_state=141
            )

            logging.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        else:
            logging.warning("No transformation dictionary provided, using original class labels.")
            logging.info(f"Total samples: {len(cl)}")

            X_train, X_test, y_train, y_test = train_test_split(
                feat, cl,
                stratify=cl,
                test_size=0.25,
                shuffle=True,
                random_state=141
            )
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in get_data: {str(e)}")
        raise

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

        logging.info("Loading training data")
        X_train, X_test, y_train, y_test = get_data(
            data_path=config['data_path'],
            features_to_use=config['features_to_use'],
            transdict=config['transdict']
        )
        
        logging.info("Creating and training model")
        model = get_model(config['model_name'])
        model.fit(X_train, y_train)
        
        logging.info("Making predictions on test set")
        pred = model.predict(X_test)
        
        logging.info("Calculating classification metrics")
        report = classification_report(y_test, pred)
        logging.info(f"\n{report}")
        
        logging.info("Generating confusion matrix")
        cm = confusion_matrix(y_test, pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=model.classes_)
        disp.plot()
        
        logging.info("Saving model")
        joblib.dump(model, config['model_output_path'])
        logging.info(f"Model successfully saved to {config['model_output_path']}")

    except FileNotFoundError as e:
        logging.error(f"File not found error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        raise

if __name__ == '__main__':
   main()
