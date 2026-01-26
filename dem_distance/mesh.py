import json
import pdal
import rasterio
import numpy as np
from scipy import ndimage
import time
import open3d as o3d
import logging
import yaml
import sys
import os
import argparse


def get_log_filename(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return f'{base}.log'

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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(get_log_filename(sys.argv[1])),
        logging.StreamHandler()
    ]
)
# def mesh_generation(las_path: str, raw_mesh: str, final_mesh: str) -> None:
#     """
#     Generate a mesh from LAS file by creating a DEM and filling nodata values.
    
#     Args:
#         las_path (str): Path to input LAS file
#         raw_mesh (str): Path for intermediate raw mesh output
#         final_mesh (str): Path for final interpolated mesh output
    
#     Raises:
#         FileNotFoundError: If input LAS file doesn't exist
#         RuntimeError: If PDAL pipeline execution fails
#         rasterio.errors.RasterioError: If there are issues with raster operations
#     """
    
#     try:
#         logging.info(f"Starting mesh generation from {las_path}")

#         pipeline_json = {
#             "pipeline": [
#                 las_path,
#                 {
#                     "type": "filters.reprojection",
#                     "in_srs": "EPSG:25832",
#                     "out_srs": "EPSG:25832"
#                 },
#                 {
#                     "type": "writers.gdal",
#                     "filename": raw_mesh,
#                     "resolution": 1.0,
#                     "radius": 2.0,
#                     "output_type": "idw",
#                     "nodata": -9999,
#                 }
#             ]
#         }

#         logging.info("Executing PDAL pipeline...")
#         p = pdal.Pipeline(json.dumps(pipeline_json))
#         if not p.execute():
#             raise RuntimeError("PDAL pipeline execution failed")

#         time.sleep(15)
#         logging.info("Reading raw mesh...")
        
#         with rasterio.open(raw_mesh) as src:
#             dem = src.read(1)
#             profile = src.profile
#             transform = src.transform

#         logging.info(f"Raw mesh shape: {dem.shape}")
#         nodata = profile.get("nodata", -9999)

#         logging.info("Filling nodata values...")
#         mask = (dem == nodata)

#         filled = ndimage.distance_transform_edt(
#             mask,
#             return_distances=False,
#             return_indices=True
#         )

#         dem_filled = dem[tuple(filled)]

#         logging.info(f"Saving final mesh to {final_mesh}")
#         profile.update(dtype=rasterio.float32, nodata=None)

#         with rasterio.open(final_mesh, "w", **profile) as dst:
#             dst.write(dem_filled.astype(rasterio.float32), 1)
        
#         logging.info("Mesh generation complete")

#     except FileNotFoundError as e:
#         logging.error(f"Input file not found: {e}")
#         raise
#     except pdal.pipeline.PipelineError as e:
#         logging.error(f"PDAL pipeline error: {e}")
#         raise
#     except rasterio.errors.RasterioError as e:
#         logging.error(f"Raster operation error: {e}")
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         raise

def mesh_generation(las_path: str, raw_mesh: str, final_mesh: str) -> None:
    """
    Generate a mesh from LAS file by creating a DEM and filling nodata values.
    
    Args:
        las_path (str): Path to input LAS file
        raw_mesh (str): Path for intermediate raw mesh output
        final_mesh (str): Path for final interpolated mesh output
    
    Raises:
        FileNotFoundError: If input LAS file doesn't exist
        RuntimeError: If PDAL pipeline execution fails
        rasterio.errors.RasterioError: If there are issues with raster operations
    """
    
    try:
        logging.info(f"Starting mesh generation from {las_path}")

        pipeline_json = {
            "pipeline": [
                las_path,
                {
                    "type": "filters.reprojection",
                    "in_srs": "EPSG:25832",
                    "out_srs": "EPSG:25832"
                },
                {
                    "type": "writers.gdal",
                    "filename": raw_mesh,
                    "resolution": 1.0,
                    "radius": 2.0,
                    "output_type": "idw",
                    "nodata": -9999,
                }
            ]
        }

        logging.info("Executing PDAL pipeline...")
        p = pdal.Pipeline(json.dumps(pipeline_json))
        if not p.execute():
            raise RuntimeError("PDAL pipeline execution failed")

        time.sleep(15)
        logging.info("Reading raw mesh...")
        
        with rasterio.open(raw_mesh) as src:
            dem = src.read(1)
            profile = src.profile
            transform = src.transform

        logging.info(f"Raw mesh shape: {dem.shape}")
        nodata = profile.get("nodata", -9999)

        logging.info("Filling nodata values...")
        mask = (dem == nodata)

        filled = ndimage.distance_transform_edt(
            mask,
            return_distances=False,
            return_indices=True
        )

        dem_filled = dem[tuple(filled)]

        logging.info(f"Saving final mesh to {final_mesh}")
        profile.update(dtype=rasterio.float32, nodata=None)

        with rasterio.open(final_mesh, "w", **profile) as dst:
            dst.write(dem_filled.astype(rasterio.float32), 1)
        
        logging.info("Mesh generation complete")

    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}")
        raise
    except RuntimeError as e:
        logging.error(f"Pipeline execution error: {e}")
        raise
    except rasterio.errors.RasterioError as e:
        logging.error(f"Raster operation error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


def main():

    logging.info('Starting script.')
    try:
        
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration: {config}")

        mesh_generation(config['las_path'], config['output_path'], config['output_path_final']) 
        logging.info("DEM generation complete")
        logging.info("Process finished")    
        logging.info(f"Mesh saved: {config['output_path_final']}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()