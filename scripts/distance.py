import laspy
import rasterio
import numpy as np
import logging
import yaml
import sys
import os
import argparse

def get_log_filename(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return os.path.join('log', f'{base}.log')


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

def distance_calculation(las_path: str, dem_path: str, output_las: str) -> None:
    """
    Calculate the vertical distance between LAS points and a DEM surface.

    Parameters:
        las_path (str): Path to input LAS file
        dem_path (str): Path to input DEM raster
        output_las (str): Path for output LAS file with distance calculations

    The function:
    1. Reads LAS points and DEM raster
    2. Calculates point locations in DEM grid
    3. Computes vertical distances
    4. Preserves RGB values if present
    5. Saves result as new LAS with added 'distance' field
    """
    try:
        with rasterio.open(dem_path) as src:
            try:
                dem = src.read(1)
                transform = src.transform
                nodata = src.nodata
                inv_transform = ~transform  # inverse affine for xy → row/col

                las = laspy.read(las_path)
                xs = las.x
                ys = las.y
                zs = las.z

                cols, rows = inv_transform * (xs, ys)
                cols = np.floor(cols).astype(int)
                rows = np.floor(rows).astype(int)

                # Keep only points inside DEM
                valid = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
                xs, ys, zs, rows, cols = xs[valid], ys[valid], zs[valid], rows[valid], cols[valid]

                # Keep RGB from original LAS if present
                if hasattr(las, "red"):
                    R = las.red[valid]
                    G = las.green[valid]
                    B = las.blue[valid]
                else:
                    # If no RGB, fill zeros
                    R = np.zeros_like(xs, dtype=np.uint16)
                    G = np.zeros_like(xs, dtype=np.uint16)
                    B = np.zeros_like(xs, dtype=np.uint16)

                dem_zs = dem[rows, cols]

                # Mask nodata if present
                if nodata is not None:
                    mask = dem_zs != nodata
                    xs, ys, zs, dem_zs = xs[mask], ys[mask], zs[mask], dem_zs[mask]

                distances = zs - dem_zs
                distances = np.maximum(distances, 0.0)  # any value <0 becomes 0

                header = laspy.LasHeader(point_format=3, version="1.2")
                las_out = laspy.LasData(header)
                las_out.x = xs
                las_out.y = ys
                las_out.z = zs  # original elevations
                las_out.red = R
                las_out.green = G
                las_out.blue = B

                las_out.add_extra_dim(laspy.ExtraBytesParams(name="distance", type=np.float32))
                las_out.distance = distances.astype(np.float32)

                las_out.write(output_las)

                logging.info(f"Saved LAS with {len(xs)} points")
                logging.info("Z = original elevation, 'distance' = point-to-DEM difference")

            except rasterio.errors.RasterioError as e:
                logging.error(f"Error reading DEM data: {e}")
                raise
            except laspy.errors.LaspyException as e:
                logging.error(f"Error processing LAS data: {e}")
                raise
            except ValueError as e:
                logging.error(f"Error in data processing: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error in processing: {e}")
                raise

    except rasterio.errors.RasterioIOError as e:
        logging.error(f"Error opening DEM file: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def main():
    try:
        args = parse_args()
        config = read_config(args.config_file)
        logging.info(f"Configuration loaded from {args.config_file}")

        las_path = config['input_las']
        dem_path = config['input_dem'] 
        output_las = config['output_las']

        logging.info(f"Processing LAS file: {las_path}")
        logging.info(f"Using DEM file: {dem_path}")
        logging.info(f"Output will be saved to: {output_las}")

        distance_calculation(las_path, dem_path, output_las)
        logging.info("Processing completed successfully")

    except KeyError as e:
        logging.error(f"Missing required configuration key: {e}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()