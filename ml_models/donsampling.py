import laspy
import numpy as np
from typing import List, Tuple
import logging

def process_point_cloud(input_path: str, output_path: str, sample_rate: float = 0.5, chunk_size: int = 1000000):
    """
    Process a large point cloud file by chunks, downsample, and save
    Args:
        input_path: Path to input LAS/LAZ file
        output_path: Path to output LAS/LAZ file
        sample_rate: Fraction of points to keep (0-1)
        chunk_size: Number of points to process at once
    """
    # Open input file and get header
    with laspy.open(input_path) as input_las:
        logging.info(f"Processing file with {input_las.header.point_count} points")
        
        # Create output file with same header
        with laspy.open(output_path, mode="w", header=input_las.header) as output_las:
            # Process chunks
            points_written = 0
            
            for points in input_las.chunk_iterator(chunk_size):
                # Random sampling for this chunk
                n_points = len(points.points)
                n_sample = int(n_points * sample_rate)
                indices = np.random.choice(n_points, n_sample, replace=False)
                
                # Create new point record for sampled points
                sampled_points = laspy.ScaleAwarePointRecord.from_point_record(
                    points.points[indices],
                    input_las.header
                )
                
                # Write sampled points
                output_las.write_points(sampled_points)
                points_written += n_sample
                
                logging.info(f"Processed chunk: {n_points} points -> {n_sample} points")
            
            logging.info(f"Finished processing. Total points written: {points_written}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Downsample point cloud using chunks")
    parser.add_argument("input_path", help="Input LAS/LAZ file path")
    parser.add_argument("output_path", help="Output LAS/LAZ file path")
    parser.add_argument("--sample-rate", type=float, default=0.5,
                       help="Fraction of points to keep (0-1)")
    parser.add_argument("--chunk-size", type=int, default=1000000,
                       help="Number of points to process at once")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    process_point_cloud(
        args.input_path,
        args.output_path,
        args.sample_rate,
        args.chunk_size
    )