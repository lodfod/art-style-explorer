import os
import argparse
import time
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.wikiart_dataloader import WikiArtDataset
from src.preprocessing.edge_detection import extract_edges, extract_edges_gpu, preprocess_image, detect_contours
from src.features.line_features import extract_line_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Process WikiArt dataset and extract features')
    
    # Data arguments
    parser.add_argument('--csv-path', type=str, default='data/wikiart_scraped.csv',
                       help='Path to the WikiArt CSV file')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--cache-dir', type=str, default='data/image_cache',
                       help='Directory to cache downloaded images')
    
    # Processing arguments
    parser.add_argument('--target-size', type=int, default=512,
                       help='Size to resize images to')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Number of images to process at once')
    parser.add_argument('--edge-method', type=str, default='canny',
                       choices=['canny', 'sobel', 'laplacian'],
                       help='Edge detection method')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to process (None for all)')
    parser.add_argument('--extract-features', action='store_true',
                       help='Extract line features from edges')
    
    # GPU acceleration
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration for preprocessing when available')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of worker processes for parallel processing')
    
    # Split arguments
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Proportion of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Proportion of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Proportion of data for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def process_image_with_gpu(image_path, target_size, edge_method):
    """Process an image using GPU acceleration"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # Preprocess (resize and convert to grayscale)
        gray = preprocess_image(img, target_size)
        
        # Use GPU edge detection
        edges = extract_edges_gpu(gray, method=edge_method)
        
        return gray, edges
    except Exception as e:
        logger.warning(f"GPU processing failed: {e}")
        return None, None


def main():
    """Main function to process the WikiArt dataset"""
    args = parse_args()
    
    # Check GPU availability
    use_gpu = args.use_gpu and torch.cuda.is_available() and cv2.cuda.getCudaEnabledDeviceCount() > 0
    if args.use_gpu:
        if use_gpu:
            logger.info(f"Using GPU acceleration with {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
        else:
            logger.info("GPU acceleration requested but no compatible GPU is available. Using CPU.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log the configuration
    logger.info(f"Processing WikiArt dataset with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load the dataset
    dataset = WikiArtDataset(args.csv_path, args.cache_dir)
    
    # Get subsamples if requested
    data = dataset.data
    if args.sample_size is not None:
        # Sample while preserving class distribution
        data = data.groupby('Style', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(args.sample_size * len(x) / len(data)))))
        )
    
    # Split the dataset
    train_df, val_df, test_df = dataset.split_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Process each split
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    processed_data = {}
    
    # Define a custom image processing function that uses GPU if available
    def process_image_batch(batch_df):
        results = []
        for _, row in batch_df.iterrows():
            try:
                # Download image
                img_path = dataset.download_image(row['Link'], row['Artwork'])
                if img_path is None:
                    continue
                
                # Process image (with GPU if available)
                if use_gpu:
                    preprocessed, edges = process_image_with_gpu(
                        img_path, 
                        target_size=(args.target_size, args.target_size),
                        edge_method=args.edge_method
                    )
                    
                    # Fall back to CPU if GPU processing failed
                    if preprocessed is None:
                        from src.preprocessing.edge_detection import process_artwork
                        preprocessed, edges = process_artwork(
                            img_path, 
                            target_size=(args.target_size, args.target_size),
                            edge_method=args.edge_method
                        )
                else:
                    from src.preprocessing.edge_detection import process_artwork
                    preprocessed, edges = process_artwork(
                        img_path, 
                        target_size=(args.target_size, args.target_size),
                        edge_method=args.edge_method
                    )
                
                # Extract features if requested
                features = None
                if args.extract_features:
                    contours = detect_contours(edges)
                    features = extract_line_features(edges, contours)
                
                # Create processed item
                style_name = row['Style']
                style_id = dataset.style_to_id[style_name]
                
                processed_item = {
                    'id': row.name,
                    'style': style_name,
                    'style_id': style_id,
                    'original_path': img_path,
                    'preprocessed': preprocessed,
                    'edges': edges,
                    'features': features
                }
                
                results.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing image from {row['Link']}: {e}")
        
        return results
    
    for split_name, split_df in splits.items():
        logger.info(f"Processing {split_name} split with {len(split_df)} samples")
        
        split_output_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Save split metadata
        split_df.to_csv(os.path.join(args.output_dir, f"{split_name}_metadata_raw.csv"), index=False)
        
        # Process images
        start_time = time.time()
        split_processed = []
        
        # Process in batches
        for i in tqdm(range(0, len(split_df), args.batch_size), desc=f"Processing {split_name}"):
            batch_df = split_df.iloc[i:i+args.batch_size]
            batch_results = process_image_batch(batch_df)
            split_processed.extend(batch_results)
        
        elapsed = time.time() - start_time
        logger.info(f"Processed {len(split_processed)} images in {elapsed:.2f} seconds")
        
        # Save processed data
        dataset.save_processed_data(split_processed, split_output_dir)
        
        processed_data[split_name] = split_processed
    
    # Save style mappings to the main output directory
    style_mapping = {
        'style_to_id': dataset.style_to_id,
        'id_to_style': {str(k): v for k, v in dataset.id_to_style.items()}
    }
    
    import json
    with open(os.path.join(args.output_dir, 'style_mapping.json'), 'w') as f:
        json.dump(style_mapping, f, indent=2)
    
    logger.info(f"Processing complete. Results saved to {args.output_dir}")
    logger.info(f"Total processed: train={len(processed_data['train'])}, "
               f"val={len(processed_data['val'])}, test={len(processed_data['test'])}")


if __name__ == "__main__":
    main() 