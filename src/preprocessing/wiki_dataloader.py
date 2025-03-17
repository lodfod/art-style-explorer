#!/usr/bin/env python3
"""
Script to filter WikiArt dataset by specified art styles and download images,
organizing them into main style classifications.
"""

import os
import argparse
import pandas as pd
import requests
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import time
import random
import numpy as np

# Define style classifications and their constituent styles
STYLE_CLASSIFICATIONS = {
       "Impressionist_and_Post_Impressionist": [
        "Impressionism", 
        "Post-Impressionism"
    ],
    "Graphic_and_Pattern_Based": [
        "Ukiyo-e", 
        "Pop-Art", 
        "Art-Nouveau-(Modern)"
    ],
    "Geometric_and_Abstract": [
        "Cubism", 
        "Abstract-Art"
    ],
    "Expressive_and_Emotional": [
        "Expressionism", 
        "Surrealism"
    ],
    "Figurative_Traditional": [
        "Early-Renaissance", 
        "High-Renaissance", 
        "Neoclassicism", 
        "Classicism"
    ],
    "Decorative_and_Ornamental": [
        "Romanesque", 
        "Baroque"
    ]
}

# Flatten the list of styles for filtering
TARGET_STYLES = []
for styles in STYLE_CLASSIFICATIONS.values():
    TARGET_STYLES.extend(styles)

# Create a mapping from individual style to its classification
STYLE_TO_CLASSIFICATION = {}
for classification, styles in STYLE_CLASSIFICATIONS.items():
    for style in styles:
        STYLE_TO_CLASSIFICATION[style] = classification

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Filter WikiArt dataset and download images')
    parser.add_argument('--csv-path', type=str, required=True, 
                        help='Path to the WikiArt CSV file')
    parser.add_argument('--output-dir', type=str, default='data/filtered',
                        help='Directory to save filtered data and images')
    parser.add_argument('--max-per-style', type=int, default=300,
                        help='Maximum number of images to download per style')
    parser.add_argument('--max-per-classification', type=int, default=500,
                        help='Maximum number of images to download per classification (overrides max-per-style)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for parallel downloading')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout for image download requests in seconds')
    parser.add_argument('--retry', type=int, default=3,
                        help='Number of retries for failed downloads')
    parser.add_argument('--hierarchical', action='store_true',
                        help='Organize images in a hierarchical directory structure by classification')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def download_image(row, output_dir, hierarchical=False, timeout=10, max_retries=3):
    """Download an image from a URL and save it to disk."""
    style, artwork, artist, date, url = row
    
    # Create a safe filename from artwork name
    safe_artwork = "".join([c if c.isalnum() else "_" for c in artwork])
    filename = f"{safe_artwork}_{artist.replace(' ', '_')}.jpg"
    
    # Determine the directory structure based on hierarchical flag
    if hierarchical:
        # Get the classification for this style
        classification = STYLE_TO_CLASSIFICATION.get(style, "Other")
        # Create classification/style directory structure
        style_dir = os.path.join(output_dir, classification, style)
    else:
        # Just use the style as the directory
        style_dir = os.path.join(output_dir, style)
    
    # Create directory if it doesn't exist
    os.makedirs(style_dir, exist_ok=True)
    
    output_path = os.path.join(style_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        return True, row
    
    # Try to download the image
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True, row
            else:
                time.sleep(1)  # Wait before retrying
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retrying
            else:
                return False, row
    
    return False, row

def main():
    """Main function to filter dataset and download images."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Filter by target styles
    filtered_df = df[df['Style'].isin(TARGET_STYLES)].copy()
    print(f"Found {len(filtered_df)} artworks across {len(filtered_df['Style'].unique())} target styles")
    
    # Print style counts by classification
    print("\nArtworks by classification:")
    for classification, styles in STYLE_CLASSIFICATIONS.items():
        class_df = filtered_df[filtered_df['Style'].isin(styles)]
        print(f"{classification}: {len(class_df)} artworks")
        for style in styles:
            style_count = len(filtered_df[filtered_df['Style'] == style])
            print(f"  - {style}: {style_count} artworks")
    
    # Limit samples per style or classification
    limited_dfs = []
    
    if args.max_per_classification:
        # Limit by classification first
        for classification, styles in STYLE_CLASSIFICATIONS.items():
            class_df = filtered_df[filtered_df['Style'].isin(styles)]
            count = min(args.max_per_classification, len(class_df))
            print(f"\n{classification}: selecting {count} of {len(class_df)} artworks")
            
            if count < len(class_df):
                # Try to balance styles within the classification
                style_dfs = []
                styles_in_data = [s for s in styles if s in filtered_df['Style'].unique()]
                
                if styles_in_data:
                    # Shuffle the styles to randomize their order
                    random.shuffle(styles_in_data)
                    print(f"  Shuffled styles order: {', '.join(styles_in_data)}")
                    
                    # Calculate target count per style
                    target_per_style = count // len(styles_in_data)
                    remaining = count % len(styles_in_data)
                    
                    for style in styles_in_data:
                        style_df = class_df[class_df['Style'] == style]
                        style_target = target_per_style + (1 if remaining > 0 else 0)
                        remaining -= 1 if remaining > 0 else 0
                        
                        style_count = min(style_target, len(style_df))
                        print(f"  - {style}: {style_count} artworks")
                        
                        if style_count < len(style_df):
                            style_df = style_df.sample(style_count, random_state=args.seed)
                        
                        style_dfs.append(style_df)
                    
                    # Combine all style dataframes for this classification
                    class_limited_df = pd.concat(style_dfs, ignore_index=True)
                    
                    # Shuffle the combined dataframe to mix styles
                    class_limited_df = class_limited_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
                    limited_dfs.append(class_limited_df)
            else:
                # Shuffle the dataframe to mix styles
                class_df = class_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
                limited_dfs.append(class_df)
    else:
        # Limit by individual style
        for style in TARGET_STYLES:
            if style in filtered_df['Style'].unique():
                style_df = filtered_df[filtered_df['Style'] == style]
                count = min(args.max_per_style, len(style_df))
                print(f"{style}: {count} of {len(style_df)} artworks")
                
                # Randomly sample if we need to limit
                if count < len(style_df):
                    style_df = style_df.sample(count, random_state=args.seed)
                
                limited_dfs.append(style_df)
            else:
                print(f"{style}: No artworks found")
    
    # Combine limited dataframes
    limited_df = pd.concat(limited_dfs, ignore_index=True)
    print(f"\nSelected {len(limited_df)} artworks for downloading")
    
    # Add classification column to the dataframe
    limited_df['Classification'] = limited_df['Style'].map(STYLE_TO_CLASSIFICATION)
    
    # Save filtered metadata
    filtered_csv_path = os.path.join(args.output_dir, 'filtered_wikiart.csv')
    limited_df.to_csv(filtered_csv_path, index=False)
    print(f"Saved filtered metadata to {filtered_csv_path}")
    
    # Download images
    print(f"Downloading images to {args.output_dir}...")
    
    # Create a list of rows to download
    download_rows = [
        (row['Style'], row['Artwork'], row['Artist'], row['Date'], row['Link']) 
        for _, row in limited_df.iterrows()
    ]
    
    # Download images in parallel
    successful_rows = []
    failed_rows = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_row = {
            executor.submit(
                download_image, row, args.output_dir, args.hierarchical, args.timeout, args.retry
            ): row for row in download_rows
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_row), 
                          total=len(download_rows), 
                          desc="Downloading images"):
            success, row = future.result()
            if success:
                successful_rows.append(row)
            else:
                failed_rows.append(row)
    
    # Create success and failure dataframes
    success_df = pd.DataFrame(successful_rows, columns=['Style', 'Artwork', 'Artist', 'Date', 'Link'])
    failed_df = pd.DataFrame(failed_rows, columns=['Style', 'Artwork', 'Artist', 'Date', 'Link'])
    
    # Add classification column to the success dataframe
    success_df['Classification'] = success_df['Style'].map(STYLE_TO_CLASSIFICATION)
    
    # Save success and failure metadata
    success_csv_path = os.path.join(args.output_dir, 'downloaded_wikiart.csv')
    failed_csv_path = os.path.join(args.output_dir, 'failed_downloads.csv')
    
    success_df.to_csv(success_csv_path, index=False)
    print(f"Successfully downloaded {len(success_df)} images")
    print(f"Saved successful download metadata to {success_csv_path}")
    
    if len(failed_df) > 0:
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"Failed to download {len(failed_df)} images")
        print(f"Saved failed download metadata to {failed_csv_path}")
    
    # Print summary by classification
    print("\nDownloaded images by classification:")
    for classification in STYLE_CLASSIFICATIONS:
        class_count = len(success_df[success_df['Classification'] == classification])
        print(f"{classification}: {class_count} images")
        
        # Print breakdown by style within classification
        styles = STYLE_CLASSIFICATIONS[classification]
        for style in styles:
            style_count = len(success_df[success_df['Style'] == style])
            if style_count > 0:
                print(f"  - {style}: {style_count} images")

if __name__ == "__main__":
    main()
