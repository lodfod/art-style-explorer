#!/usr/bin/env python3
"""
Script to filter WikiArt dataset by specified art styles and download images.
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

# List of art styles to filter
TARGET_STYLES = [
    "High-Renaissance",
    "Early-Renaissance",
    "Baroque",
    "Impressionism",
    "Post-Impressionism",
    "Expressionism",
    "Neoclassicism",
    "Classicism",
    "Cubism",
    "Surrealism",
    "Abstract-Art",
    "Pop-Art",
    "Art-Nouveau-(Modern)"
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Filter WikiArt dataset and download images')
    parser.add_argument('--csv-path', type=str, required=True, 
                        help='Path to the WikiArt CSV file')
    parser.add_argument('--output-dir', type=str, default='data/filtered',
                        help='Directory to save filtered data and images')
    parser.add_argument('--max-per-style', type=int, default=300,
                        help='Maximum number of images to download per style')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for parallel downloading')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout for image download requests in seconds')
    parser.add_argument('--retry', type=int, default=3,
                        help='Number of retries for failed downloads')
    return parser.parse_args()

def download_image(row, output_dir, timeout=10, max_retries=3):
    """Download an image from a URL and save it to disk."""
    style, artwork, artist, date, url = row
    
    # Create a safe filename from artwork name
    safe_artwork = "".join([c if c.isalnum() else "_" for c in artwork])
    filename = f"{safe_artwork}_{artist.replace(' ', '_')}.jpg"
    
    # Create style directory if it doesn't exist
    style_dir = os.path.join(output_dir, style)
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Filter by target styles
    filtered_df = df[df['Style'].isin(TARGET_STYLES)].copy()
    print(f"Found {len(filtered_df)} artworks across {len(filtered_df['Style'].unique())} target styles")
    
    # Limit to max_per_style per style
    style_counts = filtered_df['Style'].value_counts()
    limited_dfs = []
    
    for style in TARGET_STYLES:
        if style in style_counts:
            style_df = filtered_df[filtered_df['Style'] == style]
            count = min(args.max_per_style, len(style_df))
            print(f"{style}: {count} of {len(style_df)} artworks")
            
            # Randomly sample if we need to limit
            if count < len(style_df):
                style_df = style_df.sample(count, random_state=42)
            
            limited_dfs.append(style_df)
        else:
            print(f"{style}: No artworks found")
    
    # Combine limited dataframes
    limited_df = pd.concat(limited_dfs, ignore_index=True)
    print(f"Selected {len(limited_df)} artworks for downloading")
    
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
                download_image, row, args.output_dir, args.timeout, args.retry
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

if __name__ == "__main__":
    main()
