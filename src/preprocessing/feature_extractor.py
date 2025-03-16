#!/usr/bin/env python3
"""
Feature extraction module for artwork images.
Extracts color, texture, edge, shape, and curvature features from images.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from skimage import feature, color, exposure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import find_contours
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ProcessPoolExecutor

class ArtworkFeatureExtractor:
    """Class for extracting features from artwork images."""
    
    def __init__(self, image_path=None):
        """Initialize the feature extractor."""
        self.image_path = image_path
        self.image = None
        self.features = {}
        
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Load an image from disk."""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self
    
    def extract_all_features(self):
        """Extract all features from the image."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Extract all features
        self.extract_color_features()
        self.extract_texture_features()
        self.extract_edge_features()
        self.extract_curvature_features()
        
        # Combine all features into a single vector
        self.combine_features()
        
        return self.features
    
    def extract_color_features(self):
        """Extract color-based features from the image."""
        # 1. Color Histograms (RGB and HSV)
        # RGB Histogram
        hist_r = cv2.calcHist([self.image], [0], None, [64], [0, 256])
        hist_g = cv2.calcHist([self.image], [1], None, [64], [0, 256])
        hist_b = cv2.calcHist([self.image], [2], None, [64], [0, 256])
        
        # Normalize histograms
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        
        # HSV Histogram
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [36], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # 2. Color Moments
        # Calculate mean, standard deviation, and skewness for each channel
        color_moments = []
        for i in range(3):  # For each channel in RGB
            channel = self.image[:, :, i]
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = np.mean(((channel - mean) / (std + 1e-10)) ** 3)
            color_moments.extend([mean, std, skewness])
        
        # For each channel in HSV
        for i in range(3):
            channel = hsv_image[:, :, i]
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = np.mean(((channel - mean) / (std + 1e-10)) ** 3)
            color_moments.extend([mean, std, skewness])
        
        # Store color features
        self.features['color_hist_rgb'] = np.concatenate([hist_r, hist_g, hist_b])
        self.features['color_hist_hsv'] = np.concatenate([hist_h, hist_s, hist_v])
        self.features['color_moments'] = np.array(color_moments)
        
        return self
    
    def extract_texture_features(self):
        """Extract texture features from the image."""
        # Convert to grayscale for texture analysis
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # 1. Local Binary Patterns (LBP)
        # Parameters for LBP
        radius = 3
        n_points = 8 * radius
        
        # Compute LBP
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        
        # Compute histogram of LBP
        n_bins = n_points + 2  # uniform LBP has n_points + 2 distinct output values
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # 2. Haralick Features (from GLCM)
        # Quantize the image to reduce computation
        gray_image_8bit = (gray_image * 255).astype(np.uint8)
        bins = 32
        gray_image_quantized = np.floor(gray_image_8bit / (256 / bins)).astype(np.uint8)
        
        # Compute GLCM
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_image_quantized, distances, angles, bins, symmetric=True, normed=True)
        
        # Extract Haralick features
        haralick_features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            haralick_features.append(graycoprops(glcm, prop).flatten())
        
        haralick_features = np.concatenate(haralick_features)
        
        # 3. Gabor Filter Responses
        # Define Gabor filter parameters
        num_scales = 4
        num_orientations = 8
        gabor_features = []
        
        for scale in range(1, num_scales + 1):
            for orientation in range(num_orientations):
                theta = orientation * np.pi / num_orientations
                frequency = 0.1 / scale
                
                # Apply Gabor filter
                real, imag = cv2.getGaborKernel(
                    (31, 31), sigma=scale, theta=theta, lambd=1.0/frequency, 
                    gamma=0.5, psi=0, ktype=cv2.CV_32F
                ), None
                
                filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, real)
                
                # Compute mean and variance of filter response
                mean = np.mean(filtered)
                var = np.var(filtered)
                gabor_features.extend([mean, var])
        
        # Store texture features
        self.features['lbp_hist'] = hist
        self.features['haralick'] = haralick_features
        self.features['gabor'] = np.array(gabor_features)
        
        return self
    
    def extract_edge_features(self):
        """Extract edge and shape features from the image."""
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # 1. Edge Detection using Canny
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Apply Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        # Compute edge density (ratio of edge pixels to total pixels)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 2. Histogram of Oriented Gradients (HOG)
        # Resize image for HOG (to ensure consistent feature size)
        resized = cv2.resize(gray_image, (128, 128))
        
        # Compute HOG features
        hog_features, hog_image = feature.hog(
            resized, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True, 
            block_norm='L2-Hys'
        )
        
        # Store edge features
        self.features['edge_density'] = np.array([edge_density])
        self.features['hog'] = hog_features
        
        return self
    
    def extract_curvature_features(self):
        """Extract curvature and contour features from the image."""
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Apply Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Filter out very small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]
        
        # Initialize curvature features
        num_contours = len(contours)
        total_contour_length = 0
        curvature_stats = []
        
        # Process each contour
        for contour in contours:
            # Skip if contour is too small
            if len(contour) < 5:
                continue
                
            # Calculate contour length
            contour_length = cv2.arcLength(contour, closed=True)
            total_contour_length += contour_length
            
            # Approximate contour with polygon
            epsilon = 0.02 * contour_length
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            
            # Calculate curvature at each point
            curvature_values = []
            for i in range(len(contour)):
                # Get three consecutive points
                prev_idx = (i - 1) % len(contour)
                next_idx = (i + 1) % len(contour)
                
                # Extract points
                prev = contour[prev_idx][0]
                curr = contour[i][0]
                next_pt = contour[next_idx][0]
                
                # Calculate vectors
                v1 = prev - curr
                v2 = next_pt - curr
                
                # Calculate angle between vectors
                dot = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                # Avoid division by zero
                if norm_v1 * norm_v2 == 0:
                    continue
                    
                cos_angle = dot / (norm_v1 * norm_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure value is in valid range
                angle = np.arccos(cos_angle)
                
                # Curvature is inversely proportional to radius
                curvature = angle / (0.5 * (norm_v1 + norm_v2))
                curvature_values.append(curvature)
            
            # Calculate statistics of curvature values
            if curvature_values:
                curvature_stats.extend([
                    np.mean(curvature_values),
                    np.std(curvature_values),
                    np.max(curvature_values),
                    np.percentile(curvature_values, 75)
                ])
        
        # If no valid contours were found, use zeros
        if not curvature_stats:
            curvature_stats = [0, 0, 0, 0]
        
        # Normalize and pad/truncate to ensure consistent feature length
        target_length = 20  # Set a fixed length for curvature stats
        if len(curvature_stats) > target_length:
            curvature_stats = curvature_stats[:target_length]
        else:
            curvature_stats.extend([0] * (target_length - len(curvature_stats)))
        
        # Additional contour features
        contour_features = [
            num_contours,
            total_contour_length,
            total_contour_length / max(num_contours, 1)  # Average contour length
        ]
        
        # Store curvature features
        self.features['contour_features'] = np.array(contour_features)
        self.features['curvature_stats'] = np.array(curvature_stats)
        
        return self
    
    def combine_features(self):
        """Combine all extracted features into a single vector."""
        # Concatenate all feature vectors
        combined = []
        for key in sorted(self.features.keys()):
            combined.append(self.features[key])
        
        self.features['combined'] = np.concatenate(combined)
        return self

def extract_features_from_image(image_path):
    """Extract features from a single image."""
    try:
        extractor = ArtworkFeatureExtractor()
        extractor.load_image(image_path)
        features = extractor.extract_all_features()
        return image_path, features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, None

def process_directory(input_dir, output_dir, num_workers=4):
    """Process all images in a directory and extract features."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images to process")
    
    # Extract features in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(extract_features_from_image, path) for path in image_paths]
        
        for future in tqdm(futures, total=len(image_paths), desc="Extracting features"):
            path, features = future.result()
            if features is not None:
                results[path] = features
    
    print(f"Successfully processed {len(results)} images")
    
    # Save features to disk
    output_path = os.path.join(output_dir, 'artwork_features.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved features to {output_path}")
    
    # Create a summary CSV with feature dimensions
    if results:
        first_result = next(iter(results.values()))
        feature_dims = {key: len(val) for key, val in first_result.items()}
        
        summary_df = pd.DataFrame([{
            'feature_name': key,
            'dimension': dim,
            'description': get_feature_description(key)
        } for key, dim in feature_dims.items()])
        
        summary_path = os.path.join(output_dir, 'feature_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved feature summary to {summary_path}")
    
    return results

def get_feature_description(feature_name):
    """Get a description for each feature type."""
    descriptions = {
        'color_hist_rgb': 'RGB color histogram (64 bins per channel)',
        'color_hist_hsv': 'HSV color histogram (H: 36 bins, S/V: 32 bins each)',
        'color_moments': 'Statistical moments (mean, std, skewness) for RGB and HSV channels',
        'lbp_hist': 'Local Binary Pattern histogram for texture analysis',
        'haralick': 'Haralick texture features from Gray-Level Co-occurrence Matrix',
        'gabor': 'Gabor filter responses at multiple scales and orientations',
        'edge_density': 'Ratio of edge pixels to total pixels',
        'hog': 'Histogram of Oriented Gradients for shape analysis',
        'contour_features': 'Number of contours, total and average contour length',
        'curvature_stats': 'Statistical measures of curvature along contours',
        'combined': 'All features combined into a single vector'
    }
    return descriptions.get(feature_name, 'Unknown feature')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract features from artwork images')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing artwork images')
    parser.add_argument('--output-dir', type=str, default='data/features',
                        help='Directory to save extracted features')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for parallel extraction')
    return parser.parse_args()

def main():
    """Main function to extract features from artwork images."""
    args = parse_args()
    process_directory(args.input_dir, args.output_dir, args.num_workers)

if __name__ == "__main__":
    main() 