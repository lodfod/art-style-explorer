#!/usr/bin/env python3
"""
Script to augment extracted feature vectors for improving model training.
This augmentation happens after feature extraction but before model training.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from collections import Counter

# List of target art styles
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

def extract_style_from_path(path):
    """Extract style name from image path."""
    # Assuming the directory structure is .../style_name/image_name.jpg
    return Path(path).parts[-2]

def load_features(features_path):
    """Load extracted features from a pickle file."""
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    return features

def get_feature_groups(feature_dict):
    """Group features by type for targeted augmentation."""
    groups = {}
    
    # Color features
    color_features = []
    if 'color_hist_rgb' in feature_dict:
        color_features.append(('color_hist_rgb', feature_dict['color_hist_rgb']))
    if 'color_hist_hsv' in feature_dict:
        color_features.append(('color_hist_hsv', feature_dict['color_hist_hsv']))
    if 'color_moments' in feature_dict:
        color_features.append(('color_moments', feature_dict['color_moments']))
    
    # Texture features
    texture_features = []
    if 'lbp_hist' in feature_dict:
        texture_features.append(('lbp_hist', feature_dict['lbp_hist']))
    if 'haralick' in feature_dict:
        texture_features.append(('haralick', feature_dict['haralick']))
    if 'gabor' in feature_dict:
        texture_features.append(('gabor', feature_dict['gabor']))
    
    # Edge and shape features
    edge_features = []
    if 'edge_density' in feature_dict:
        edge_features.append(('edge_density', feature_dict['edge_density']))
    if 'hog' in feature_dict:
        edge_features.append(('hog', feature_dict['hog']))
    
    # Contour features
    contour_features = []
    if 'contour_features' in feature_dict:
        contour_features.append(('contour_features', feature_dict['contour_features']))
    if 'curvature_stats' in feature_dict:
        contour_features.append(('curvature_stats', feature_dict['curvature_stats']))
    
    groups['color'] = color_features
    groups['texture'] = texture_features
    groups['edge'] = edge_features
    groups['contour'] = contour_features
    
    return groups

def augment_color_features(feature_name, feature_vector, intensity=0.1):
    """Augment color-based features."""
    if feature_name == 'color_hist_rgb' or feature_name == 'color_hist_hsv':
        # For histograms, we can shift bins slightly or adjust overall intensity
        augmented = feature_vector.copy()
        
        # Random shift (move some values to adjacent bins)
        shift_indices = np.random.choice(len(augmented), size=int(len(augmented) * 0.2), replace=False)
        for idx in shift_indices:
            if idx > 0 and idx < len(augmented) - 1:
                direction = np.random.choice([-1, 1])
                amount = augmented[idx] * intensity
                augmented[idx] -= amount
                augmented[idx + direction] += amount
        
        # Ensure non-negativity and normalization
        augmented = np.maximum(0, augmented)
        if np.sum(augmented) > 0:
            augmented = augmented / np.sum(augmented) * np.sum(feature_vector)
        
        return augmented
    
    elif feature_name == 'color_moments':
        # For color moments, we can slightly adjust the values
        augmented = feature_vector.copy()
        noise = np.random.normal(0, intensity, size=augmented.shape)
        augmented = augmented * (1 + noise)
        return augmented
    
    return feature_vector.copy()

def augment_texture_features(feature_name, feature_vector, intensity=0.1):
    """Augment texture-based features."""
    augmented = feature_vector.copy()
    
    if feature_name == 'lbp_hist':
        # Similar to color histograms, shift some bins
        shift_indices = np.random.choice(len(augmented), size=int(len(augmented) * 0.2), replace=False)
        for idx in shift_indices:
            if idx > 0 and idx < len(augmented) - 1:
                direction = np.random.choice([-1, 1])
                amount = augmented[idx] * intensity
                augmented[idx] -= amount
                augmented[idx + direction] += amount
        
        # Ensure non-negativity and normalization
        augmented = np.maximum(0, augmented)
        if np.sum(augmented) > 0:
            augmented = augmented / np.sum(augmented) * np.sum(feature_vector)
    
    elif feature_name in ['haralick', 'gabor']:
        # Add small random noise
        noise = np.random.normal(0, intensity, size=augmented.shape)
        augmented = augmented * (1 + noise)
    
    return augmented

def augment_edge_features(feature_name, feature_vector, intensity=0.1):
    """Augment edge and shape features."""
    augmented = feature_vector.copy()
    
    if feature_name == 'edge_density':
        # Slightly adjust edge density
        noise = np.random.normal(0, intensity)
        augmented = augmented * (1 + noise)
        # Ensure it stays in valid range [0, 1]
        augmented = np.clip(augmented, 0, 1)
    
    elif feature_name == 'hog':
        # For HOG features, add small random noise
        noise = np.random.normal(0, intensity, size=augmented.shape)
        augmented = augmented * (1 + noise)
        # HOG features should be normalized
        if np.sum(augmented) > 0:
            augmented = augmented / np.linalg.norm(augmented) * np.linalg.norm(feature_vector)
    
    return augmented

def augment_contour_features(feature_name, feature_vector, intensity=0.1):
    """Augment contour and curvature features."""
    augmented = feature_vector.copy()
    
    # Add small random noise
    noise = np.random.normal(0, intensity, size=augmented.shape)
    augmented = augmented * (1 + noise)
    
    # Ensure non-negativity for features that should be positive
    if feature_name == 'contour_features':
        augmented = np.maximum(0, augmented)
    
    return augmented

def augment_feature_vector(feature_dict, augmentation_strategy='all', intensity=0.1):
    """Augment a single feature vector based on the specified strategy."""
    augmented_dict = {}
    feature_groups = get_feature_groups(feature_dict)
    
    # Determine which feature groups to augment
    groups_to_augment = []
    if augmentation_strategy == 'all':
        groups_to_augment = ['color', 'texture', 'edge', 'contour']
    elif augmentation_strategy == 'color_only':
        groups_to_augment = ['color']
    elif augmentation_strategy == 'texture_only':
        groups_to_augment = ['texture']
    elif augmentation_strategy == 'edge_only':
        groups_to_augment = ['edge']
    elif augmentation_strategy == 'contour_only':
        groups_to_augment = ['contour']
    elif augmentation_strategy == 'color_texture':
        groups_to_augment = ['color', 'texture']
    elif augmentation_strategy == 'edge_contour':
        groups_to_augment = ['edge', 'contour']
    
    # Apply augmentation to selected feature groups
    for group in groups_to_augment:
        for feature_name, feature_vector in feature_groups[group]:
            if group == 'color':
                augmented_dict[feature_name] = augment_color_features(feature_name, feature_vector, intensity)
            elif group == 'texture':
                augmented_dict[feature_name] = augment_texture_features(feature_name, feature_vector, intensity)
            elif group == 'edge':
                augmented_dict[feature_name] = augment_edge_features(feature_name, feature_vector, intensity)
            elif group == 'contour':
                augmented_dict[feature_name] = augment_contour_features(feature_name, feature_vector, intensity)
    
    # Copy non-augmented features
    for feature_name, feature_vector in feature_dict.items():
        if feature_name not in augmented_dict:
            augmented_dict[feature_name] = feature_vector.copy()
    
    # Recombine features if 'combined' was in the original
    if 'combined' in feature_dict:
        combined = []
        for key in sorted(augmented_dict.keys()):
            if key != 'combined':
                combined.append(augmented_dict[key])
        augmented_dict['combined'] = np.concatenate(combined)
    
    return augmented_dict

def generate_synthetic_samples(features, styles, num_samples_per_style=None, augmentation_strategy='all', intensity=0.1):
    """Generate synthetic samples for each style to balance the dataset."""
    # Count samples per style
    style_counts = Counter(styles)
    
    # Determine target number of samples per style
    if num_samples_per_style is None:
        num_samples_per_style = max(style_counts.values())
    
    # Group features by style
    style_to_features = {}
    for i, style in enumerate(styles):
        if style not in style_to_features:
            style_to_features[style] = []
        style_to_features[style].append(i)
    
    # Generate synthetic samples
    synthetic_features = []
    synthetic_styles = []
    
    for style, indices in tqdm(style_to_features.items(), desc="Generating synthetic samples"):
        current_count = len(indices)
        if current_count >= num_samples_per_style:
            continue
        
        # Number of synthetic samples needed
        num_synthetic = num_samples_per_style - current_count
        
        # Generate synthetic samples
        for _ in range(num_synthetic):
            # Randomly select a sample to augment
            idx = np.random.choice(indices)
            feature_dict = features[idx]
            
            # Augment the feature vector
            augmented_dict = augment_feature_vector(feature_dict, augmentation_strategy, intensity)
            
            # Add to synthetic samples
            synthetic_features.append(augmented_dict)
            synthetic_styles.append(style)
    
    return synthetic_features, synthetic_styles

def augment_dataset(features_path, output_path, augmentation_strategy='all', intensity=0.1, 
                    balance_classes=True, num_samples_per_style=None, feature_type='combined'):
    """Augment the dataset and save the augmented features."""
    # Load features
    features = load_features(features_path)
    
    # Extract styles from paths
    paths = list(features.keys())
    styles = [extract_style_from_path(path) for path in paths]
    
    # Filter to include only target styles
    valid_indices = [i for i, style in enumerate(styles) if style in TARGET_STYLES]
    valid_paths = [paths[i] for i in valid_indices]
    valid_styles = [styles[i] for i in valid_indices]
    
    # Create feature dictionaries list
    feature_dicts = [features[path] for path in valid_paths]
    
    # Generate synthetic samples if balancing classes
    if balance_classes:
        print("Balancing classes with synthetic samples...")
        synthetic_features, synthetic_styles = generate_synthetic_samples(
            feature_dicts, valid_styles, num_samples_per_style, augmentation_strategy, intensity
        )
        
        # Add synthetic samples to the dataset
        augmented_feature_dicts = feature_dicts + synthetic_features
        augmented_styles = valid_styles + synthetic_styles
        augmented_paths = valid_paths + [f"synthetic_{i}" for i in range(len(synthetic_features))]
    else:
        # Just augment existing samples
        print("Augmenting existing samples...")
        augmented_feature_dicts = []
        for feature_dict in tqdm(feature_dicts, desc="Augmenting features"):
            augmented_dict = augment_feature_vector(feature_dict, augmentation_strategy, intensity)
            augmented_feature_dicts.append(augmented_dict)
        
        augmented_styles = valid_styles
        augmented_paths = valid_paths
    
    # Create augmented features dictionary
    augmented_features = {}
    for i, path in enumerate(augmented_paths):
        augmented_features[path] = augmented_feature_dicts[i]
    
    # Save augmented features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(augmented_features, f)
    
    # Print statistics
    original_counts = Counter(valid_styles)
    augmented_counts = Counter(augmented_styles)
    
    print("\nDataset Statistics:")
    print(f"Original dataset: {len(valid_paths)} samples")
    print(f"Augmented dataset: {len(augmented_paths)} samples")
    print("\nSamples per style:")
    
    for style in sorted(TARGET_STYLES):
        original = original_counts.get(style, 0)
        augmented = augmented_counts.get(style, 0)
        print(f"  {style}: {original} → {augmented}")
    
    return augmented_features

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Augment extracted feature vectors')
    parser.add_argument('--features-path', type=str, required=True,
                        help='Path to the artwork features pickle file')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the augmented features')
    parser.add_argument('--augmentation-strategy', type=str, default='all',
                        choices=['all', 'color_only', 'texture_only', 'edge_only', 
                                'contour_only', 'color_texture', 'edge_contour'],
                        help='Strategy for feature augmentation')
    parser.add_argument('--intensity', type=float, default=0.1,
                        help='Intensity of augmentation (0.0-1.0)')
    parser.add_argument('--balance-classes', action='store_true',
                        help='Balance classes by generating synthetic samples')
    parser.add_argument('--num-samples-per-style', type=int, default=None,
                        help='Target number of samples per style (default: max of original counts)')
    return parser.parse_args()

def main():
    """Main function to augment the dataset."""
    args = parse_args()
    
    print(f"Loading features from {args.features_path}...")
    
    # Augment dataset
    augmented_features = augment_dataset(
        args.features_path,
        args.output_path,
        args.augmentation_strategy,
        args.intensity,
        args.balance_classes,
        args.num_samples_per_style
    )
    
    print(f"Augmented features saved to {args.output_path}")

if __name__ == "__main__":
    main() 