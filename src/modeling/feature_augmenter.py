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

# Define style classifications and their constituent styles
STYLE_CLASSIFICATIONS = {
    "Classical_And_Renaissance": [
        "Early-Renaissance", 
        "High-Renaissance", 
        "Mannerism", 
        "Neoclassicism", 
        "Classicism"
    ],
    "Medieval_And_Ornamental": [
        "Romanesque", 
        "Baroque"
    ],
    "Impressionist_Movements": [
        "Impressionism", 
        "Post-Impressionism"
    ],
    "Expressionist_And_Surrealist": [
        "Expressionism", 
        "Surrealism"
    ],
    "Abstract_And_Fragmented": [
        "Cubism", 
        "Abstract-Art"
    ],
    "Graphic_Styles": [
        "Ukiyo-e", 
        "Pop-Art", 
        "Art-Nouveau-(Modern)"
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

def extract_style_from_path(path):
    """Extract style name from image path."""
    # Assuming the directory structure is .../style_name/image_name.jpg
    # or .../classification/style_name/image_name.jpg
    parts = Path(path).parts
    for style in TARGET_STYLES:
        if style in parts:
            return style
    return parts[-2]  # Fallback to second-to-last part

def extract_classification_from_path(path):
    """Extract classification from image path or derive it from style."""
    parts = Path(path).parts
    
    # First check if any classification name is in the path
    for classification in STYLE_CLASSIFICATIONS.keys():
        if classification in parts:
            return classification
    
    # If not, extract the style and map to classification
    style = extract_style_from_path(path)
    return STYLE_TO_CLASSIFICATION.get(style, "Unknown")

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

def generate_synthetic_samples(features, styles, classifications=None, num_samples_per_style=None, 
                              num_samples_per_classification=None, augmentation_strategy='all', intensity=0.1):
    """Generate synthetic samples for each style or classification to balance the dataset."""
    # Count samples per style and classification
    style_counts = Counter(styles)
    
    if classifications:
        classification_counts = Counter(classifications)
        
        # Determine if we're balancing by style or classification
        if num_samples_per_classification is not None:
            # Group features by classification
            classification_to_features = {}
            for i, classification in enumerate(classifications):
                if classification not in classification_to_features:
                    classification_to_features[classification] = []
                classification_to_features[classification].append(i)
            
            # Generate synthetic samples by classification
            synthetic_features = []
            synthetic_styles = []
            synthetic_classifications = []
            
            for classification, indices in tqdm(classification_to_features.items(), 
                                              desc="Generating synthetic samples by classification"):
                current_count = len(indices)
                if current_count >= num_samples_per_classification:
                    continue
                
                # Number of synthetic samples needed
                num_synthetic = num_samples_per_classification - current_count
                
                # Get styles in this classification
                class_styles = set([styles[i] for i in indices])
                
                # Generate synthetic samples, trying to balance styles within classification
                style_targets = {}
                total_per_style = num_synthetic // len(class_styles)
                remainder = num_synthetic % len(class_styles)
                
                for style in class_styles:
                    style_targets[style] = total_per_style + (1 if remainder > 0 else 0)
                    remainder -= 1 if remainder > 0 else 0
                
                # Generate samples for each style in the classification
                for style in class_styles:
                    # Get indices for this style
                    style_indices = [i for i in indices if styles[i] == style]
                    
                    # Generate synthetic samples for this style
                    for _ in range(style_targets[style]):
                        # Randomly select a sample to augment
                        idx = np.random.choice(style_indices)
                        feature_dict = features[idx]
                        
                        # Augment the feature vector
                        augmented_dict = augment_feature_vector(feature_dict, augmentation_strategy, intensity)
                        
                        # Add to synthetic samples
                        synthetic_features.append(augmented_dict)
                        synthetic_styles.append(style)
                        synthetic_classifications.append(classification)
            
            return synthetic_features, synthetic_styles, synthetic_classifications
    
    # If we're not balancing by classification or classifications not provided, balance by style
    if num_samples_per_style is None:
        num_samples_per_style = max(style_counts.values())
    
    # Group features by style
    style_to_features = {}
    for i, style in enumerate(styles):
        if style not in style_to_features:
            style_to_features[style] = []
        style_to_features[style].append(i)
    
    # Generate synthetic samples by style
    synthetic_features = []
    synthetic_styles = []
    synthetic_classifications = []
    
    for style, indices in tqdm(style_to_features.items(), desc="Generating synthetic samples by style"):
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
            if classifications:
                synthetic_classifications.append(classifications[idx])
    
    if classifications:
        return synthetic_features, synthetic_styles, synthetic_classifications
    else:
        return synthetic_features, synthetic_styles, None

def augment_dataset(features_path, output_path, augmentation_strategy='all', intensity=0.1, 
                    balance_classes=True, num_samples_per_style=None, num_samples_per_classification=None,
                    use_classifications=False, feature_type='combined'):
    """Augment the dataset and save the augmented features."""
    # Load features
    features = load_features(features_path)
    
    # Extract styles and classifications from paths
    paths = list(features.keys())
    styles = [extract_style_from_path(path) for path in paths]
    
    # Extract classifications if needed
    classifications = None
    if use_classifications:
        classifications = [extract_classification_from_path(path) for path in paths]
    
    # Filter to include only target styles
    valid_indices = [i for i, style in enumerate(styles) if style in TARGET_STYLES]
    valid_paths = [paths[i] for i in valid_indices]
    valid_styles = [styles[i] for i in valid_indices]
    valid_classifications = None
    if classifications:
        valid_classifications = [classifications[i] for i in valid_indices]
    
    # Create feature dictionaries list
    feature_dicts = [features[path] for path in valid_paths]
    
    # Generate synthetic samples if balancing classes
    if balance_classes:
        if use_classifications and num_samples_per_classification is not None:
            print(f"Balancing classes with synthetic samples (target: {num_samples_per_classification} per classification)...")
            synthetic_features, synthetic_styles, synthetic_classifications = generate_synthetic_samples(
                feature_dicts, valid_styles, valid_classifications, 
                num_samples_per_classification=num_samples_per_classification,
                augmentation_strategy=augmentation_strategy, intensity=intensity
            )
        else:
            print(f"Balancing classes with synthetic samples (target: {num_samples_per_style} per style)...")
            synthetic_features, synthetic_styles, synthetic_classifications = generate_synthetic_samples(
                feature_dicts, valid_styles, valid_classifications,
                num_samples_per_style=num_samples_per_style,
                augmentation_strategy=augmentation_strategy, intensity=intensity
            )
        
        # Add synthetic samples to the dataset
        augmented_feature_dicts = feature_dicts + synthetic_features
        augmented_styles = valid_styles + synthetic_styles
        augmented_paths = valid_paths + [f"synthetic_{i}" for i in range(len(synthetic_features))]
        
        if valid_classifications and synthetic_classifications:
            augmented_classifications = valid_classifications + synthetic_classifications
        else:
            augmented_classifications = None
    else:
        # Just augment existing samples
        print("Augmenting existing samples...")
        augmented_feature_dicts = []
        for feature_dict in tqdm(feature_dicts, desc="Augmenting features"):
            augmented_dict = augment_feature_vector(feature_dict, augmentation_strategy, intensity)
            augmented_feature_dicts.append(augmented_dict)
        
        augmented_styles = valid_styles
        augmented_paths = valid_paths
        augmented_classifications = valid_classifications
    
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
    
    if use_classifications and augmented_classifications:
        original_class_counts = Counter(valid_classifications)
        augmented_class_counts = Counter(augmented_classifications)
        
        print("\nSamples per classification:")
        for classification in sorted(STYLE_CLASSIFICATIONS.keys()):
            original = original_class_counts.get(classification, 0)
            augmented = augmented_class_counts.get(classification, 0)
            print(f"  {classification}: {original} → {augmented}")
            
            # Print breakdown by style within classification
            styles_in_class = STYLE_CLASSIFICATIONS[classification]
            for style in styles_in_class:
                orig_style_count = sum(1 for i, s in enumerate(valid_styles) 
                                     if s == style and valid_classifications[i] == classification)
                aug_style_count = sum(1 for i, s in enumerate(augmented_styles) 
                                    if s == style and augmented_classifications[i] == classification)
                if orig_style_count > 0 or aug_style_count > 0:
                    print(f"    - {style}: {orig_style_count} → {aug_style_count}")
    else:
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
    parser.add_argument('--use-classifications', action='store_true',
                        help='Use the 6 main classifications instead of individual styles')
    parser.add_argument('--num-samples-per-classification', type=int, default=None,
                        help='Target number of samples per classification (only used with --use-classifications)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    """Main function to augment the dataset."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print(f"Loading features from {args.features_path}...")
    
    # Augment dataset
    augmented_features = augment_dataset(
        args.features_path,
        args.output_path,
        args.augmentation_strategy,
        args.intensity,
        args.balance_classes,
        args.num_samples_per_style,
        args.num_samples_per_classification,
        args.use_classifications
    )
    
    print(f"Augmented features saved to {args.output_path}")

if __name__ == "__main__":
    main() 