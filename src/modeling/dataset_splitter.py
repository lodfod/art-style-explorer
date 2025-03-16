#!/usr/bin/env python3
"""
Script to split the feature-extracted dataset into train and test sets.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

def prepare_dataset(features, feature_type='combined'):
    """Prepare dataset for model training."""
    X = []
    paths = []
    
    for path, feature_dict in features.items():
        if feature_type in feature_dict:
            X.append(feature_dict[feature_type])
            paths.append(path)
    
    X = np.array(X)
    styles = [extract_style_from_path(path) for path in paths]
    
    # Filter to include only target styles
    valid_indices = [i for i, style in enumerate(styles) if style in TARGET_STYLES]
    X_filtered = X[valid_indices]
    styles_filtered = [styles[i] for i in valid_indices]
    paths_filtered = [paths[i] for i in valid_indices]
    
    # Encode style labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(styles_filtered)
    
    # Create a mapping from encoded labels to style names
    label_mapping = {i: style for i, style in enumerate(label_encoder.classes_)}
    
    return X_filtered, y, paths_filtered, label_mapping

def split_dataset(X, y, paths, test_size=0.2, random_state=42):
    """Split dataset into train and test sets."""
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'paths_train': paths_train,
        'paths_test': paths_test
    }

def save_splits(splits, label_mapping, output_dir):
    """Save train/test splits and label mapping to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    with open(os.path.join(output_dir, 'train_test_splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
    
    # Save label mapping
    with open(os.path.join(output_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    
    # Create summary CSV files
    train_df = pd.DataFrame({
        'path': splits['paths_train'],
        'label': splits['y_train'],
        'style': [label_mapping[label] for label in splits['y_train']]
    })
    
    test_df = pd.DataFrame({
        'path': splits['paths_test'],
        'label': splits['y_test'],
        'style': [label_mapping[label] for label in splits['y_test']]
    })
    
    # Save summary CSVs
    train_df.to_csv(os.path.join(output_dir, 'train_summary.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_summary.csv'), index=False)
    
    # Create a label mapping CSV
    mapping_df = pd.DataFrame({
        'label': list(label_mapping.keys()),
        'style': list(label_mapping.values())
    })
    mapping_df.to_csv(os.path.join(output_dir, 'label_mapping.csv'), index=False)
    
    # Print dataset statistics
    print(f"Dataset split complete:")
    print(f"  Training set: {len(splits['X_train'])} samples")
    print(f"  Test set: {len(splits['X_test'])} samples")
    print(f"  Number of classes: {len(label_mapping)}")
    
    # Print class distribution
    train_class_dist = train_df['style'].value_counts().sort_index()
    test_class_dist = test_df['style'].value_counts().sort_index()
    
    print("\nClass distribution:")
    for style in sorted(label_mapping.values()):
        train_count = train_class_dist.get(style, 0)
        test_count = test_class_dist.get(style, 0)
        print(f"  {style}: {train_count} train, {test_count} test")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Split feature-extracted dataset into train and test sets')
    parser.add_argument('--features-path', type=str, required=True,
                        help='Path to the artwork features pickle file')
    parser.add_argument('--output-dir', type=str, default='data/model',
                        help='Directory to save train/test splits')
    parser.add_argument('--feature-type', type=str, default='combined',
                        help='Type of feature to use (e.g., combined, color_hist_rgb, hog)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    return parser.parse_args()

def main():
    """Main function to split the dataset."""
    args = parse_args()
    
    # Load features
    print(f"Loading features from {args.features_path}...")
    features = load_features(args.features_path)
    
    # Prepare dataset
    print(f"Preparing dataset using {args.feature_type} features...")
    X, y, paths, label_mapping = prepare_dataset(features, args.feature_type)
    print(f"Prepared dataset with {len(X)} samples and {len(label_mapping)} classes")
    
    # Split dataset
    print(f"Splitting dataset with test_size={args.test_size}...")
    splits = split_dataset(X, y, paths, args.test_size, args.random_state)
    
    # Save splits
    print(f"Saving splits to {args.output_dir}...")
    save_splits(splits, label_mapping, args.output_dir)
    
    print(f"Dataset splitting complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 