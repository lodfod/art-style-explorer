#!/usr/bin/env python3
"""
Script to split the feature-extracted dataset into train and test sets.
Supports both individual style classification and hierarchical classification.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

def prepare_dataset(features, feature_type='combined', use_classifications=False):
    """Prepare dataset for model training."""
    X = []
    paths = []
    
    for path, feature_dict in features.items():
        if feature_type in feature_dict:
            X.append(feature_dict[feature_type])
            paths.append(path)
    
    X = np.array(X)
    
    if use_classifications:
        # Use the 6 main classifications as labels
        labels = [extract_classification_from_path(path) for path in paths]
        # Filter to include only paths with valid classifications
        valid_indices = [i for i, label in enumerate(labels) if label != "Unknown"]
    else:
        # Use individual styles as labels
        labels = [extract_style_from_path(path) for path in paths]
        # Filter to include only target styles
        valid_indices = [i for i, label in enumerate(labels) if label in TARGET_STYLES]
    
    X_filtered = X[valid_indices]
    labels_filtered = [labels[i] for i in valid_indices]
    paths_filtered = [paths[i] for i in valid_indices]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_filtered)
    
    # Create a mapping from encoded labels to style/classification names
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
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

def save_splits(splits, label_mapping, output_dir, use_classifications=False):
    """Save train/test splits and label mapping to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    with open(os.path.join(output_dir, 'train_test_splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
    
    # Save label mapping
    with open(os.path.join(output_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    
    # Create summary CSV files
    label_column_name = 'classification' if use_classifications else 'style'
    
    train_df = pd.DataFrame({
        'path': splits['paths_train'],
        'label': splits['y_train'],
        label_column_name: [label_mapping[label] for label in splits['y_train']]
    })
    
    test_df = pd.DataFrame({
        'path': splits['paths_test'],
        'label': splits['y_test'],
        label_column_name: [label_mapping[label] for label in splits['y_test']]
    })
    
    # If using individual styles, add classification column
    if not use_classifications:
        train_df['classification'] = train_df['style'].map(STYLE_TO_CLASSIFICATION)
        test_df['classification'] = test_df['style'].map(STYLE_TO_CLASSIFICATION)
    
    # Save summary CSVs
    train_df.to_csv(os.path.join(output_dir, 'train_summary.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_summary.csv'), index=False)
    
    # Create a label mapping CSV
    mapping_df = pd.DataFrame({
        'label': list(label_mapping.keys()),
        label_column_name: list(label_mapping.values())
    })
    mapping_df.to_csv(os.path.join(output_dir, 'label_mapping.csv'), index=False)
    
    # Print dataset statistics
    print(f"Dataset split complete:")
    print(f"  Training set: {len(splits['X_train'])} samples")
    print(f"  Test set: {len(splits['X_test'])} samples")
    print(f"  Number of classes: {len(label_mapping)}")
    
    # Print class distribution
    train_class_dist = train_df[label_column_name].value_counts().sort_index()
    test_class_dist = test_df[label_column_name].value_counts().sort_index()
    
    print(f"\nClass distribution ({label_column_name}):")
    for class_name in sorted(label_mapping.values()):
        train_count = train_class_dist.get(class_name, 0)
        test_count = test_class_dist.get(class_name, 0)
        print(f"  {class_name}: {train_count} train, {test_count} test")
    
    # If using individual styles, also print distribution by classification
    if not use_classifications:
        train_classification_dist = train_df['classification'].value_counts().sort_index()
        test_classification_dist = test_df['classification'].value_counts().sort_index()
        
        print("\nClassification distribution:")
        for classification in sorted(STYLE_CLASSIFICATIONS.keys()):
            train_count = train_classification_dist.get(classification, 0)
            test_count = test_classification_dist.get(classification, 0)
            print(f"  {classification}: {train_count} train, {test_count} test")

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
    parser.add_argument('--use-classifications', action='store_true',
                        help='Use the 6 main classifications instead of individual styles')
    return parser.parse_args()

def main():
    """Main function to split the dataset."""
    args = parse_args()
    
    # Load features
    print(f"Loading features from {args.features_path}...")
    features = load_features(args.features_path)
    
    # Prepare dataset
    label_type = "classifications" if args.use_classifications else "styles"
    print(f"Preparing dataset using {args.feature_type} features and {label_type} as labels...")
    X, y, paths, label_mapping = prepare_dataset(features, args.feature_type, args.use_classifications)
    print(f"Prepared dataset with {len(X)} samples and {len(label_mapping)} classes")
    
    # Split dataset
    print(f"Splitting dataset with test_size={args.test_size}...")
    splits = split_dataset(X, y, paths, args.test_size, args.random_state)
    
    # Save splits
    print(f"Saving splits to {args.output_dir}...")
    save_splits(splits, label_mapping, args.output_dir, args.use_classifications)
    
    print(f"Dataset splitting complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 