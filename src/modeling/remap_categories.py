#!/usr/bin/env python3
"""
Script to remap art style categories without re-extracting features.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Define your new category mapping here
NEW_STYLE_CATEGORIES = {
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
for styles in NEW_STYLE_CATEGORIES.values():
    TARGET_STYLES.extend(styles)

# Create a mapping from original style to new category
STYLE_TO_NEW_CATEGORY = {}
for new_category, styles in NEW_STYLE_CATEGORIES.items():
    for style in styles:
        STYLE_TO_NEW_CATEGORY[style] = new_category

def extract_style_from_path(path):
    """Extract style name from image path."""
    parts = Path(path).parts
    for style in TARGET_STYLES:
        if style in parts:
            return style
    
    # If not found in TARGET_STYLES, check for old style names
    old_styles = ["Early-Renaissance", "High-Renaissance", "Mannerism", "Neoclassicism", 
                 "Classicism", "Romanesque", "Baroque", "Impressionism", "Post-Impressionism",
                 "Expressionism", "Surrealism", "Cubism", "Abstract-Art", "Ukiyo-e", 
                 "Pop-Art", "Art-Nouveau-(Modern)"]
    
    for style in old_styles:
        if style in parts:
            return style
    
    return parts[-2]  # Fallback to second-to-last part

def remap_categories(input_dir, output_dir):
    """Remap categories without re-extracting features."""
    # Load existing data
    with open(os.path.join(input_dir, 'train_test_splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    
    with open(os.path.join(input_dir, 'label_mapping.pkl'), 'rb') as f:
        old_label_mapping = pickle.load(f)
    
    # Get original style names
    train_styles = [extract_style_from_path(path) for path in splits['paths_train']]
    test_styles = [extract_style_from_path(path) for path in splits['paths_test']]
    
    # Map to new categories
    train_new_categories = [STYLE_TO_NEW_CATEGORY.get(style, "Other") for style in train_styles]
    test_new_categories = [STYLE_TO_NEW_CATEGORY.get(style, "Other") for style in test_styles]
    
    # Encode the new categories
    label_encoder = LabelEncoder()
    all_categories = train_new_categories + test_new_categories
    label_encoder.fit(all_categories)
    
    y_train_new = label_encoder.transform(train_new_categories)
    y_test_new = label_encoder.transform(test_new_categories)
    
    # Create new label mapping
    new_label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Create new splits dictionary
    new_splits = {
        'X_train': splits['X_train'],
        'X_test': splits['X_test'],
        'y_train': y_train_new,
        'y_test': y_test_new,
        'paths_train': splits['paths_train'],
        'paths_test': splits['paths_test'],
        'styles_train': train_styles,
        'styles_test': test_styles,
        'classifications_train': train_new_categories,
        'classifications_test': test_new_categories
    }
    
    # Save the new splits and label mapping
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train_test_splits.pkl'), 'wb') as f:
        pickle.dump(new_splits, f)
    
    with open(os.path.join(output_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(new_label_mapping, f)
    
    # Create summary CSVs
    train_df = pd.DataFrame({
        'path': splits['paths_train'],
        'original_style': train_styles,
        'new_category': train_new_categories,
        'label': y_train_new
    })
    
    test_df = pd.DataFrame({
        'path': splits['paths_test'],
        'original_style': test_styles,
        'new_category': test_new_categories,
        'label': y_test_new
    })
    
    train_df.to_csv(os.path.join(output_dir, 'train_summary.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_summary.csv'), index=False)
    
    # Create a mapping CSV
    mapping_df = pd.DataFrame({
        'original_style': sorted(set(train_styles + test_styles)),
        'new_category': [STYLE_TO_NEW_CATEGORY.get(style, "Other") for style in sorted(set(train_styles + test_styles))]
    })
    mapping_df.to_csv(os.path.join(output_dir, 'category_mapping.csv'), index=False)
    
    # Print statistics
    print(f"Remapped dataset statistics:")
    print(f"  Training set: {len(new_splits['X_train'])} samples")
    print(f"  Test set: {len(new_splits['X_test'])} samples")
    print(f"  Number of new categories: {len(new_label_mapping)}")
    
    # Print class distribution
    train_class_dist = pd.Series(train_new_categories).value_counts()
    test_class_dist = pd.Series(test_new_categories).value_counts()
    
    print("\nClass distribution:")
    for class_name in sorted(new_label_mapping.values()):
        train_count = train_class_dist.get(class_name, 0)
        test_count = test_class_dist.get(class_name, 0)
        print(f"  {class_name}: {train_count} train, {test_count} test")
    
    return new_splits, new_label_mapping

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Remap art style categories without re-extracting features')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing original train/test splits')
    parser.add_argument('--output-dir', type=str, default='data/model_remapped',
                        help='Directory to save remapped data')
    return parser.parse_args()

def main():
    """Main function to remap categories."""
    args = parse_args()
    remap_categories(args.input_dir, args.output_dir)
    print(f"Category remapping complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
