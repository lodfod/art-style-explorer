#!/usr/bin/env python3
"""
Visualization tools for artwork features.
Provides functions to visualize and analyze extracted features from artwork images.
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
import umap

def load_features(features_path):
    """Load extracted features from a pickle file."""
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    return features

def extract_style_from_path(path):
    """Extract style name from image path."""
    # Assuming the directory structure is .../style_name/image_name.jpg
    return Path(path).parts[-2]

def prepare_feature_data(features, feature_type='combined'):
    """Prepare feature data for visualization."""
    X = []
    paths = []
    
    for path, feature_dict in features.items():
        if feature_type in feature_dict:
            X.append(feature_dict[feature_type])
            paths.append(path)
    
    X = np.array(X)
    styles = [extract_style_from_path(path) for path in paths]
    
    return X, styles, paths

def apply_dimensionality_reduction(X, method='pca', n_components=2, random_state=42):
    """Apply dimensionality reduction to feature data."""
    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)
    
    if method.lower() == 'pca':
        model = PCA(n_components=n_components, random_state=random_state)
    elif method.lower() == 'tsne':
        model = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
    elif method.lower() == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_reduced = model.fit_transform(X_scaled)
    
    # If PCA, calculate explained variance
    explained_variance = None
    if method.lower() == 'pca':
        explained_variance = model.explained_variance_ratio_
    
    return X_reduced, explained_variance

def plot_2d_scatter(X_2d, styles, title, output_path=None, figsize=(12, 10)):
    """Create a 2D scatter plot of reduced features."""
    plt.figure(figsize=figsize)
    
    # Get unique styles and assign colors
    unique_styles = sorted(set(styles))
    
    # Create scatter plot
    for style in unique_styles:
        mask = [s == style for s in styles]
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=style, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()

def plot_3d_scatter(X_3d, styles, title, output_path=None, figsize=(12, 10)):
    """Create a 3D scatter plot of reduced features."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique styles and assign colors
    unique_styles = sorted(set(styles))
    
    # Create scatter plot
    for style in unique_styles:
        mask = [s == style for s in styles]
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], label=style, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()

def plot_feature_importance(pca, feature_names, n_components=5, output_path=None, figsize=(12, 8)):
    """Plot feature importance based on PCA components."""
    plt.figure(figsize=figsize)
    
    # Get the absolute values of the components
    components = np.abs(pca.components_)
    
    # Plot heatmap of component weights
    sns.heatmap(
        components[:n_components, :],
        annot=False,
        cmap='viridis',
        xticklabels=feature_names,
        yticklabels=[f'PC{i+1}' for i in range(n_components)]
    )
    
    plt.title('Feature Importance in Principal Components')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()

def plot_explained_variance(explained_variance, output_path=None, figsize=(10, 6)):
    """Plot explained variance ratio of PCA components."""
    plt.figure(figsize=figsize)
    
    # Cumulative explained variance
    cumulative = np.cumsum(explained_variance)
    
    # Plot individual and cumulative explained variance
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative) + 1), cumulative, where='mid', label='Cumulative')
    
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()

def plot_style_distribution(styles, output_path=None, figsize=(12, 8)):
    """Plot distribution of art styles in the dataset."""
    plt.figure(figsize=figsize)
    
    # Count occurrences of each style
    style_counts = pd.Series(styles).value_counts().sort_values(ascending=False)
    
    # Create bar plot
    sns.barplot(x=style_counts.values, y=style_counts.index)
    
    plt.title('Distribution of Art Styles')
    plt.xlabel('Count')
    plt.ylabel('Art Style')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()

def plot_feature_correlation(X, feature_names, output_path=None, figsize=(14, 12)):
    """Plot correlation matrix of features."""
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, 
                xticklabels=feature_names, yticklabels=feature_names)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize artwork features')
    parser.add_argument('--features-path', type=str, required=True,
                        help='Path to the artwork features pickle file')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--feature-type', type=str, default='combined',
                        help='Type of feature to visualize (e.g., combined, color_hist_rgb, hog)')
    parser.add_argument('--reduction-method', type=str, default='pca',
                        choices=['pca', 'tsne', 'umap'],
                        help='Dimensionality reduction method')
    parser.add_argument('--n-components', type=int, default=2,
                        help='Number of components for dimensionality reduction')
    return parser.parse_args()

def main():
    """Main function to visualize artwork features."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.features_path}...")
    features = load_features(args.features_path)
    
    # Prepare feature data
    print(f"Preparing {args.feature_type} features...")
    X, styles, paths = prepare_feature_data(features, args.feature_type)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    
    # Plot style distribution
    print("Plotting style distribution...")
    plot_style_distribution(
        styles,
        output_path=os.path.join(args.output_dir, 'style_distribution.png')
    )
    
    # Apply dimensionality reduction
    print(f"Applying {args.reduction_method} dimensionality reduction...")
    X_reduced, explained_variance = apply_dimensionality_reduction(
        X, method=args.reduction_method, n_components=args.n_components
    )
    
    # Plot reduced features
    print("Plotting reduced features...")
    if args.n_components == 2:
        plot_2d_scatter(
            X_reduced, styles,
            title=f'{args.reduction_method.upper()} of {args.feature_type} Features',
            output_path=os.path.join(args.output_dir, f'{args.reduction_method}_{args.feature_type}_2d.png')
        )
    elif args.n_components == 3:
        plot_3d_scatter(
            X_reduced, styles,
            title=f'{args.reduction_method.upper()} of {args.feature_type} Features',
            output_path=os.path.join(args.output_dir, f'{args.reduction_method}_{args.feature_type}_3d.png')
        )
    
    # If using PCA, plot explained variance
    if args.reduction_method == 'pca' and explained_variance is not None:
        print("Plotting explained variance...")
        plot_explained_variance(
            explained_variance,
            output_path=os.path.join(args.output_dir, 'pca_explained_variance.png')
        )
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 