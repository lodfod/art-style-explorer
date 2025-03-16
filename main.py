#!/usr/bin/env python3
import os
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.edge_detection import process_artwork, detect_contours, analyze_contours
from src.preprocessing.normalization import standardize_artwork
from src.features.line_features import extract_line_features, extract_hough_lines
from src.features.composition import extract_composition_features
from src.model.network import ArtStyleNetwork
from src.model.training import load_model
from src.utils.data_loader import load_artist_mapping
from src.utils.visualization import (
    visualize_edges,
    visualize_line_features,
    visualize_composition_features,
    visualize_similar_artworks
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Art Style Explorer')
    
    # Input/output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input artwork image')
    parser.add_argument('--output-dir', '-o', type=str, default='results',
                        help='Directory to save results')
    
    # Database arguments
    parser.add_argument('--database', '-db', type=str, required=True,
                        help='Path to artwork database metadata CSV')
    parser.add_argument('--data-dir', '-dd', type=str, required=True,
                        help='Directory containing artwork images')
    parser.add_argument('--mappings', '-m', type=str, required=True,
                        help='Path to artist mapping JSON file')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference (cuda or cpu)')
    
    # Processing arguments
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of similar artworks to retrieve')
    parser.add_argument('--target-size', type=int, default=512,
                        help='Target size for image processing')
    parser.add_argument('--edge-method', type=str, default='canny',
                        choices=['canny', 'sobel', 'laplacian'],
                        help='Edge detection method')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize',
                        help='Do not visualize results')
    parser.set_defaults(visualize=True)
    
    return parser.parse_args()


def extract_features_from_image(image_path: str, 
                               target_size: int = 512,
                               edge_method: str = 'canny') -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Extract features from an artwork image
    
    Args:
        image_path: Path to the artwork image
        target_size: Target size for image processing
        edge_method: Edge detection method
        
    Returns:
        Tuple of (line_features, composition_features, preprocessed_image, edge_image)
    """
    # Process artwork to extract line work
    preprocessed, edges = process_artwork(
        image_path, 
        target_size=(target_size, target_size),
        edge_method=edge_method
    )
    
    # Detect contours
    contours = detect_contours(edges)
    
    # Extract line features
    line_features = extract_line_features(edges, contours)
    
    # Extract composition features
    composition_features = extract_composition_features(preprocessed)
    
    return line_features, composition_features, preprocessed, edges


def get_features_vector(line_features: Dict[str, Any], composition_features: Dict[str, Any]) -> np.ndarray:
    """
    Convert extracted features to a feature vector
    
    Args:
        line_features: Dictionary of line features
        composition_features: Dictionary of composition features
        
    Returns:
        Feature vector as numpy array
    """
    # Select numerical features from line_features
    line_values = [
        line_features['line_count'],
        line_features['mean_length'],
        line_features['std_length'],
        line_features['max_length'],
        line_features['min_length'],
        line_features['horizontal_ratio'],
        line_features['vertical_ratio'],
        line_features['diagonal_ratio'],
        line_features['mean_curvature'],
        line_features['std_curvature'],
        line_features['max_curvature'],
        line_features['min_curvature'],
        line_features['intersection_count'],
        line_features['contour_count'],
        line_features['avg_complexity']
    ]
    
    # Add orientation histogram
    line_values.extend(line_features['orientation_histogram'])
    
    # Select numerical features from composition_features
    comp_values = [
        composition_features['horizontal_line_energy'],
        composition_features['vertical_line_energy'],
        composition_features['intersection_energy'],
        composition_features['thirds_adherence'],
        composition_features['horizontal_symmetry'],
        composition_features['vertical_symmetry'],
        composition_features['diagonal_symmetry'],
        composition_features['overall_symmetry'],
        composition_features['golden_horizontal_energy'],
        composition_features['golden_vertical_energy'],
        composition_features['golden_spiral_energy'],
        composition_features['golden_ratio_adherence'],
        composition_features['horizontal_balance'],
        composition_features['vertical_balance'],
        composition_features['radial_balance']
    ]
    
    # Combine line and composition features
    features_vector = np.array(line_values + comp_values, dtype=np.float32)
    
    return features_vector


def load_database_features(database_path: str, 
                          data_dir: str,
                          target_size: int = 512,
                          edge_method: str = 'canny',
                          cache_path: Optional[str] = None) -> Tuple[List[str], List[str], List[str], List[np.ndarray]]:
    """
    Load features from artwork database
    
    Args:
        database_path: Path to the artwork database metadata CSV
        data_dir: Directory containing artwork images
        target_size: Target size for image processing
        edge_method: Edge detection method
        cache_path: Optional path to cached features
        
    Returns:
        Tuple of (image_paths, artist_names, artwork_titles, features_vector_list)
    """
    # Load database metadata
    database = pd.read_csv(database_path)
    
    # Check if we can use cached features
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        features_data = np.load(cache_path, allow_pickle=True).item()
        image_paths = features_data['image_paths']
        artist_names = features_data['artist_names']
        artwork_titles = features_data['artwork_titles']
        features_vector_list = features_data['features_vector_list']
    else:
        # Extract features for each artwork in the database
        image_paths = []
        artist_names = []
        artwork_titles = []
        features_vector_list = []
        
        print(f"Extracting features from {len(database)} artworks...")
        for idx, row in database.iterrows():
            # Get image path
            filename = row['filename']
            image_path = os.path.join(data_dir, filename)
            
            # Skip if file doesn't exist
            if not os.path.exists(image_path):
                print(f"Warning: File not found: {image_path}")
                continue
            
            # Extract features
            try:
                print(f"Processing {idx+1}/{len(database)}: {filename}")
                line_features, composition_features, _, _ = extract_features_from_image(
                    image_path, target_size, edge_method
                )
                
                # Convert to feature vector
                features_vector = get_features_vector(line_features, composition_features)
                
                # Append to lists
                image_paths.append(image_path)
                artist_names.append(row['artist'])
                artwork_titles.append(row.get('title', filename))
                features_vector_list.append(features_vector)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Cache the features if a cache path is provided
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            features_data = {
                'image_paths': image_paths,
                'artist_names': artist_names,
                'artwork_titles': artwork_titles,
                'features_vector_list': features_vector_list
            }
            
            np.save(cache_path, features_data)
            print(f"Cached features saved to {cache_path}")
    
    return image_paths, artist_names, artwork_titles, features_vector_list


def find_similar_artworks(model: torch.nn.Module,
                         query_image_path: str,
                         database_image_paths: List[str],
                         database_artist_names: List[str],
                         database_artwork_titles: List[str],
                         database_features: List[np.ndarray],
                         target_size: int = 512,
                         edge_method: str = 'canny',
                         top_k: int = 5,
                         device: str = 'cuda') -> Tuple[List[str], List[str], List[str], List[float]]:
    """
    Find artworks with similar styles in the database
    
    Args:
        model: Trained neural network model
        query_image_path: Path to the query artwork image
        database_image_paths: List of paths to database artwork images
        database_artist_names: List of database artist names
        database_artwork_titles: List of database artwork titles
        database_features: List of database feature vectors
        target_size: Target size for image processing
        edge_method: Edge detection method
        top_k: Number of similar artworks to retrieve
        device: Device to use for inference
        
    Returns:
        Tuple of (similar_image_paths, similar_artists, similar_titles, similarities)
    """
    # Extract features from query image
    line_features, composition_features, _, _ = extract_features_from_image(
        query_image_path, target_size, edge_method
    )
    
    # Convert to feature vector
    query_features_vector = get_features_vector(line_features, composition_features)
    
    # Load query image for the model
    query_image = cv2.imread(query_image_path)
    query_image = cv2.resize(query_image, (target_size, target_size))
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Convert to torch tensor
    query_image_tensor = torch.tensor(query_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    query_features_tensor = torch.tensor(query_features_vector, dtype=torch.float32).unsqueeze(0)
    
    # Move to device
    query_image_tensor = query_image_tensor.to(device)
    query_features_tensor = query_features_tensor.to(device)
    
    # Extract embedding from query image
    with torch.no_grad():
        query_embedding = model.extract_features(query_image_tensor, query_features_tensor)
    
    # Calculate similarities with all database images
    similarities = []
    
    for features_vector in database_features:
        # Convert to torch tensor
        db_features_tensor = torch.tensor(features_vector, dtype=torch.float32).unsqueeze(0).to(device)
        
        # For simplicity, we're using only the composition features for comparison
        # In a real application, you would need to process the database images through the model
        # and extract embeddings from them
        db_image_tensor = torch.zeros((1, 1, target_size, target_size), device=device)  # Dummy tensor
        
        with torch.no_grad():
            db_embedding = model.extract_features(db_image_tensor, db_features_tensor)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(query_embedding, db_embedding).item()
            similarities.append(similarity)
    
    # Find top-k similar artworks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    similar_image_paths = [database_image_paths[i] for i in top_indices]
    similar_artists = [database_artist_names[i] for i in top_indices]
    similar_titles = [database_artwork_titles[i] for i in top_indices]
    similar_scores = [similarities[i] for i in top_indices]
    
    return similar_image_paths, similar_artists, similar_titles, similar_scores


def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load artist mappings
    artist_to_id, id_to_artist = load_artist_mapping(args.mappings)
    
    # Initialize model
    num_artists = len(artist_to_id)
    model = ArtStyleNetwork(
        input_channels=1,  # Grayscale images
        line_feature_dim=128,
        comp_feature_dim=64,
        embedding_dim=256,
        num_classes=num_artists,
        comp_input_dim=33  # 15 line features + 18 orientation bins
    )
    
    # Load trained model
    model = load_model(model, args.model, device=args.device)
    
    # Extract features from query image
    print(f"Extracting features from query image: {args.input}")
    line_features, composition_features, preprocessed, edges = extract_features_from_image(
        args.input, args.target_size, args.edge_method
    )
    
    # Visualize processing results
    if args.visualize:
        # Visualize edge detection
        edge_fig = visualize_edges(preprocessed, edges)
        edge_path = os.path.join(args.output_dir, 'edges.png')
        edge_fig.savefig(edge_path, dpi=300, bbox_inches='tight')
        print(f"Edge visualization saved to {edge_path}")
        
        # Visualize line features
        lines = extract_hough_lines(edges)
        line_fig = visualize_line_features(preprocessed, lines)
        line_path = os.path.join(args.output_dir, 'lines.png')
        line_fig.savefig(line_path, dpi=300, bbox_inches='tight')
        print(f"Line visualization saved to {line_path}")
        
        # Visualize composition features
        comp_fig = visualize_composition_features(preprocessed, composition_features)
        comp_path = os.path.join(args.output_dir, 'composition.png')
        comp_fig.savefig(comp_path, dpi=300, bbox_inches='tight')
        print(f"Composition visualization saved to {comp_path}")
    
    # Load or extract database features
    cache_path = os.path.join(args.output_dir, 'database_features.npy')
    image_paths, artist_names, artwork_titles, features_vector_list = load_database_features(
        args.database, args.data_dir, args.target_size, args.edge_method, cache_path
    )
    
    # Find similar artworks
    print(f"Finding top {args.top_k} similar artworks...")
    similar_paths, similar_artists, similar_titles, similarities = find_similar_artworks(
        model, args.input, image_paths, artist_names, artwork_titles, features_vector_list,
        args.target_size, args.edge_method, args.top_k, args.device
    )
    
    # Print results
    print("\nTop similar artworks:")
    for i, (artist, title, similarity) in enumerate(zip(similar_artists, similar_titles, similarities)):
        print(f"{i+1}. {artist} - {title} (Similarity: {similarity:.4f})")
    
    # Visualize similar artworks
    if args.visualize and similar_paths:
        # Load query image
        query_image = cv2.imread(args.input)
        
        # Load similar images
        similar_images = [cv2.imread(path) for path in similar_paths]
        
        # Visualize
        similar_fig = visualize_similar_artworks(
            query_image, similar_images, similar_artists, similar_titles, similarities
        )
        similar_path = os.path.join(args.output_dir, 'similar_artworks.png')
        similar_fig.savefig(similar_path, dpi=300, bbox_inches='tight')
        print(f"Similar artworks visualization saved to {similar_path}")
    
    print(f"All results saved to {args.output_dir}")


if __name__ == '__main__':
    main() 