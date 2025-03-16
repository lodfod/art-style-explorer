#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

from src.model.network import ArtStyleNetwork
from src.model.training import train_model, visualize_training, ArtworkDataset
from src.utils.data_loader import (
    load_metadata, 
    create_artist_mapping, 
    save_artist_mapping, 
    split_dataset,
    create_image_path_list,
    create_artist_label_list,
    get_image_transform,
    get_dataset_stats
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Art Style Explorer model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing artwork images')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to metadata CSV file')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints and results')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda or cpu)')
    
    # GPU optimization arguments
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training (faster on newer GPUs)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    
    # Image processing arguments
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size for model input')
    parser.add_argument('--data-augmentation', action='store_true',
                       help='Use data augmentation during training')
    
    # Feature arguments
    parser.add_argument('--features-path', type=str, default=None,
                       help='Path to precomputed composition features')
    parser.add_argument('--extract-features', action='store_true',
                       help='Extract composition features on-the-fly')
    
    # Model arguments
    parser.add_argument('--line-feature-dim', type=int, default=128,
                       help='Dimension of line feature vector')
    parser.add_argument('--comp-feature-dim', type=int, default=64,
                       help='Dimension of composition feature vector')
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Dimension of art style embedding vector')
    parser.add_argument('--comp-input-dim', type=int, default=33,
                       help='Dimension of input composition feature vector')
    
    return parser.parse_args()


def extract_features_function(image_path, target_size=224, edge_method='canny'):
    """
    Function to extract features on-the-fly
    
    Args:
        image_path: Path to the artwork image
        target_size: Target size for image processing
        edge_method: Edge detection method
        
    Returns:
        Composition features as numpy array
    """
    from src.preprocessing.edge_detection import process_artwork, detect_contours
    from src.features.line_features import extract_line_features
    from src.features.composition import extract_composition_features
    from main import get_features_vector
    
    try:
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
        
        # Convert to feature vector
        features_vector = get_features_vector(line_features, composition_features)
        
        return features_vector
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None


def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    if args.device.startswith('cuda'):
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Setup mixed precision if requested
    use_mixed_precision = args.mixed_precision and args.device.startswith('cuda')
    if use_mixed_precision:
        print("Using mixed precision training")
        scaler = GradScaler()
    else:
        scaler = None
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(args.metadata)
    
    # Create artist mappings
    artist_to_id, id_to_artist = create_artist_mapping(metadata)
    
    # Save artist mappings
    mappings_path = os.path.join(args.output_dir, 'artist_mappings.json')
    save_artist_mapping(artist_to_id, id_to_artist, mappings_path)
    
    # Split dataset
    print("Splitting dataset...")
    train_metadata, val_metadata, test_metadata = split_dataset(metadata)
    
    print(f"Train: {len(train_metadata)} samples")
    print(f"Validation: {len(val_metadata)} samples")
    print(f"Test: {len(test_metadata)} samples")
    
    # Create image path lists
    train_paths = create_image_path_list(train_metadata, args.data_dir)
    val_paths = create_image_path_list(val_metadata, args.data_dir)
    test_paths = create_image_path_list(test_metadata, args.data_dir)
    
    # Create artist label lists
    train_labels = create_artist_label_list(train_metadata, artist_to_id)
    val_labels = create_artist_label_list(val_metadata, artist_to_id)
    test_labels = create_artist_label_list(test_metadata, artist_to_id)
    
    # Get image transformations
    train_transform = get_image_transform(
        target_size=(args.img_size, args.img_size),
        normalize=True,
        data_augmentation=args.data_augmentation
    )
    
    val_transform = get_image_transform(
        target_size=(args.img_size, args.img_size),
        normalize=True,
        data_augmentation=False
    )
    
    # Set up feature extraction function if needed
    extract_func = extract_features_function if args.extract_features else None
    
    # Create datasets
    train_dataset = ArtworkDataset(
        train_paths, 
        train_labels, 
        transform=train_transform,
        feature_path=args.features_path,
        extract_features_fn=extract_func
    )
    
    val_dataset = ArtworkDataset(
        val_paths, 
        val_labels, 
        transform=val_transform,
        feature_path=args.features_path,
        extract_features_fn=extract_func
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Initialize model
    num_artists = len(artist_to_id)
    model = ArtStyleNetwork(
        input_channels=3,  # RGB images
        line_feature_dim=args.line_feature_dim,
        comp_feature_dim=args.comp_feature_dim,
        embedding_dim=args.embedding_dim,
        num_classes=num_artists,
        comp_input_dim=args.comp_input_dim
    )
    
    # Move model to device and optimize for speed
    model = model.to(device)
    if device.type == 'cuda':
        # Optimize model for GPU
        model = torch.compile(model) if hasattr(torch, 'compile') else model
    
    # Print model summary
    print(f"\nModel summary:")
    print(f"Number of artist classes: {num_artists}")
    print(f"Line feature dimension: {args.line_feature_dim}")
    print(f"Composition feature dimension: {args.comp_feature_dim}")
    print(f"Embedding dimension: {args.embedding_dim}")
    
    try:
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
    except:
        pass
    
    # Train model
    print("\nStarting training...")
    metrics = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=args.output_dir,
        use_mixed_precision=use_mixed_precision,
        scaler=scaler
    )
    
    # Visualize training results
    visualization_path = os.path.join(args.output_dir, 'training_visualization.png')
    visualize_training(metrics, save_path=visualization_path)
    
    print(f"Training completed. Model checkpoints and results saved to {args.output_dir}")


if __name__ == '__main__':
    main() 