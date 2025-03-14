import os
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image
from torchvision import transforms


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load artwork metadata from a CSV file
    
    Args:
        metadata_path: Path to the metadata CSV file
        
    Returns:
        DataFrame containing artwork metadata
    """
    metadata = pd.read_csv(metadata_path)
    
    # Check if required columns exist
    required_columns = ['filename', 'artist']
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    
    if missing_columns:
        raise ValueError(f"Metadata file is missing required columns: {missing_columns}")
    
    return metadata


def create_artist_mapping(metadata: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create mappings between artist names and IDs
    
    Args:
        metadata: DataFrame containing artwork metadata
        
    Returns:
        Tuple of (artist_to_id, id_to_artist) dictionaries
    """
    # Get unique artist names
    unique_artists = sorted(metadata['artist'].unique())
    
    # Create mappings
    artist_to_id = {artist: i for i, artist in enumerate(unique_artists)}
    id_to_artist = {i: artist for artist, i in artist_to_id.items()}
    
    return artist_to_id, id_to_artist


def save_artist_mapping(artist_to_id: Dict[str, int], 
                       id_to_artist: Dict[int, str], 
                       output_path: str) -> None:
    """
    Save artist mappings to a JSON file
    
    Args:
        artist_to_id: Dictionary mapping artist names to IDs
        id_to_artist: Dictionary mapping IDs to artist names
        output_path: Path to save the JSON file
    """
    # Convert integer keys to strings for JSON serialization
    id_to_artist_str = {str(k): v for k, v in id_to_artist.items()}
    
    mappings = {
        'artist_to_id': artist_to_id,
        'id_to_artist': id_to_artist_str
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"Artist mappings saved to {output_path}")


def load_artist_mapping(mappings_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load artist mappings from a JSON file
    
    Args:
        mappings_path: Path to the mappings JSON file
        
    Returns:
        Tuple of (artist_to_id, id_to_artist) dictionaries
    """
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    
    artist_to_id = mappings['artist_to_id']
    
    # Convert string keys back to integers
    id_to_artist = {int(k): v for k, v in mappings['id_to_artist'].items()}
    
    return artist_to_id, id_to_artist


def split_dataset(metadata: pd.DataFrame,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test sets
    
    Args:
        metadata: DataFrame containing artwork metadata
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
        test_ratio: Ratio of test samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_metadata, val_metadata, test_metadata) DataFrames
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Group by artist
    artist_groups = metadata.groupby('artist')
    
    train_data = []
    val_data = []
    test_data = []
    
    # For each artist, split their artworks into train/val/test
    for artist, group in artist_groups:
        # Shuffle the artworks
        shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Calculate split indices
        n_samples = len(shuffled)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Split the data
        train_data.append(shuffled[:n_train])
        val_data.append(shuffled[n_train:n_train+n_val])
        test_data.append(shuffled[n_train+n_val:])
    
    # Combine the splits
    train_metadata = pd.concat(train_data).reset_index(drop=True)
    val_metadata = pd.concat(val_data).reset_index(drop=True)
    test_metadata = pd.concat(test_data).reset_index(drop=True)
    
    return train_metadata, val_metadata, test_metadata


def get_image_transform(target_size: Tuple[int, int] = (512, 512),
                       normalize: bool = True,
                       data_augmentation: bool = False) -> transforms.Compose:
    """
    Get image transformation pipeline
    
    Args:
        target_size: Target image size (height, width)
        normalize: Whether to normalize the image
        data_augmentation: Whether to apply data augmentation
        
    Returns:
        Composed transformation pipeline
    """
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize(target_size))
    
    # Data augmentation for training
    if data_augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    return transforms.Compose(transform_list)


def get_dataset_stats(image_paths: List[str], 
                     sample_size: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Calculate dataset mean and standard deviation
    
    Args:
        image_paths: List of image file paths
        sample_size: Number of images to sample for calculation
        
    Returns:
        Tuple of (mean, std) lists for each channel
    """
    # Sample images if there are more than sample_size
    if len(image_paths) > sample_size:
        sampled_paths = random.sample(image_paths, sample_size)
    else:
        sampled_paths = image_paths
    
    # Initialize arrays for means and stds
    means = []
    stds = []
    
    # Process each image
    for path in sampled_paths:
        try:
            # Open image and convert to RGB
            img = Image.open(path).convert('RGB')
            
            # Convert to numpy array and normalize to [0, 1]
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Calculate mean and std for each channel
            means.append(np.mean(img_np, axis=(0, 1)))
            stds.append(np.std(img_np, axis=(0, 1)))
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Calculate overall mean and std
    mean = np.mean(means, axis=0).tolist()
    std = np.mean(stds, axis=0).tolist()
    
    return mean, std


def load_features(feature_path: str) -> Dict[str, np.ndarray]:
    """
    Load pre-extracted features from a numpy file
    
    Args:
        feature_path: Path to the features numpy file
        
    Returns:
        Dictionary mapping image paths to feature vectors
    """
    return np.load(feature_path, allow_pickle=True).item()


def save_features(features: Dict[str, np.ndarray], output_path: str) -> None:
    """
    Save extracted features to a numpy file
    
    Args:
        features: Dictionary mapping image paths to feature vectors
        output_path: Path to save the features
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save features
    np.save(output_path, features)
    
    print(f"Features saved to {output_path}")


def create_image_path_list(metadata: pd.DataFrame, 
                          data_dir: str) -> List[str]:
    """
    Create a list of full image paths
    
    Args:
        metadata: DataFrame containing artwork metadata
        data_dir: Base directory containing the images
        
    Returns:
        List of full image paths
    """
    image_paths = []
    
    for _, row in metadata.iterrows():
        # Get filename from metadata
        filename = row['filename']
        
        # Create full path
        full_path = os.path.join(data_dir, filename)
        
        # Check if file exists
        if os.path.exists(full_path):
            image_paths.append(full_path)
        else:
            print(f"Warning: Image not found at {full_path}")
    
    return image_paths


def create_artist_label_list(metadata: pd.DataFrame, 
                            artist_to_id: Dict[str, int]) -> List[int]:
    """
    Create a list of artist labels
    
    Args:
        metadata: DataFrame containing artwork metadata
        artist_to_id: Dictionary mapping artist names to IDs
        
    Returns:
        List of artist label IDs
    """
    return [artist_to_id[artist] for artist in metadata['artist']] 