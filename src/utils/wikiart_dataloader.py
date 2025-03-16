import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import requests
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from pathlib import Path
import logging

from src.preprocessing.edge_detection import (
    process_artwork, 
    extract_edges, 
    preprocess_image
)
from src.features.line_features import extract_line_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikiArtDataset:
    """Class to handle loading and processing of the WikiArt dataset"""
    
    def __init__(self, csv_path: str, cache_dir: str = 'data/image_cache'):
        """
        Initialize the WikiArt dataset
        
        Args:
            csv_path: Path to the WikiArt CSV file
            cache_dir: Directory to cache downloaded images
        """
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.data = None
        self.style_to_id = None
        self.id_to_style = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load the dataset
        self._load_data()
        
    def _load_data(self):
        """Load the WikiArt dataset from CSV"""
        logger.info(f"Loading WikiArt dataset from {self.csv_path}")
        
        # Load the CSV
        try:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded dataset with {len(self.data)} entries")
            
            # Check required columns
            required_columns = ['Style', 'Artwork', 'Artist', 'Link']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Dataset is missing required columns: {missing_columns}")
                
            # Create style mappings
            unique_styles = sorted(self.data['Style'].unique())
            self.style_to_id = {style: i for i, style in enumerate(unique_styles)}
            self.id_to_style = {i: style for i, style in enumerate(unique_styles)}
            
            logger.info(f"Found {len(unique_styles)} unique art styles")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def split_dataset(self, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, 
                     seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train, validation, and test sets
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
        
        # Group by style to ensure stratified sampling
        train_dfs, val_dfs, test_dfs = [], [], []
        
        np.random.seed(seed)
        
        # Process each style separately to maintain class balance
        for style, group in self.data.groupby('Style'):
            # Shuffle the data
            indices = np.random.permutation(len(group))
            group_shuffled = group.iloc[indices].reset_index(drop=True)
            
            # Calculate split sizes
            n_train = int(len(group_shuffled) * train_ratio)
            n_val = int(len(group_shuffled) * val_ratio)
            
            # Split the data
            train_data = group_shuffled.iloc[:n_train]
            val_data = group_shuffled.iloc[n_train:n_train + n_val]
            test_data = group_shuffled.iloc[n_train + n_val:]
            
            train_dfs.append(train_data)
            val_dfs.append(val_data)
            test_dfs.append(test_data)
        
        # Combine all styles
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        val_df = pd.concat(val_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)
        
        logger.info(f"Dataset split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def download_image(self, url: str, artwork_id: str) -> Optional[Image.Image]:
        """
        Download an image from a URL or load from cache
        
        Args:
            url: URL of the image
            artwork_id: Unique identifier for the artwork
            
        Returns:
            PIL Image object or None if download fails
        """
        # Create a safe filename
        safe_id = ''.join(c if c.isalnum() else '_' for c in artwork_id)
        cache_path = os.path.join(self.cache_dir, f"{safe_id}.jpg")
        
        # Check if image is already cached
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path)
            except Exception as e:
                logger.warning(f"Error loading cached image {cache_path}: {e}")
                # If loading fails, try downloading again
                
        # Download the image
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            img = Image.open(BytesIO(response.content))
            
            # Save to cache
            img.save(cache_path)
            
            return img
            
        except Exception as e:
            logger.warning(f"Error downloading image from {url}: {e}")
            return None
    
    def preprocess_images(self, 
                         data_df: pd.DataFrame, 
                         target_size: Tuple[int, int] = (512, 512),
                         batch_size: int = 32,
                         edge_method: str = 'canny',
                         extract_features: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess images by downloading, extracting edges, and calculating features
        
        Args:
            data_df: DataFrame containing image data
            target_size: Size to resize images to
            batch_size: Number of images to process at once
            edge_method: Method for edge detection
            extract_features: Whether to extract line features
            
        Returns:
            Dictionary mapping artwork IDs to dictionaries of processed data
        """
        results = {}
        
        for i in tqdm(range(0, len(data_df), batch_size), desc="Processing images"):
            batch = data_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                artwork_id = row['Artwork']
                url = row['Link']
                style = row['Style']
                
                # Download image
                img = self.download_image(url, artwork_id)
                
                if img is None:
                    continue
                
                # Convert PIL image to numpy array for OpenCV
                img_array = np.array(img)
                
                # Handle grayscale images
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    # Convert RGBA to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                
                try:
                    # Preprocess image and extract edges
                    preprocessed = preprocess_image(img_array, target_size=target_size)
                    edges = extract_edges(preprocessed, method=edge_method)
                    
                    # Create result entry
                    result_entry = {
                        'style': style,
                        'style_id': self.style_to_id[style],
                        'preprocessed': preprocessed,
                        'edges': edges
                    }
                    
                    # Extract line features if requested
                    if extract_features:
                        line_features = extract_line_features(edges)
                        
                        # Ensure all required features have valid numeric values
                        required_features = [
                            'line_count', 
                            'mean_length', 
                            'std_length',
                            'intersection_count'
                        ]
                        
                        # Convert any missing features to numeric defaults
                        for feature in required_features:
                            # If feature is missing or has non-numeric value, set to 0
                            if feature not in line_features or not isinstance(line_features[feature], (int, float)):
                                line_features[feature] = 0
                                
                        result_entry['features'] = line_features
                    
                    results[artwork_id] = result_entry
                    
                except Exception as e:
                    logger.warning(f"Error processing image for {artwork_id}: {e}")
        
        logger.info(f"Processed {len(results)} images successfully")
        return results
    
    def save_processed_data(self, processed_data: Dict[str, Dict[str, Any]], output_dir: str):
        """
        Save processed image data to disk
        
        Args:
            processed_data: Dictionary of processed image data
            output_dir: Directory to save processed data
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        preprocessed_dir = os.path.join(output_dir, 'preprocessed')
        edges_dir = os.path.join(output_dir, 'edges')
        features_dir = os.path.join(output_dir, 'features')
        
        os.makedirs(preprocessed_dir, exist_ok=True)
        os.makedirs(edges_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        
        # Save metadata
        metadata = []
        
        for artwork_id, data in tqdm(processed_data.items(), desc="Saving processed data"):
            # Create safe filename
            safe_id = ''.join(c if c.isalnum() else '_' for c in artwork_id)
            
            # Save preprocessed image
            preprocessed_path = os.path.join(preprocessed_dir, f"{safe_id}.jpg")
            cv2.imwrite(preprocessed_path, data['preprocessed'])
            
            # Save edge image
            edges_path = os.path.join(edges_dir, f"{safe_id}.jpg")
            cv2.imwrite(edges_path, data['edges'])
            
            # Save features if available
            if 'features' in data:
                features_path = os.path.join(features_dir, f"{safe_id}.npy")
                # Make sure all feature values are numeric
                feature_array = np.array([
                    data['features'].get('line_count', 0),
                    data['features'].get('mean_length', 0),  # Changed from avg_line_length
                    data['features'].get('std_length', 0),   # Changed from line_length_std
                    data['features'].get('mean_angle', 0),   # Changed from avg_line_angle
                    data['features'].get('std_angle', 0),    # Changed from line_angle_std
                    data['features'].get('intersection_count', 0)
                ])
                np.save(features_path, feature_array)
            
            # Add to metadata
            metadata.append({
                'artwork_id': artwork_id,
                'style': data['style'],
                'style_id': data['style_id'],
                'preprocessed_path': os.path.relpath(preprocessed_path, output_dir),
                'edges_path': os.path.relpath(edges_path, output_dir),
                'features_path': os.path.relpath(features_path, output_dir) if 'features' in data else None
            })
        
        # Save metadata as CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        
        # Save style mappings
        style_mapping = {
            'style_to_id': self.style_to_id,
            'id_to_style': {str(k): v for k, v in self.id_to_style.items()}  # Convert keys to strings for JSON
        }
        
        with open(os.path.join(output_dir, 'style_mapping.json'), 'w') as f:
            import json
            json.dump(style_mapping, f, indent=2)
            
        logger.info(f"Saved processed data to {output_dir}")


class WikiArtTorchDataset(Dataset):
    """PyTorch Dataset for WikiArt processed data"""
    
    def __init__(self, 
                metadata_path: str,
                data_dir: str,
                transform: Optional[transforms.Compose] = None,
                edge_transform: Optional[transforms.Compose] = None,
                use_features: bool = True):
        """
        Initialize the WikiArt PyTorch Dataset
        
        Args:
            metadata_path: Path to the metadata CSV file
            data_dir: Directory containing processed data
            transform: Transforms to apply to preprocessed images
            edge_transform: Transforms to apply to edge images
            use_features: Whether to use extracted features
        """
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = data_dir
        self.transform = transform
        self.edge_transform = edge_transform
        self.use_features = use_features
        
        # Filter out rows with missing data
        if use_features:
            self.metadata = self.metadata.dropna(subset=['features_path'])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.metadata.iloc[idx]
        
        # Load preprocessed image
        preprocessed_path = os.path.join(self.data_dir, row['preprocessed_path'])
        preprocessed = Image.open(preprocessed_path).convert('RGB')
        
        # Load edge image
        edges_path = os.path.join(self.data_dir, row['edges_path'])
        edges = Image.open(edges_path).convert('L')  # Load as grayscale
        
        # Apply transforms if provided
        if self.transform:
            preprocessed = self.transform(preprocessed)
            
        if self.edge_transform:
            edges = self.edge_transform(edges)
        else:
            # Default transform to tensor
            edges = transforms.ToTensor()(edges)
        
        # Get label
        label = row['style_id']
        
        # Create sample
        sample = {
            'preprocessed': preprocessed,
            'edges': edges,
            'label': label,
            'artwork_id': row['artwork_id']
        }
        
        # Add features if requested
        if self.use_features and not pd.isna(row['features_path']):
            features_path = os.path.join(self.data_dir, row['features_path'])
            features = np.load(features_path)
            sample['features'] = torch.tensor(features, dtype=torch.float32)
        
        return sample


def create_dataloaders(metadata_path: str,
                     data_dir: str,
                     batch_size: int = 32,
                     image_size: int = 224,
                     use_features: bool = True,
                     num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing
    
    Args:
        metadata_path: Path to the metadata CSV file
        data_dir: Directory containing processed data
        batch_size: Batch size for DataLoaders
        image_size: Size to resize images to
        use_features: Whether to use extracted features
        num_workers: Number of workers for DataLoader
        
    Returns:
        Dictionary of DataLoaders
    """
    # Split metadata into train, val, test
    metadata = pd.read_csv(metadata_path)
    
    # 70% train, 15% val, 15% test
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    
    # Group by style to ensure stratified sampling
    train_dfs, val_dfs, test_dfs = [], [], []
    
    np.random.seed(42)
    
    # Process each style separately to maintain class balance
    for style_id, group in metadata.groupby('style_id'):
        # Shuffle the data
        indices = np.random.permutation(len(group))
        group_shuffled = group.iloc[indices].reset_index(drop=True)
        
        # Calculate split sizes
        n_train = int(len(group_shuffled) * train_ratio)
        n_val = int(len(group_shuffled) * val_ratio)
        
        # Split the data
        train_data = group_shuffled.iloc[:n_train]
        val_data = group_shuffled.iloc[n_train:n_train + n_val]
        test_data = group_shuffled.iloc[n_train + n_val:]
        
        train_dfs.append(train_data)
        val_dfs.append(val_data)
        test_dfs.append(test_data)
    
    # Combine all styles
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    
    # Save split metadata
    train_df.to_csv(os.path.join(data_dir, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_metadata.csv'), index=False)
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    edge_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = WikiArtTorchDataset(
        os.path.join(data_dir, 'train_metadata.csv'),
        data_dir,
        transform=train_transform,
        edge_transform=edge_transform,
        use_features=use_features
    )
    
    val_dataset = WikiArtTorchDataset(
        os.path.join(data_dir, 'val_metadata.csv'),
        data_dir,
        transform=val_transform,
        edge_transform=edge_transform,
        use_features=use_features
    )
    
    test_dataset = WikiArtTorchDataset(
        os.path.join(data_dir, 'test_metadata.csv'),
        data_dir,
        transform=val_transform,
        edge_transform=edge_transform,
        use_features=use_features
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 