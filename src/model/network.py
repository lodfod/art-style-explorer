import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class FeatureExtractor(nn.Module):
    """Neural network for extracting features from line work images"""
    
    def __init__(self, input_channels: int = 1, feature_dim: int = 128):
        """
        Initialize the feature extractor
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for color)
            feature_dim: Dimension of the output feature vector
        """
        super(FeatureExtractor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate output size of convolutional layers for a 512x512 input
        self._conv_output_size = self._calculate_conv_output_size(input_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_conv_output_size(self, input_channels: int) -> int:
        """
        Calculate the output size of the convolutional layers
        
        Args:
            input_channels: Number of input channels
            
        Returns:
            Size of the flattened output from convolutional layers
        """
        # Create a dummy input tensor
        x = torch.zeros(1, input_channels, 512, 512)
        
        # Forward pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Get the flattened size
        return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            Feature vector of shape (batch_size, feature_dim)
        """
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Normalize the feature vector
        x = F.normalize(x, p=2, dim=1)
        
        return x


class CompositionEncoder(nn.Module):
    """Neural network for encoding composition features"""
    
    def __init__(self, input_dim: int = 30, feature_dim: int = 64):
        """
        Initialize the composition encoder
        
        Args:
            input_dim: Dimension of the input composition feature vector
            feature_dim: Dimension of the output feature vector
        """
        super(CompositionEncoder, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Feature vector of shape (batch_size, feature_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Normalize the feature vector
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ArtStyleEmbedder(nn.Module):
    """Neural network for combining line features and composition features"""
    
    def __init__(self, 
                line_feature_dim: int = 128, 
                comp_feature_dim: int = 64,
                output_dim: int = 256):
        """
        Initialize the art style embedder
        
        Args:
            line_feature_dim: Dimension of the line feature vector
            comp_feature_dim: Dimension of the composition feature vector
            output_dim: Dimension of the output embedding vector
        """
        super(ArtStyleEmbedder, self).__init__()
        
        # Concatenated input dimension
        concat_dim = line_feature_dim + comp_feature_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(concat_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, 
               line_features: torch.Tensor, 
               comp_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            line_features: Line features tensor of shape (batch_size, line_feature_dim)
            comp_features: Composition features tensor of shape (batch_size, comp_feature_dim)
            
        Returns:
            Embedding vector of shape (batch_size, output_dim)
        """
        # Concatenate features
        x = torch.cat([line_features, comp_features], dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Normalize the embedding vector
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ArtStyleClassifier(nn.Module):
    """Neural network for classifying art styles"""
    
    def __init__(self, 
                embedding_dim: int = 256, 
                num_classes: int = 100):
        """
        Initialize the art style classifier
        
        Args:
            embedding_dim: Dimension of the input embedding vector
            num_classes: Number of art style classes
        """
        super(ArtStyleClassifier, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input embedding tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ArtStyleNetwork(nn.Module):
    """Full network for art style analysis"""
    
    def __init__(self, 
                input_channels: int = 1, 
                line_feature_dim: int = 128, 
                comp_feature_dim: int = 64,
                embedding_dim: int = 256,
                num_classes: int = 100,
                comp_input_dim: int = 30):
        """
        Initialize the full art style network
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for color)
            line_feature_dim: Dimension of the line feature vector
            comp_feature_dim: Dimension of the composition feature vector
            embedding_dim: Dimension of the art style embedding vector
            num_classes: Number of art style classes
            comp_input_dim: Dimension of the input composition feature vector
        """
        super(ArtStyleNetwork, self).__init__()
        
        # Initialize sub-networks
        self.feature_extractor = FeatureExtractor(input_channels, line_feature_dim)
        self.composition_encoder = CompositionEncoder(comp_input_dim, comp_feature_dim)
        self.style_embedder = ArtStyleEmbedder(line_feature_dim, comp_feature_dim, embedding_dim)
        self.style_classifier = ArtStyleClassifier(embedding_dim, num_classes)
        
    def extract_features(self, 
                        image: torch.Tensor, 
                        composition_features: torch.Tensor) -> torch.Tensor:
        """
        Extract art style embedding
        
        Args:
            image: Input image tensor of shape (batch_size, input_channels, height, width)
            composition_features: Composition features tensor of shape (batch_size, comp_input_dim)
            
        Returns:
            Embedding vector of shape (batch_size, embedding_dim)
        """
        # Extract line features
        line_features = self.feature_extractor(image)
        
        # Encode composition features
        comp_features = self.composition_encoder(composition_features)
        
        # Generate embedding
        embedding = self.style_embedder(line_features, comp_features)
        
        return embedding
    
    def forward(self, 
               image: torch.Tensor, 
               composition_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            image: Input image tensor of shape (batch_size, input_channels, height, width)
            composition_features: Composition features tensor of shape (batch_size, comp_input_dim)
            
        Returns:
            Tuple of (embedding, class_logits)
        """
        # Extract embedding
        embedding = self.extract_features(image, composition_features)
        
        # Classify art style
        logits = self.style_classifier(embedding)
        
        return embedding, logits


def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity between two embedding vectors
    
    Args:
        embedding1: First embedding tensor of shape (batch_size, embedding_dim)
        embedding2: Second embedding tensor of shape (batch_size, embedding_dim)
        
    Returns:
        Cosine similarity of shape (batch_size,)
    """
    return F.cosine_similarity(embedding1, embedding2, dim=1)


def find_similar_artworks(query_embedding: torch.Tensor, 
                         database_embeddings: torch.Tensor, 
                         top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the most similar artworks from a database
    
    Args:
        query_embedding: Query embedding tensor of shape (1, embedding_dim)
        database_embeddings: Database embeddings tensor of shape (num_samples, embedding_dim)
        top_k: Number of similar artworks to retrieve
        
    Returns:
        Tuple of (indices, similarities) for the top-k similar artworks
    """
    # Calculate cosine similarity between query and all database embeddings
    similarities = F.cosine_similarity(query_embedding, database_embeddings)
    
    # Get top-k similarities and indices
    top_similarities, top_indices = torch.topk(similarities, k=top_k)
    
    return top_indices, top_similarities 