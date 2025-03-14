import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt

from .network import ArtStyleNetwork


class ArtworkDataset(Dataset):
    """Dataset for artwork images and features"""
    
    def __init__(self, 
                image_paths: List[str],
                artist_labels: List[int],
                transform: Optional[Callable] = None,
                feature_path: Optional[str] = None,
                extract_features_fn: Optional[Callable] = None):
        """
        Initialize the artwork dataset
        
        Args:
            image_paths: List of paths to artwork images
            artist_labels: List of artist labels (integer indices)
            transform: Optional transform to apply to images
            feature_path: Optional path to pre-extracted features
            extract_features_fn: Optional function to extract features on-the-fly
        """
        self.image_paths = image_paths
        self.artist_labels = artist_labels
        self.transform = transform
        
        # Load pre-extracted features if available
        self.features = None
        if feature_path and os.path.exists(feature_path):
            self.features = np.load(feature_path, allow_pickle=True).item()
        
        self.extract_features_fn = extract_features_fn
        
    def __len__(self) -> int:
        """Get the number of samples in the dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        image_path = self.image_paths[idx]
        artist_label = self.artist_labels[idx]
        
        # Load image
        image = plt.imread(image_path)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # Apply transform if available
        if self.transform:
            image = self.transform(image)
        
        # Get composition features
        composition_features = None
        
        if self.features is not None:
            # Load pre-extracted features
            composition_features = self.features.get(image_path, None)
        
        if composition_features is None and self.extract_features_fn:
            # Extract features on-the-fly
            composition_features = self.extract_features_fn(image_path)
        
        # Default to zeros if features are still None
        if composition_features is None:
            composition_features = np.zeros(30, dtype=np.float32)  # Assuming 30-dim composition features
        
        # Convert to tensor if not already
        if not isinstance(composition_features, torch.Tensor):
            composition_features = torch.tensor(composition_features, dtype=torch.float32)
        
        return {
            'image': image,
            'composition_features': composition_features,
            'artist_label': artist_label,
            'image_path': image_path
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning style similarity
    
    This loss encourages similar styles to have similar embeddings
    and dissimilar styles to have different embeddings.
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.1):
        """
        Initialize the contrastive loss
        
        Args:
            margin: Margin for the contrastive loss
            temperature: Temperature parameter for the softmax
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, 
               embeddings: torch.Tensor, 
               labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the contrastive loss
        
        Args:
            embeddings: Batch of embeddings (batch_size, embedding_dim)
            labels: Batch of labels (batch_size,)
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create a mask for positive pairs (same artist/style)
        mask_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Remove diagonal elements (self-similarity)
        mask_diagonal = torch.eye(labels.size(0), device=labels.device).bool()
        mask_positive = mask_positive & ~mask_diagonal
        
        # Calculate loss for each row (anchor sample)
        loss = 0.0
        for i in range(labels.size(0)):
            if mask_positive[i].sum() > 0:  # If there are positive pairs
                # Positive pairs: maximize similarity
                positive_similarities = similarity_matrix[i][mask_positive[i]]
                
                # Negative pairs: minimize similarity
                negative_similarities = similarity_matrix[i][~mask_positive[i] & ~mask_diagonal[i]]
                
                # Combine positive and negative similarities into a softmax
                logits = torch.cat([positive_similarities, negative_similarities])
                targets = torch.zeros(logits.size(0), device=logits.device)
                targets[:positive_similarities.size(0)] = 1.0
                
                # Cross-entropy loss
                row_loss = nn.functional.cross_entropy(logits.unsqueeze(0), targets.unsqueeze(0))
                loss += row_loss
        
        # Average loss over batch
        if labels.size(0) > 0:
            loss = loss / labels.size(0)
        
        return loss


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: Optional[DataLoader] = None,
               num_epochs: int = 50,
               learning_rate: float = 1e-4,
               weight_decay: float = 1e-5,
               device: str = 'cuda',
               checkpoint_dir: str = 'checkpoints',
               log_interval: int = 10) -> Dict[str, List[float]]:
    """
    Train the art style network
    
    Args:
        model: The art style network model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        device: Device to use for training ('cuda' or 'cpu')
        checkpoint_dir: Directory to save model checkpoints
        log_interval: Logging interval (in batches)
        
    Returns:
        Dictionary of training and validation metrics
    """
    # Make sure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model and move to device
    model = model.to(device)
    
    # Define loss functions
    classification_loss_fn = nn.CrossEntropyLoss()
    contrastive_loss_fn = ContrastiveLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize metrics
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(device)
            comp_features = batch['composition_features'].to(device)
            labels = batch['artist_label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            embeddings, logits = model(images, comp_features)
            
            # Calculate losses
            class_loss = classification_loss_fn(logits, labels)
            contrast_loss = contrastive_loss_fn(embeddings, labels)
            
            # Combined loss (weighted sum)
            loss = class_loss + 0.5 * contrast_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total
                })
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Progress bar for validation
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(pbar):
                    # Get data
                    images = batch['image'].to(device)
                    comp_features = batch['composition_features'].to(device)
                    labels = batch['artist_label'].to(device)
                    
                    # Forward pass
                    embeddings, logits = model(images, comp_features)
                    
                    # Calculate losses
                    class_loss = classification_loss_fn(logits, labels)
                    contrast_loss = contrastive_loss_fn(embeddings, labels)
                    
                    # Combined loss
                    loss = class_loss + 0.5 * contrast_loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    if batch_idx % log_interval == 0:
                        pbar.set_postfix({
                            'loss': val_loss / (batch_idx + 1),
                            'acc': 100. * val_correct / val_total
                        })
            
            # Calculate epoch metrics
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}/{num_epochs} - '
                 f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                 f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        else:
            # Print epoch summary (train only)
            print(f'Epoch {epoch+1}/{num_epochs} - '
                 f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if val_loader else None,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'model_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': metrics['train_loss'][-1],
        'val_loss': metrics['val_loss'][-1] if val_loader else None,
    }, final_path)
    print(f'Final model saved to {final_path}')
    
    return metrics


def visualize_training(metrics: Dict[str, List[float]], 
                     save_path: Optional[str] = None) -> None:
    """
    Visualize training and validation metrics
    
    Args:
        metrics: Dictionary of training and validation metrics
        save_path: Optional path to save the visualization
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    epochs = range(1, len(metrics['train_loss']) + 1)
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in metrics and metrics['val_loss']:
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
    if 'val_acc' in metrics and metrics['val_acc']:
        ax2.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f'Training visualization saved to {save_path}')
    
    # Show plot
    plt.show()


def load_model(model: nn.Module, 
              checkpoint_path: str, 
              device: str = 'cuda') -> nn.Module:
    """
    Load model weights from checkpoint
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded from {checkpoint_path} (epoch {checkpoint["epoch"]})')
    
    return model 