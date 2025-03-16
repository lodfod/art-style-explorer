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
from torch.cuda.amp import autocast

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


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, 
               weight_decay=1e-5, device='cuda', checkpoint_dir=None, 
               use_mixed_precision=False, scaler=None):
    """
    Train the art style model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to use for training
        checkpoint_dir: Directory to save checkpoints
        use_mixed_precision: Whether to use mixed precision training (faster on newer GPUs)
        scaler: GradScaler for mixed precision training
        
    Returns:
        Dictionary of training metrics
    """
    # Set device
    device = torch.device(device)
    model = model.to(device)
    
    # Define loss functions
    classification_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize metrics
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': [],
        'best_val_accuracy': 0.0,
        'best_epoch': 0
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, features, labels in train_bar:
            # Move data to device
            images = images.to(device)
            if features is not None:
                features = features.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Mixed precision training
            if use_mixed_precision:
                with autocast():
                    # Forward pass
                    embeddings, logits = model(images, features)
                    
                    # Calculate classification loss
                    class_loss = classification_loss(logits, labels)
                    
                    # Calculate contrastive loss
                    contrast_loss = contrastive_loss(embeddings, labels)
                    
                    # Combined loss
                    loss = class_loss + 0.5 * contrast_loss
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                embeddings, logits = model(images, features)
                
                # Calculate classification loss
                class_loss = classification_loss(logits, labels)
                
                # Calculate contrastive loss
                contrast_loss = contrastive_loss(embeddings, labels)
                
                # Combined loss
                loss = class_loss + 0.5 * contrast_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_accuracy = 100.0 * train_correct / train_total
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_accuracy:.2f}%"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Create progress bar
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for images, features, labels in val_bar:
                # Move data to device
                images = images.to(device)
                if features is not None:
                    features = features.to(device)
                labels = labels.to(device)
                
                # Mixed precision inference (though less critical for validation)
                if use_mixed_precision:
                    with autocast():
                        # Forward pass
                        embeddings, logits = model(images, features)
                        
                        # Calculate classification loss
                        class_loss = classification_loss(logits, labels)
                        
                        # Calculate contrastive loss
                        contrast_loss = contrastive_loss(embeddings, labels)
                        
                        # Combined loss
                        loss = class_loss + 0.5 * contrast_loss
                else:
                    # Forward pass
                    embeddings, logits = model(images, features)
                    
                    # Calculate classification loss
                    class_loss = classification_loss(logits, labels)
                    
                    # Calculate contrastive loss
                    contrast_loss = contrastive_loss(embeddings, labels)
                    
                    # Combined loss
                    loss = class_loss + 0.5 * contrast_loss
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_accuracy = 100.0 * val_correct / val_total
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{val_accuracy:.2f}%"
                })
        
        # Calculate metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_accuracy = 100.0 * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100.0 * val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_accuracy)
        
        # Update metrics
        metrics['train_loss'].append(epoch_train_loss)
        metrics['train_accuracy'].append(epoch_train_accuracy)
        metrics['val_loss'].append(epoch_val_loss)
        metrics['val_accuracy'].append(epoch_val_accuracy)
        metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if epoch_val_accuracy > metrics['best_val_accuracy']:
            metrics['best_val_accuracy'] = epoch_val_accuracy
            metrics['best_epoch'] = epoch
            
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
            
            # Also save final model
            if epoch == num_epochs - 1:
                final_path = os.path.join(checkpoint_dir, 'model_final.pth')
                torch.save(model.state_dict(), final_path)
    
    # Return training metrics
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
    ax2.plot(epochs, metrics['train_accuracy'], 'b-', label='Training Accuracy')
    if 'val_accuracy' in metrics and metrics['val_accuracy']:
        ax2.plot(epochs, metrics['val_accuracy'], 'r-', label='Validation Accuracy')
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