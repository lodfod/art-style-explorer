#!/usr/bin/env python3
"""
Script to train a neural network for art style classification based on extracted features.
Includes enhanced regularization and feature selection/reduction techniques.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Sklearn imports for evaluation and feature selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class ArtStyleDataset(Dataset):
    """Dataset class for art style classification."""
    
    def __init__(self, features, labels, transform=None):
        """Initialize the dataset."""
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

class ArtStyleClassifier(nn.Module):
    """Neural network for art style classification with enhanced regularization."""
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.5, 
                 use_batch_norm=True, weight_decay=1e-5, alpha=0.2):
        """
        Initialize the neural network with enhanced regularization.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            weight_decay: L2 regularization strength
            alpha: LeakyReLU negative slope
        """
        super(ArtStyleClassifier, self).__init__()
        
        # Create a list of layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.LeakyReLU(alpha))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.LeakyReLU(alpha))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Store weight decay for L2 regularization
        self.weight_decay = weight_decay
    
    def _init_weights(self, m):
        """Initialize weights using He initialization."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def l2_regularization(self):
        """Calculate L2 regularization loss."""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.weight_decay * l2_loss

def load_data(data_dir):
    """Load train/test splits and label mapping."""
    # Load splits
    with open(os.path.join(data_dir, 'train_test_splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    
    # Load label mapping
    with open(os.path.join(data_dir, 'label_mapping.pkl'), 'rb') as f:
        label_mapping = pickle.load(f)
    
    return splits, label_mapping

def apply_feature_selection(X_train, y_train, X_test, method='anova', n_features=None, variance_threshold=0.95):
    """
    Apply feature selection or dimensionality reduction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        method: Feature selection method ('anova', 'mutual_info', 'pca', 'none')
        n_features: Number of features to select (if None, use variance_threshold for PCA)
        variance_threshold: Variance threshold for PCA (only used if method='pca' and n_features=None)
    
    Returns:
        X_train_selected: Selected training features
        X_test_selected: Selected test features
        selector: Feature selector object
    """
    if method == 'none':
        return X_train, X_test, None
    
    if method == 'anova':
        # Select features based on ANOVA F-value
        if n_features is None:
            n_features = min(X_train.shape[1] // 2, 500)  # Default to half of features or 500, whichever is smaller
        
        selector = SelectKBest(f_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        print(f"Selected {len(selected_indices)} features using ANOVA F-test")
        
    elif method == 'mutual_info':
        # Select features based on mutual information
        if n_features is None:
            n_features = min(X_train.shape[1] // 2, 500)  # Default to half of features or 500, whichever is smaller
        
        selector = SelectKBest(mutual_info_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        print(f"Selected {len(selected_indices)} features using Mutual Information")
        
    elif method == 'pca':
        # Apply PCA for dimensionality reduction
        if n_features is None:
            # Use variance threshold to determine number of components
            pca_full = PCA().fit(X_train)
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_features = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        selector = PCA(n_components=n_features)
        X_train_selected = selector.fit_transform(X_train)
        X_test_selected = selector.transform(X_test)
        
        # Print variance explained
        explained_variance = sum(selector.explained_variance_ratio_)
        print(f"PCA with {n_features} components explains {explained_variance:.2%} of variance")
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    return X_train_selected, X_test_selected, selector

def preprocess_data(splits, feature_selection_method='none', n_features=None, variance_threshold=0.95):
    """Preprocess data for model training with optional feature selection."""
    # Extract features and labels
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_test = splits['X_test']
    y_test = splits['y_test']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply feature selection if requested
    if feature_selection_method != 'none':
        X_train_selected, X_test_selected, selector = apply_feature_selection(
            X_train_scaled, y_train, X_test_scaled, 
            method=feature_selection_method, 
            n_features=n_features,
            variance_threshold=variance_threshold
        )
    else:
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
        selector = None
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler, selector

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders for training and testing."""
    # Create datasets
    train_dataset = ArtStyleDataset(X_train, y_train)
    test_dataset = ArtStyleDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                device, num_epochs=100, early_stopping_patience=10, 
                use_l2_reg=True, mixup_alpha=0.2, use_mixup=False):
    """
    Train the neural network model with enhanced regularization techniques.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        use_l2_reg: Whether to use L2 regularization
        mixup_alpha: Alpha parameter for mixup
        use_mixup: Whether to use mixup data augmentation
    """
    # Initialize variables for training
    best_test_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Move model to device
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Apply mixup if enabled
            if use_mixup and mixup_alpha > 0:
                features, labels_a, labels_b, lam = mixup_data(features, labels, mixup_alpha, device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features)
                
                # Mixup loss
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                # Add L2 regularization if enabled
                if use_l2_reg:
                    loss += model.l2_regularization()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics (use original labels for accuracy)
                train_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                # For mixup, we approximate accuracy using the dominant label
                train_correct += (predicted == labels).sum().item()
            else:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Add L2 regularization if enabled
                if use_l2_reg:
                    loss += model.l2_regularization()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Track statistics
                test_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # Calculate average test loss and accuracy
        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Update learning rate scheduler
        scheduler.step(test_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        # Check for improvement
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

def mixup_data(x, y, alpha=0.2, device='cpu'):
    """
    Applies mixup augmentation to the batch.
    
    Args:
        x: Input features
        y: Input labels
        alpha: Mixup alpha parameter
        device: Device to use
    
    Returns:
        mixed_x: Mixed features
        y_a: First set of labels
        y_b: Second set of labels
        lam: Lambda value for mixing
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Applies mixup criterion to the predictions.
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Lambda value for mixing
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate_model(model, X_test, y_test, label_mapping, device):
    """Evaluate the trained model."""
    # Move model to device
    model.to(device)
    model.eval()
    
    # Convert data to tensors and move to device
    X_test = X_test.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
    
    # Convert tensors to numpy arrays
    y_true = y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate classification report
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, report, cm, class_names

def plot_training_history(history, output_dir):
    """Plot training and validation loss and accuracy."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['test_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train Accuracy')
    plt.plot(history['test_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_model(model, scaler, label_mapping, output_dir, feature_selector=None):
    """Save the trained model and associated metadata."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    
    # Save model architecture
    with open(os.path.join(output_dir, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))
    
    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save label mapping
    with open(os.path.join(output_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    
    # Save feature selector if available
    if feature_selector is not None:
        with open(os.path.join(output_dir, 'feature_selector.pkl'), 'wb') as f:
            pickle.dump(feature_selector, f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a neural network for art style classification')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing train/test splits')
    parser.add_argument('--output-dir', type=str, default='results/model',
                        help='Directory to save model and results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden-sizes', type=str, default='1024,512,256',
                        help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='Dropout rate for regularization')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization) strength')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--feature-selection', type=str, default='none',
                        choices=['none', 'anova', 'mutual_info', 'pca'],
                        help='Feature selection method to use')
    parser.add_argument('--n-features', type=int, default=None,
                        help='Number of features to select (if None, use variance threshold for PCA)')
    parser.add_argument('--variance-threshold', type=float, default=0.95,
                        help='Variance threshold for PCA (only used if feature-selection=pca and n-features=None)')
    parser.add_argument('--use-batch-norm', action='store_true',
                        help='Use batch normalization')
    parser.add_argument('--use-mixup', action='store_true',
                        help='Use mixup data augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Alpha parameter for mixup')
    return parser.parse_args()

def main():
    """Main function to train the model."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    splits, label_mapping = load_data(args.data_dir)
    
    # Preprocess data with feature selection
    print("Preprocessing data...")
    X_train, y_train, X_test, y_test, scaler, feature_selector = preprocess_data(
        splits, 
        feature_selection_method=args.feature_selection,
        n_features=args.n_features,
        variance_threshold=args.variance_threshold
    )
    
    # Create dataloaders
    print(f"Creating dataloaders with batch size {args.batch_size}...")
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, args.batch_size)
    
    # Parse hidden sizes
    hidden_sizes = [int(size) for size in args.hidden_sizes.split(',')]
    
    # Create model with enhanced regularization
    print(f"Creating model with hidden sizes {hidden_sizes}...")
    input_size = X_train.shape[1]
    num_classes = len(label_mapping)
    model = ArtStyleClassifier(
        input_size, 
        hidden_sizes, 
        num_classes, 
        dropout_rate=args.dropout_rate,
        use_batch_norm=args.use_batch_norm,
        weight_decay=args.weight_decay
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)  # We handle L2 reg manually
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train model with enhanced regularization
    print(f"Training model for {args.num_epochs} epochs...")
    start_time = time.time()
    model, history = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, 
        device, args.num_epochs, args.early_stopping_patience,
        use_l2_reg=True, mixup_alpha=args.mixup_alpha, use_mixup=args.use_mixup
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, report, cm, class_names = evaluate_model(model, X_test, y_test, label_mapping, device)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"model_{timestamp}")
    
    # Plot training history
    print(f"Plotting training history to {output_dir}...")
    plot_training_history(history, output_dir)
    
    # Plot confusion matrix
    print(f"Plotting confusion matrix to {output_dir}...")
    plot_confusion_matrix(cm, class_names, output_dir)
    
    # Save model
    print(f"Saving model to {output_dir}...")
    save_model(model, scaler, label_mapping, output_dir, feature_selector)
    
    # Save evaluation results
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nModel Architecture:\n")
        f.write(str(model))
        f.write("\n\nTraining Parameters:\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Hidden sizes: {hidden_sizes}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Dropout rate: {args.dropout_rate}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Feature selection: {args.feature_selection}\n")
        if args.feature_selection != 'none':
            if args.feature_selection == 'pca':
                f.write(f"PCA variance threshold: {args.variance_threshold}\n")
            if args.n_features is not None:
                f.write(f"Number of selected features: {args.n_features}\n")
        f.write(f"Batch normalization: {args.use_batch_norm}\n")
        f.write(f"Mixup augmentation: {args.use_mixup}\n")
        if args.use_mixup:
            f.write(f"Mixup alpha: {args.mixup_alpha}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Early stopping patience: {args.early_stopping_patience}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
    
    print(f"Model training and evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 