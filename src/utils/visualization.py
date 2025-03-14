import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import cv2
from typing import List, Dict, Tuple, Any, Optional, Union
import torch


def visualize_edges(original_image: np.ndarray, 
                   edge_image: np.ndarray, 
                   figsize: Tuple[int, int] = (12, 6),
                   save_path: Optional[str] = None) -> Figure:
    """
    Visualize original image alongside its edge detection result
    
    Args:
        original_image: Original artwork image
        edge_image: Edge-detected image
        figsize: Figure size (width, height)
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Show original image
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if len(original_image.shape) == 3 else original_image, 
               cmap='gray' if len(original_image.shape) == 2 else None)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Show edge image
    ax2.imshow(edge_image, cmap='gray')
    ax2.set_title('Edge Detection')
    ax2.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def visualize_line_features(image: np.ndarray, 
                           lines: np.ndarray, 
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None) -> Figure:
    """
    Visualize detected lines on an image
    
    Args:
        image: Input image
        lines: Array of line segments [x1, y1, x2, y2]
        figsize: Figure size (width, height)
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create a copy of the image
    result = image.copy()
    
    # Ensure image is in color format for drawing colored lines
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # Draw lines with different colors based on orientation
    for x1, y1, x2, y2 in lines:
        # Calculate line angle
        angle = np.arctan2(y2 - y1, x2 - x1)
        if angle < 0:
            angle += np.pi
        
        # Determine color based on orientation
        if angle < np.pi/6 or angle > 5*np.pi/6:  # Horizontal
            color = (0, 0, 255)  # Red
        elif np.pi/3 < angle < 2*np.pi/3:  # Vertical
            color = (0, 255, 0)  # Green
        else:  # Diagonal
            color = (255, 0, 0)  # Blue
        
        # Draw the line
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Show original image
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image,
               cmap='gray' if len(image.shape) == 2 else None)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Show image with lines
    ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Detected Lines ({len(lines)} lines)')
    ax2.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Horizontal'),
        Patch(facecolor='green', label='Vertical'),
        Patch(facecolor='blue', label='Diagonal')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def visualize_composition_features(image: np.ndarray, 
                                  composition_features: Dict[str, Any],
                                  figsize: Tuple[int, int] = (15, 10),
                                  save_path: Optional[str] = None) -> Figure:
    """
    Visualize composition features on an image
    
    Args:
        image: Input image
        composition_features: Dictionary of composition features
        figsize: Figure size (width, height)
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show original image
    if len(image.shape) == 3 and image.shape[2] == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap='gray')
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Draw rule of thirds grid
    rule_thirds_color = 'white'
    rule_thirds_alpha = 0.5
    
    # Horizontal lines
    ax.axhline(y=height/3, color=rule_thirds_color, alpha=rule_thirds_alpha, linestyle='--')
    ax.axhline(y=2*height/3, color=rule_thirds_color, alpha=rule_thirds_alpha, linestyle='--')
    
    # Vertical lines
    ax.axvline(x=width/3, color=rule_thirds_color, alpha=rule_thirds_alpha, linestyle='--')
    ax.axvline(x=2*width/3, color=rule_thirds_color, alpha=rule_thirds_alpha, linestyle='--')
    
    # Draw golden ratio grid
    golden_ratio = (1 + np.sqrt(5)) / 2
    golden_color = 'yellow'
    golden_alpha = 0.4
    
    # Horizontal lines
    ax.axhline(y=height/golden_ratio, color=golden_color, alpha=golden_alpha, linestyle='-.')
    ax.axhline(y=height-height/golden_ratio, color=golden_color, alpha=golden_alpha, linestyle='-.')
    
    # Vertical lines
    ax.axvline(x=width/golden_ratio, color=golden_color, alpha=golden_alpha, linestyle='-.')
    ax.axvline(x=width-width/golden_ratio, color=golden_color, alpha=golden_alpha, linestyle='-.')
    
    # If focal points are available, draw them
    if 'focal_points_info' in composition_features and 'focal_points' in composition_features['focal_points_info']:
        focal_points = composition_features['focal_points_info']['focal_points']
        
        for i, (x, y, val) in enumerate(focal_points):
            # Convert normalized coordinates to pixel coordinates
            px = int(x * width)
            py = int(y * height)
            
            # Draw point
            circle = plt.Circle((px, py), radius=min(width, height)*0.03, 
                               color='red', alpha=0.7)
            ax.add_patch(circle)
            
            # Add label
            ax.text(px + 10, py + 10, f"F{i+1}", color='white', 
                   backgroundcolor='black', fontsize=12)
    
    # Add symmetry axis if symmetry is high
    if 'horizontal_symmetry' in composition_features and composition_features['horizontal_symmetry'] > 0.7:
        ax.axvline(x=width/2, color='cyan', alpha=0.6, linestyle='-', linewidth=2)
    
    if 'vertical_symmetry' in composition_features and composition_features['vertical_symmetry'] > 0.7:
        ax.axhline(y=height/2, color='cyan', alpha=0.6, linestyle='-', linewidth=2)
    
    # Add title with key metrics
    title_parts = []
    
    if 'thirds_adherence' in composition_features:
        title_parts.append(f"Rule of Thirds: {composition_features['thirds_adherence']:.2f}")
    
    if 'overall_symmetry' in composition_features:
        title_parts.append(f"Symmetry: {composition_features['overall_symmetry']:.2f}")
    
    if 'golden_ratio_adherence' in composition_features:
        title_parts.append(f"Golden Ratio: {composition_features['golden_ratio_adherence']:.2f}")
    
    if 'horizontal_balance' in composition_features and 'vertical_balance' in composition_features:
        balance = (composition_features['horizontal_balance'] + composition_features['vertical_balance']) / 2
        title_parts.append(f"Balance: {balance:.2f}")
    
    ax.set_title(' | '.join(title_parts))
    ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=rule_thirds_color, linestyle='--', label='Rule of Thirds'),
        Line2D([0], [0], color=golden_color, linestyle='-.', label='Golden Ratio'),
        Line2D([0], [0], color='cyan', linestyle='-', label='Symmetry Axis'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Focal Points')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def visualize_similar_artworks(query_image: np.ndarray,
                              similar_images: List[np.ndarray],
                              similar_artists: List[str],
                              similar_titles: List[str],
                              similarities: List[float],
                              figsize: Tuple[int, int] = (15, 10),
                              save_path: Optional[str] = None) -> Figure:
    """
    Visualize query image and its most similar artworks
    
    Args:
        query_image: Query artwork image
        similar_images: List of similar artwork images
        similar_artists: List of similar artwork artists
        similar_titles: List of similar artwork titles
        similarities: List of similarity scores
        figsize: Figure size (width, height)
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    n_similar = len(similar_images)
    
    # Create figure grid
    fig = plt.figure(figsize=figsize)
    
    # Add query image at a larger size
    ax_query = plt.subplot2grid((3, n_similar), (0, 0), colspan=n_similar, rowspan=1)
    
    if len(query_image.shape) == 3 and query_image.shape[2] == 3:
        ax_query.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    else:
        ax_query.imshow(query_image, cmap='gray')
    
    ax_query.set_title('Query Image', fontsize=14)
    ax_query.axis('off')
    
    # Add similar images
    for i in range(n_similar):
        ax = plt.subplot2grid((3, n_similar), (1, i), rowspan=2)
        
        if len(similar_images[i].shape) == 3 and similar_images[i].shape[2] == 3:
            ax.imshow(cv2.cvtColor(similar_images[i], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(similar_images[i], cmap='gray')
        
        # Format title with artist, title, and similarity score
        title = f"{similar_artists[i]}\n{similar_titles[i]}\nSimilarity: {similarities[i]:.2f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def visualize_embedding_space(embeddings: np.ndarray, 
                             labels: List[int],
                             artist_names: Optional[List[str]] = None,
                             n_components: int = 2,
                             figsize: Tuple[int, int] = (12, 10),
                             save_path: Optional[str] = None) -> Figure:
    """
    Visualize embedding space using dimensionality reduction
    
    Args:
        embeddings: Embedding vectors (n_samples, embedding_dim)
        labels: Artist labels
        artist_names: Optional list of artist names for labels
        n_components: Number of components for dimensionality reduction (2 or 3)
        figsize: Figure size (width, height)
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()
    
    # Apply dimensionality reduction
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create figure
    if n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels
    unique_labels = sorted(set(labels))
    
    # Create colormap
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    n_artists = len(unique_labels)
    cmap = ListedColormap(cm.tab20.colors) if n_artists <= 20 else ListedColormap(cm.nipy_spectral.colors)
    
    # Plot each artist's embeddings
    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        points = reduced_embeddings[mask]
        
        if n_components == 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=[cmap(i / n_artists)], label=artist_names[label] if artist_names else f"Artist {label}")
        else:
            ax.scatter(points[:, 0], points[:, 1], 
                      c=[cmap(i / n_artists)], label=artist_names[label] if artist_names else f"Artist {label}")
    
    # Set labels
    if n_components == 3:
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
    else:
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
    
    # Set title
    ax.set_title(f"{n_components}D t-SNE Visualization of Art Style Embeddings")
    
    # Add legend (with smaller font for many artists)
    if n_artists > 10:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig 