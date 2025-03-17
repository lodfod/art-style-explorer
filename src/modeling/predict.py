#!/usr/bin/env python3
"""
Script to make predictions with the trained model on new artwork images.
Supports both individual style classification and the 6 main classifications.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import sys
import re

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from src.preprocessing.feature_extractor import ArtworkFeatureExtractor
from src.modeling.train_model import ArtStyleClassifier

# Define style classifications and their constituent styles
STYLE_CLASSIFICATIONS = {
 "Impressionist_and_Post_Impressionist": [
        "Impressionism", 
        "Post-Impressionism"
    ],
    "Graphic_and_Pattern_Based": [
        "Ukiyo-e", 
        "Pop-Art", 
        "Art-Nouveau-(Modern)"
    ],
    "Geometric_and_Abstract": [
        "Cubism", 
        "Abstract-Art"
    ],
    "Expressive_and_Emotional": [
        "Expressionism", 
        "Surrealism"
    ],
    "Figurative_Traditional": [
        "Early-Renaissance", 
        "High-Renaissance", 
        "Neoclassicism", 
        "Classicism"
    ],
    "Decorative_and_Ornamental": [
        "Romanesque", 
        "Baroque"
    ]
}


# for all predicted outputs, map the classifications to the following:

REMAP_CLASSIFICATIONS_FOR_OUTPUTS = {
 "Impressionist_and_Post_Impressionist": "Impressionist",
    "Graphic_and_Pattern_Based": "Graphical",
    "Geometric_and_Abstract": "Geometric/Abstract",
    "Expressive_and_Emotional": "Expressionism/Surrealism",
    "Figurative_Traditional": "Traditional/Classical",
    "Decorative_and_Ornamental": "Romanesque/Baroque"
}
# Create a mapping from individual style to its classification
STYLE_TO_CLASSIFICATION = {}
for classification, styles in STYLE_CLASSIFICATIONS.items():
    for style in styles:
        STYLE_TO_CLASSIFICATION[style] = classification

def load_model(model_dir):
    """Load the trained model and associated metadata."""
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist")
    
    print(f"Looking for model files in: {model_dir}")
    print(f"Available files: {os.listdir(model_dir)}")
    
    # Load label mapping
    label_mapping_path = os.path.join(model_dir, 'label_mapping.pkl')
    if not os.path.exists(label_mapping_path):
        # Try CSV as fallback
        csv_path = os.path.join(model_dir, 'label_mapping.csv')
        if os.path.exists(csv_path):
            print(f"Using label_mapping.csv instead of label_mapping.pkl")
            label_df = pd.read_csv(csv_path)
            label_mapping = {row['index']: row['label'] for _, row in label_df.iterrows()}
        else:
            raise FileNotFoundError(f"Label mapping file not found in {model_dir}")
    else:
        with open(label_mapping_path, 'rb') as f:
            label_mapping = pickle.load(f)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"Warning: Scaler file not found at {scaler_path}")
        print("Using a default StandardScaler instead. This may affect prediction accuracy.")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Load model architecture
    architecture_path = os.path.join(model_dir, 'model_architecture.txt')
    if not os.path.exists(architecture_path):
        print(f"Warning: Model architecture file not found at {architecture_path}")
        print("Using default architecture. This may not match the trained model.")
        # Default architecture from train_model.py
        input_size = 512  # Default feature size
        hidden_sizes = [1024, 512, 256]  # Default hidden layer sizes from train_model.py
        use_batch_norm = True
        dropout_rate = 0.5
    else:
        with open(architecture_path, 'r') as f:
            architecture = f.read()
        
        # Parse input size and hidden sizes from architecture
        input_size = None
        hidden_sizes = []
        dropout_rate = 0.5  # Default
        use_batch_norm = False
        
        # Check for batch normalization
        if "BatchNorm" in architecture:
            use_batch_norm = True
        
        # Extract dropout rate if present
        dropout_match = re.search(r"Dropout\(p=([0-9.]+)", architecture)
        if dropout_match:
            dropout_rate = float(dropout_match.group(1))
        
        # Parse linear layers
        linear_layers = re.findall(r"Linear\(in_features=(\d+), out_features=(\d+)", architecture)
        if linear_layers:
            # First layer gives us input size
            input_size = int(linear_layers[0][0])
            
            # All but the last layer give us hidden sizes
            hidden_sizes = [int(layer[1]) for layer in linear_layers[:-1]]
            
            # Last layer gives us number of classes
            num_classes = int(linear_layers[-1][1])
            
            print(f"Parsed architecture: input_size={input_size}, hidden_sizes={hidden_sizes}, num_classes={num_classes}")
        else:
            print("Warning: Could not parse Linear layers from architecture file.")
            input_size = 512
            hidden_sizes = [1024, 512, 256]
    
    # If we couldn't parse the architecture or hidden_sizes is empty, use defaults
    if input_size is None:
        print("Warning: Could not parse input size from architecture file. Using default value.")
        input_size = 512
    
    if not hidden_sizes:
        print("Warning: Could not parse hidden sizes from architecture file. Using default values from train_model.py.")
        hidden_sizes = [1024, 512, 256]
    
    print(f"Using model architecture: input_size={input_size}, hidden_sizes={hidden_sizes}, dropout_rate={dropout_rate}, use_batch_norm={use_batch_norm}")
    
    # Create model
    num_classes = len(label_mapping)
    model = ArtStyleClassifier(
        input_size, 
        hidden_sizes, 
        num_classes,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )
    
    # Load model weights
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Determine if the model was trained on classifications or individual styles
    # Check if any of the labels match our classification names
    is_classification_model = any(label in STYLE_CLASSIFICATIONS.keys() for label in label_mapping.values())
    
    return model, scaler, label_mapping, is_classification_model

def extract_features_from_image(image_path, feature_type='combined'):
    """Extract features from a single image."""
    try:
        extractor = ArtworkFeatureExtractor()
        extractor.load_image(image_path)
        features = extractor.extract_all_features()
        return features[feature_type]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def predict_style(model, scaler, label_mapping, features, device, is_classification_model=False):
    """Predict art style from features."""
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Convert to tensor
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Get predicted class and probability
    predicted_class = predicted.item()
    predicted_label = label_mapping[predicted_class]
    probability = probabilities[0][predicted_class].item()
    
    # Get top-3 predictions
    top3_values, top3_indices = torch.topk(probabilities, min(3, len(label_mapping)))
    top3_predictions = []
    for idx, prob in zip(top3_indices[0], top3_values[0]):
        style_label = label_mapping[idx.item()]
        
        # Apply classification mapping if not a classification model
        if not is_classification_model and style_label in STYLE_TO_CLASSIFICATION:
            style_classification = STYLE_TO_CLASSIFICATION.get(style_label, "Unknown")
            
            # Apply remapping for display
            if style_classification in REMAP_CLASSIFICATIONS_FOR_OUTPUTS:
                display_classification = REMAP_CLASSIFICATIONS_FOR_OUTPUTS[style_classification]
            else:
                display_classification = style_classification
                
            top3_predictions.append((style_label, prob.item(), display_classification))
        else:
            # This is already a classification, apply remapping
            if style_label in REMAP_CLASSIFICATIONS_FOR_OUTPUTS:
                display_classification = REMAP_CLASSIFICATIONS_FOR_OUTPUTS[style_label]
            else:
                display_classification = style_label
                
            top3_predictions.append((style_label, prob.item(), display_classification))
    
    # If this is a style model but we want to show classifications too
    if not is_classification_model and predicted_label in STYLE_TO_CLASSIFICATION:
        predicted_classification = STYLE_TO_CLASSIFICATION.get(predicted_label, "Unknown")
    else:
        predicted_classification = predicted_label
    
    # Remap classification names for output display
    if predicted_classification in REMAP_CLASSIFICATIONS_FOR_OUTPUTS:
        display_classification = REMAP_CLASSIFICATIONS_FOR_OUTPUTS[predicted_classification]
    else:
        display_classification = predicted_classification
    
    return predicted_label, probability, top3_predictions, predicted_classification, display_classification

def process_image(image_path, model, scaler, label_mapping, feature_type='combined', device='cpu', is_classification_model=False):
    """Process a single image and predict its art style."""
    # Extract features
    features = extract_features_from_image(image_path, feature_type)
    
    if features is None:
        return None
    
    # Predict style
    predicted_label, probability, top3_predictions, predicted_classification, display_classification = predict_style(
        model, scaler, label_mapping, features, device, is_classification_model
    )
    
    result = {
        'path': image_path,
        'predicted_label': predicted_label,
        'probability': probability,
        'top3_predictions': top3_predictions
    }
    
    if predicted_classification:
        result['predicted_classification'] = predicted_classification
        result['display_classification'] = display_classification
    
    return result

def process_directory(input_dir, model, scaler, label_mapping, feature_type='combined', device='cpu', is_classification_model=False):
    """Process all images in a directory and predict their art styles."""
    # Find all image files
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    results = []
    for path in image_paths:
        print(f"Processing {path}...")
        result = process_image(path, model, scaler, label_mapping, feature_type, device, is_classification_model)
        if result:
            results.append(result)
    
    return results

def visualize_predictions(results, output_dir, is_classification_model=False):
    """Visualize prediction results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary DataFrame
    summary_data = []
    for result in results:
        data = {
            'path': result['path'],
            'predicted_label': result['predicted_label'],
            'probability': result['probability']
        }
        
        if 'predicted_classification' in result:
            data['predicted_classification'] = result['predicted_classification']
            
        if 'display_classification' in result:
            data['display_classification'] = result['display_classification']
            
        summary_data.append(data)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'prediction_summary.csv'), index=False)
    

    
    # Create a visual grid of predictions
    num_images = min(20, len(results))  # Show at most 20 images
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(15, rows * 4))
    for i in range(num_images):
        result = results[i]
        img = cv2.imread(result['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        
        # Display the appropriate label based on model type
        title = f"{result['predicted_label']}\n{result['probability']:.2f}"
        if 'display_classification' in result:
            title = f"{result['display_classification']}: {result['predicted_label']}\n{result['probability']:.2f}"
        elif 'predicted_classification' in result:
            title = f"{result['predicted_classification']}: {result['predicted_label']}\n{result['probability']:.2f}"
            
        plt.title(title, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_grid.png'))
    plt.close()
    
    # Create individual prediction visualizations with top-3 probabilities
    for i, result in enumerate(results):
        if i >= 20:  # Limit to 20 detailed visualizations
            break
            
        img = cv2.imread(result['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 6))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        
        # Display the appropriate title based on model type
        title = f"Predicted: {result['predicted_label']}"
        if 'display_classification' in result:
            title = f"Predicted: {result['display_classification']}"
        elif 'predicted_classification' in result:
            title = f"Predicted: {result['predicted_classification']}"
            
        plt.title(title)
        plt.axis('off')
        
        # Display top-3 probabilities
        plt.subplot(1, 2, 2)
        
        # Use the display classifications for top3 predictions if available
        if len(result['top3_predictions'][0]) == 3:  # Check if it has display classifications
            labels = [f"{p[2]}" for p in result['top3_predictions']]
            probs = [p[1] for p in result['top3_predictions']]
        else:
            labels = [p[0] for p in result['top3_predictions']]
            probs = [p[1] for p in result['top3_predictions']]
        
        plt.barh(labels, probs)
        plt.xlim(0, 1)
        plt.xlabel('Probability')
        plt.title('Top-3 Predictions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"prediction_{i+1}.png"))
        plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict art styles using the trained model')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to an image or directory of images')
    parser.add_argument('--output-dir', type=str, default='results/predictions',
                        help='Directory to save prediction results')
    parser.add_argument('--feature-type', type=str, default='combined',
                        help='Type of feature to use (e.g., combined, color_hist_rgb, hog)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    return parser.parse_args()

def main():
    """Main function to make predictions."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    model, scaler, label_mapping, is_classification_model = load_model(args.model_dir)
    model.to(device)
    
    # Print model type
    if is_classification_model:
        print("Detected a classification model (6 main categories)")
    else:
        print("Detected a style model (individual styles)")
    
    # Process input
    if os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        results = process_directory(args.input, model, scaler, label_mapping, args.feature_type, device, is_classification_model)
    else:
        print(f"Processing image: {args.input}")
        result = process_image(args.input, model, scaler, label_mapping, args.feature_type, device, is_classification_model)
        results = [result] if result else []
    
    # Visualize predictions
    if results:
        print(f"Visualizing predictions to {args.output_dir}...")
        visualize_predictions(results, args.output_dir, is_classification_model)
        print(f"Prediction complete. Results saved to {args.output_dir}")
    else:
        print("No valid predictions were made.")

if __name__ == "__main__":
    main() 