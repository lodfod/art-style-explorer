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
from src.feature_extraction.feature_extractor import ArtworkFeatureExtractor
from src.modeling.train_model import ArtStyleClassifier

# Define style classifications and their constituent styles
STYLE_CLASSIFICATIONS = {
    "Classical_And_Renaissance": [
        "Early-Renaissance", 
        "High-Renaissance", 
        "Mannerism", 
        "Neoclassicism", 
        "Classicism"
    ],
    "Medieval_And_Ornamental": [
        "Romanesque", 
        "Baroque"
    ],
    "Impressionist_Movements": [
        "Impressionism", 
        "Post-Impressionism"
    ],
    "Expressionist_And_Surrealist": [
        "Expressionism", 
        "Surrealism"
    ],
    "Abstract_And_Fragmented": [
        "Cubism", 
        "Abstract-Art"
    ],
    "Graphic_Styles": [
        "Ukiyo-e", 
        "Pop-Art", 
        "Art-Nouveau-(Modern)"
    ]
}

# Create a mapping from individual style to its classification
STYLE_TO_CLASSIFICATION = {}
for classification, styles in STYLE_CLASSIFICATIONS.items():
    for style in styles:
        STYLE_TO_CLASSIFICATION[style] = classification

def load_model(model_dir):
    """Load the trained model and associated metadata."""
    # Load label mapping
    with open(os.path.join(model_dir, 'label_mapping.pkl'), 'rb') as f:
        label_mapping = pickle.load(f)
    
    # Load scaler
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model architecture
    with open(os.path.join(model_dir, 'model_architecture.txt'), 'r') as f:
        architecture = f.read()
    
    # Parse input size and hidden sizes from architecture
    # This is a simple approach and might need adjustment based on your model architecture
    lines = architecture.split('\n')
    input_size = None
    hidden_sizes = []
    
    for line in lines:
        if 'Linear' in line:
            parts = line.split('(')
            if len(parts) > 1:
                size_part = parts[1].split(',')[0]
                if 'in_features=' in size_part:
                    in_features = int(size_part.split('=')[1])
                    out_features = int(parts[1].split('out_features=')[1].split(',')[0])
                    
                    if input_size is None:
                        input_size = in_features
                        hidden_sizes.append(out_features)
                    elif len(hidden_sizes) < 3:  # Assuming 3 hidden layers
                        hidden_sizes.append(out_features)
    
    # Remove the last element (output layer)
    hidden_sizes = hidden_sizes[:-1]
    
    # Create model
    num_classes = len(label_mapping)
    model = ArtStyleClassifier(input_size, hidden_sizes, num_classes)
    
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=torch.device('cpu')))
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
    top3_predictions = [
        (label_mapping[idx.item()], prob.item())
        for idx, prob in zip(top3_indices[0], top3_values[0])
    ]
    
    # If this is a style model but we want to show classifications too
    if not is_classification_model and predicted_label in STYLE_TO_CLASSIFICATION:
        predicted_classification = STYLE_TO_CLASSIFICATION.get(predicted_label, "Unknown")
    else:
        predicted_classification = None
    
    return predicted_label, probability, top3_predictions, predicted_classification

def process_image(image_path, model, scaler, label_mapping, feature_type='combined', device='cpu', is_classification_model=False):
    """Process a single image and predict its art style."""
    # Extract features
    features = extract_features_from_image(image_path, feature_type)
    
    if features is None:
        return None
    
    # Predict style
    predicted_label, probability, top3_predictions, predicted_classification = predict_style(
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
            
        summary_data.append(data)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'prediction_summary.csv'), index=False)
    
    # Plot label distribution
    plt.figure(figsize=(12, 8))
    label_column = 'predicted_label'
    label_counts = summary_df[label_column].value_counts().sort_values(ascending=False)
    plt.bar(label_counts.index, label_counts.values)
    plt.xlabel('Art Style' if not is_classification_model else 'Classification')
    plt.ylabel('Count')
    plt.title(f'Distribution of Predicted {label_column}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    plt.close()
    
    # If we have classification data, plot that too
    if 'predicted_classification' in summary_df.columns and not is_classification_model:
        plt.figure(figsize=(12, 8))
        classification_counts = summary_df['predicted_classification'].value_counts().sort_values(ascending=False)
        plt.bar(classification_counts.index, classification_counts.values)
        plt.xlabel('Classification')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Classifications')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_distribution.png'))
        plt.close()
    
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
        if 'predicted_classification' in result:
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
        if 'predicted_classification' in result:
            title = f"Predicted: {result['predicted_classification']}\n{result['predicted_label']}"
            
        plt.title(title)
        plt.axis('off')
        
        # Display top-3 probabilities
        plt.subplot(1, 2, 2)
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