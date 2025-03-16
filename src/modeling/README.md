# Art Style Classification

This directory contains scripts for training and using neural network models to classify artwork by style based on extracted features.

## Overview

The workflow consists of three main steps:

1. **Split Dataset**: Split the feature-extracted dataset into training and test sets
2. **Train Model**: Train a neural network classifier on the training set
3. **Predict**: Use the trained model to predict art styles for new artwork images

## Target Art Styles

The model is designed to classify artworks into the following 13 styles:

1. High-Renaissance
2. Early-Renaissance
3. Baroque
4. Impressionism
5. Post-Impressionism
6. Expressionism
7. Neoclassicism
8. Classicism
9. Cubism
10. Surrealism
11. Abstract-Art
12. Pop-Art
13. Art-Nouveau-(Modern)

## Scripts

### 1. Dataset Splitter (`dataset_splitter.py`)

Splits the feature-extracted dataset into training and test sets.

```bash
python src/modeling/dataset_splitter.py \
  --features-path data/features/artwork_features.pkl \
  --output-dir data/model \
  --feature-type combined \
  --test-size 0.2
```

#### Arguments:
- `--features-path`: Path to the artwork features pickle file (required)
- `--output-dir`: Directory to save train/test splits (default: 'data/model')
- `--feature-type`: Type of feature to use (default: 'combined')
- `--test-size`: Proportion of the dataset to include in the test split (default: 0.2)
- `--random-state`: Random state for reproducibility (default: 42)

### 2. Model Trainer (`train_model.py`)

Trains a neural network classifier on the training set.

```bash
python src/modeling/train_model.py \
  --data-dir data/model \
  --output-dir results/model \
  --batch-size 32 \
  --hidden-sizes 1024,512,256 \
  --learning-rate 0.001 \
  --dropout-rate 0.5 \
  --num-epochs 100
```

#### Arguments:
- `--data-dir`: Directory containing train/test splits (required)
- `--output-dir`: Directory to save model and results (default: 'results/model')
- `--batch-size`: Batch size for training (default: 32)
- `--hidden-sizes`: Comma-separated list of hidden layer sizes (default: '1024,512,256')
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--dropout-rate`: Dropout rate for regularization (default: 0.5)
- `--num-epochs`: Number of epochs to train for (default: 100)
- `--early-stopping-patience`: Number of epochs without improvement before early stopping (default: 10)
- `--no-cuda`: Disable CUDA even if available

### 3. Predictor (`predict.py`)

Uses the trained model to predict art styles for new artwork images.

```bash
python src/modeling/predict.py \
  --model-dir results/model/model_20230415_123456 \
  --input path/to/artwork/image.jpg \
  --output-dir results/predictions
```

#### Arguments:
- `--model-dir`: Directory containing the trained model (required)
- `--input`: Path to an image or directory of images (required)
- `--output-dir`: Directory to save prediction results (default: 'results/predictions')
- `--feature-type`: Type of feature to use (default: 'combined')
- `--no-cuda`: Disable CUDA even if available

## Complete Workflow Example

Here's a complete workflow example:

```bash
# 1. Create necessary directories
mkdir -p data/model results/model results/predictions

# 2. Split the dataset
python src/modeling/dataset_splitter.py \
  --features-path data/features/artwork_features.pkl \
  --output-dir data/model

# 3. Train the model
python src/modeling/train_model.py \
  --data-dir data/model \
  --output-dir results/model

# 4. Make predictions on new images
python src/modeling/predict.py \
  --model-dir results/model/model_TIMESTAMP \
  --input path/to/new/artwork/images \
  --output-dir results/predictions
```

## Model Architecture

The neural network model consists of:
- Input layer with size matching the feature vector dimension
- Three hidden layers with ReLU activation, batch normalization, and dropout
- Output layer with 13 units (one for each art style)

The model is trained using:
- Cross-entropy loss function
- Adam optimizer
- Learning rate scheduler that reduces the learning rate when validation loss plateaus
- Early stopping to prevent overfitting

## Outputs

### Training Outputs
- Model weights and architecture
- Training history plots (loss and accuracy)
- Confusion matrix
- Classification report with precision, recall, and F1-score for each class
- Evaluation results summary

### Prediction Outputs
- Prediction summary CSV file
- Style distribution plot
- Visual grid of predictions
- Individual prediction visualizations with top-3 probabilities 