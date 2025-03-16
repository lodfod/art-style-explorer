# Feature Vector Augmentation

This script provides methods to augment extracted feature vectors for improving model training. The augmentation happens after feature extraction but before model training, and can help address issues like class imbalance and overfitting.

## Key Features

- **Feature-specific augmentation**: Applies appropriate transformations to different types of features (color, texture, edge, contour)
- **Class balancing**: Generates synthetic samples for underrepresented classes
- **Configurable intensity**: Controls the strength of augmentations
- **Multiple strategies**: Allows focusing on specific feature types

## How It Works

The augmentation process works at the feature vector level, not on the original images. It applies small, controlled perturbations to the feature vectors that preserve their statistical properties while creating variations that help the model generalize better.

### Augmentation Strategies

1. **Color Features**:
   - Shifts histogram bins slightly
   - Adjusts color moments with small random noise

2. **Texture Features**:
   - Modifies LBP histograms
   - Adds controlled noise to Haralick and Gabor features

3. **Edge Features**:
   - Slightly adjusts edge density
   - Adds small variations to HOG features while preserving normalization

4. **Contour Features**:
   - Adds controlled noise to contour statistics

## Usage

```bash
python src/modeling/feature_augmenter.py \
  --features-path data/features/artwork_features.pkl \
  --output-path data/features/augmented_features.pkl \
  --augmentation-strategy all \
  --intensity 0.1 \
  --balance-classes
```

### Command Line Arguments

- `--features-path`: Path to the original artwork features pickle file (required)
- `--output-path`: Path to save the augmented features (required)
- `--augmentation-strategy`: Strategy for feature augmentation (default: 'all')
  - Options: 'all', 'color_only', 'texture_only', 'edge_only', 'contour_only', 'color_texture', 'edge_contour'
- `--intensity`: Intensity of augmentation (0.0-1.0, default: 0.1)
- `--balance-classes`: Balance classes by generating synthetic samples (flag)
- `--num-samples-per-style`: Target number of samples per style (default: max of original counts)

## Workflow Example

Here's a complete workflow example that includes feature augmentation:

```bash
# 1. Extract features from artwork images
python src/feature_extraction/feature_extractor.py \
  --input-dir data/filtered \
  --output-dir data/features

# 2. Augment features to balance classes and improve generalization
python src/modeling/feature_augmenter.py \
  --features-path data/features/artwork_features.pkl \
  --output-path data/features/augmented_features.pkl \
  --augmentation-strategy all \
  --intensity 0.1 \
  --balance-classes

# 3. Split the augmented dataset
python src/modeling/dataset_splitter.py \
  --features-path data/features/augmented_features.pkl \
  --output-dir data/model_augmented

# 4. Train the model on the augmented dataset
python src/modeling/train_model.py \
  --data-dir data/model_augmented \
  --output-dir results/model_augmented
```

## Tips for Effective Augmentation

1. **Start with low intensity** (0.05-0.1) and gradually increase if needed
2. **Balance classes** to ensure the model learns from all styles equally
3. **Try different strategies** to see which features benefit most from augmentation
4. **Combine with regularization** in the model training for best results
5. **Monitor validation performance** to ensure augmentation is helping, not hurting

## Example: Addressing Class Imbalance

If your dataset has significantly fewer samples for certain art styles, you can balance it:

```bash
python src/modeling/feature_augmenter.py \
  --features-path data/features/artwork_features.pkl \
  --output-path data/features/balanced_features.pkl \
  --balance-classes \
  --num-samples-per-style 300
```

This will generate synthetic samples for each style until all styles have 300 samples.

## Example: Focused Augmentation

If you want to focus augmentation on specific feature types:

```bash
python src/modeling/feature_augmenter.py \
  --features-path data/features/artwork_features.pkl \
  --output-path data/features/color_texture_augmented.pkl \
  --augmentation-strategy color_texture \
  --intensity 0.15
```

This will apply stronger augmentation (0.15 intensity) only to color and texture features. 