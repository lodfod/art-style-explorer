# Art Style Classification System

This README explains how to use the updated art style classification system with the new hierarchical classification structure.

## Classification Structure

The system now organizes art styles into 6 main classifications:

1. **Classical And Renaissance Styles**
   - Early-Renaissance
   - High-Renaissance
   - Mannerism
   - Neoclassicism
   - Classicism

2. **Medieval & Ornamental Architecture Styles**
   - Romanesque
   - Baroque

3. **Impressionist Movements Styles**
   - Impressionism
   - Post-Impressionism

4. **Expressionist & Surrealist Styles**
   - Expressionism
   - Surrealism

5. **Abstract & Fragmented Forms Styles**
   - Cubism
   - Abstract-Art

6. **Graphic Styles**
   - Ukiyo-e
   - Pop-Art
   - Art-Nouveau-(Modern)

## Complete Pipeline

Here's how to run the complete pipeline with the new classification structure:

### 1. Download and Filter Artwork Images

```bash
mkdir -p data/filtered
python src/preprocessing/wiki_dataloader.py \
  --csv-path data/wikiart_scraped.csv \
  --output-dir data/filtered \
  --max-per-classification 500 \
  --hierarchical \
  --num-workers 4
```

This will:
- Download up to 500 images per classification
- Organize images in a hierarchical structure (classification/style/image)
- Balance styles within each classification
- Shuffle styles within classifications to prevent bias

### 2. Extract Features from Images

```bash
mkdir -p data/features
python src/feature_extraction/feature_extractor.py \
  --input-dir data/filtered \
  --output-dir data/features \
  --num-workers 4
```

### 3. Augment Features (Optional)

```bash
mkdir -p data/features/augmented
python src/modeling/feature_augmenter.py \
  --features-path data/features/artwork_features.pkl \
  --output-path data/features/augmented/augmented_features.pkl \
  --augmentation-strategy all \
  --intensity 0.1 \
  --balance-classes \
  --use-classifications \
  --num-samples-per-classification 800
```

This will:
- Augment features using all feature types
- Balance the dataset to have 800 samples per classification
- Use the 6 main classifications instead of individual styles

### 4. Split the Dataset

```bash
mkdir -p data/model_classified
python src/modeling/dataset_splitter.py \
  --features-path data/features/augmented/augmented_features.pkl \
  --output-dir data/model_classified \
  --feature-type combined \
  --test-size 0.2 \
  --use-classifications
```

The `--use-classifications` flag tells the splitter to use the 6 main classifications as labels instead of individual styles.

### 5. Train the Model

```bash
mkdir -p results/model_classified
python src/modeling/train_model.py \
  --data-dir data/model_classified \
  --output-dir results/model_classified \
  --batch-size 32 \
  --hidden-sizes 512,256,128 \
  --learning-rate 0.0005 \
  --dropout-rate 0.6 \
  --num-epochs 100 \
  --early-stopping-patience 15
```

### 6. Make Predictions on New Images

```bash
python src/modeling/predict.py \
  --model-dir results/model_classified/model_TIMESTAMP \
  --input-path path/to/new/artwork/images \
  --output-dir results/predictions
```

## Complete Example Command

Here's a complete example command that runs the entire pipeline in sequence:

```bash
# 1. Download and filter artwork images
mkdir -p data/filtered
python src/preprocessing/wiki_dataloader.py \
  --csv-path data/wikiart_scraped.csv \
  --output-dir data/filtered \
  --max-per-classification 500 \
  --hierarchical \
  --num-workers 4

# 2. Extract features from images
mkdir -p data/features
python src/feature_extraction/feature_extractor.py \
  --input-dir data/filtered \
  --output-dir data/features \
  --num-workers 4

# 3. Augment features
mkdir -p data/features/augmented
python src/modeling/feature_augmenter.py \
  --features-path data/features/artwork_features.pkl \
  --output-path data/features/augmented/augmented_features.pkl \
  --augmentation-strategy all \
  --intensity 0.1 \
  --balance-classes \
  --use-classifications \
  --num-samples-per-classification 800

# 4. Split the dataset
mkdir -p data/model_classified
python src/modeling/dataset_splitter.py \
  --features-path data/features/augmented/augmented_features.pkl \
  --output-dir data/model_classified \
  --feature-type combined \
  --test-size 0.2 \
  --use-classifications

# 5. Train the model
mkdir -p results/model_classified
python src/modeling/train_model.py \
  --data-dir data/model_classified \
  --output-dir results/model_classified \
  --batch-size 32 \
  --hidden-sizes 512,256,128 \
  --learning-rate 0.0005 \
  --dropout-rate 0.6 \
  --num-epochs 100 \
  --early-stopping-patience 15
```

## Command-Line Arguments

### wiki_dataloader.py

- `--csv-path`: Path to the WikiArt CSV file
- `--output-dir`: Directory to save filtered data and images
- `--max-per-style`: Maximum number of images to download per style
- `--max-per-classification`: Maximum number of images to download per classification (overrides max-per-style)
- `--hierarchical`: Organize images in a hierarchical directory structure by classification
- `--num-workers`: Number of workers for parallel downloading
- `--seed`: Random seed for reproducibility (default: 42)

### feature_augmenter.py

- `--features-path`: Path to the artwork features pickle file
- `--output-path`: Path to save the augmented features
- `--augmentation-strategy`: Strategy for feature augmentation (all, color_only, texture_only, etc.)
- `--intensity`: Intensity of augmentation (0.0-1.0)
- `--balance-classes`: Balance classes by generating synthetic samples
- `--num-samples-per-style`: Target number of samples per style
- `--use-classifications`: Use the 6 main classifications instead of individual styles
- `--num-samples-per-classification`: Target number of samples per classification
- `--seed`: Random seed for reproducibility (default: 42)

### dataset_splitter.py

- `--features-path`: Path to the artwork features pickle file
- `--output-dir`: Directory to save train/test splits
- `--feature-type`: Type of feature to use (e.g., combined, color_hist_rgb, hog)
- `--test-size`: Proportion of the dataset to include in the test split
- `--random-state`: Random seed for reproducibility
- `--use-classifications`: Use the 6 main classifications instead of individual styles

## Tips for Better Results

1. **Use Hierarchical Structure**: The `--hierarchical` flag in the dataloader organizes images by classification and style, making it easier to work with the dataset.

2. **Balance by Classification**: Instead of balancing individual styles, use `--use-classifications` and `--num-samples-per-classification` to ensure each main classification has enough samples.

3. **Shuffle Styles**: The dataloader now shuffles styles within each classification to prevent bias in the dataset.

4. **Adjust Feature Augmentation**: You can use different augmentation strategies for different classifications by running the augmenter multiple times with different settings.

5. **Cross-Validation**: For more robust evaluation, consider implementing cross-validation across the 6 main classifications.

## Example: Training on Specific Classifications

If you want to focus on specific classifications, you can modify the `STYLE_CLASSIFICATIONS` dictionary in both scripts to include only the classifications you're interested in.

For example, to train a model that distinguishes only between Classical/Renaissance and Abstract/Fragmented styles:

```bash
# First, modify the STYLE_CLASSIFICATIONS in both scripts to include only these two
# Then run the pipeline as usual
```

## Visualizing Classification Results

After training, you can visualize the confusion matrix to see how well the model distinguishes between the 6 main classifications:

```bash
python src/visualization/visualize_results.py \
  --model-dir results/model_classified/model_TIMESTAMP \
  --output-dir results/visualizations \
  --plot-confusion-matrix
``` 