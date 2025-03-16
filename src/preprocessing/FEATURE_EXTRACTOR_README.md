# Artwork Feature Extractor

This module extracts a comprehensive set of visual features from artwork images, which can be used for style analysis, classification, and exploration.

## Features Extracted

The feature extractor computes the following types of features:

### 1. Color-Based Features
- **Color Histograms**: Distribution of colors in RGB and HSV color spaces
- **Color Moments**: Statistical moments (mean, variance, skewness) for each color channel

### 2. Texture Features
- **Local Binary Patterns (LBP)**: Captures micro-textures by comparing each pixel to its neighborhood
- **Haralick Features**: Derived from the Gray-Level Co-occurrence Matrix (GLCM), describing texture properties
- **Gabor Filter Responses**: Analyzes the image at multiple scales and orientations

### 3. Edge and Shape Features
- **Edge Density**: Ratio of edge pixels to total pixels
- **Histogram of Oriented Gradients (HOG)**: Captures the distribution of edge directions

### 4. Curvature and Contour Features
- **Contour Statistics**: Number of contours, total and average contour length
- **Curvature Measures**: Statistical analysis of curvature along detected contours

## Usage

### Basic Usage

```bash
python src/preprocessing/feature_extractor.py --input-dir data/filtered --output-dir data/features
```

### Command Line Arguments

- `--input-dir`: Directory containing artwork images (required)
- `--output-dir`: Directory to save extracted features (default: 'data/features')
- `--num-workers`: Number of worker processes for parallel extraction (default: 4)

## Workflow Example

1. First, download and filter the WikiArt dataset:
   ```bash
   python src/preprocessing/wiki_dataloader.py --csv-path data/wikiart_scraped.csv --output-dir data/filtered
   ```

2. Then, extract features from the downloaded images:
   ```bash
   python src/preprocessing/feature_extractor.py --input-dir data/filtered --output-dir data/features
   ```

## Output

The script creates the following output files:

1. `artwork_features.pkl`: A pickle file containing all extracted features for each image
2. `feature_summary.csv`: A CSV file summarizing the feature dimensions and descriptions

## Feature Dimensions

The feature extractor produces features with the following dimensions:

| Feature Name | Dimension | Description |
|--------------|-----------|-------------|
| color_hist_rgb | 192 | RGB color histogram (64 bins per channel) |
| color_hist_hsv | 100 | HSV color histogram (H: 36 bins, S/V: 32 bins each) |
| color_moments | 18 | Statistical moments (mean, std, skewness) for RGB and HSV channels |
| lbp_hist | 26 | Local Binary Pattern histogram for texture analysis |
| haralick | 40 | Haralick texture features from Gray-Level Co-occurrence Matrix |
| gabor | 64 | Gabor filter responses at multiple scales and orientations |
| edge_density | 1 | Ratio of edge pixels to total pixels |
| hog | 1764 | Histogram of Oriented Gradients for shape analysis |
| contour_features | 3 | Number of contours, total and average contour length |
| curvature_stats | 20 | Statistical measures of curvature along contours |
| combined | 2228 | All features combined into a single vector |

## Using the Extracted Features

The extracted features can be used for various tasks:

1. **Style Classification**: Train a classifier to predict art styles based on extracted features
2. **Style Similarity**: Compute similarity between artworks based on feature vectors
3. **Dimensionality Reduction**: Apply techniques like PCA or t-SNE to visualize artwork relationships
4. **Feature Importance Analysis**: Determine which features are most discriminative for different art styles

Example code for loading and using the features:

```python
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load features
with open('data/features/artwork_features.pkl', 'rb') as f:
    features = pickle.load(f)

# Extract combined feature vectors and style labels
X = []
styles = []
for path, feature_dict in features.items():
    X.append(feature_dict['combined'])
    # Extract style from path (assuming directory structure from wiki_dataloader.py)
    style = path.split('/')[-2]
    styles.append(style)

X = np.array(X)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA results
plt.figure(figsize=(12, 10))
for style in set(styles):
    mask = [s == style for s in styles]
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=style, alpha=0.7)

plt.legend()
plt.title('PCA of Artwork Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
``` 