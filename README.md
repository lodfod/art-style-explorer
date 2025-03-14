# Art Style Explorer

An application that identifies artists and artworks with similar styles to an input image.

## Project Overview

Art Style Explorer analyzes artwork images and identifies stylistically similar artists and works. Unlike conventional reverse image search tools that find visually identical images, this application focuses on artistic style analysis, particularly focusing on line work and composition.

## Features

- Image preprocessing and line extraction
- Custom neural network for style recognition
- Style similarity analysis
- Artist and artwork recommendations
- WikiArt dataset integration with 120k+ artwork images

## Project Structure

```
art-style-explorer/
├── data/                       # Data directory
│   ├── wikiart_scraped.csv     # WikiArt dataset with image links
│   ├── image_cache/            # Cached images from WikiArt
│   ├── processed/              # Processed images and features
│   └── README.md               # Data documentation
├── src/                        # Source code
│   ├── preprocessing/          # Image preprocessing modules
│   │   ├── __init__.py
│   │   ├── edge_detection.py   # Line extraction using OpenCV
│   │   ├── normalization.py    # Image normalization
│   │   └── wikiart_processor.py # WikiArt dataset processor
│   ├── features/               # Feature extraction
│   │   ├── __init__.py
│   │   ├── composition.py      # Composition analysis
│   │   └── line_features.py    # Line work feature extraction
│   ├── model/                  # Neural network model
│   │   ├── __init__.py
│   │   ├── network.py          # Custom neural network architecture
│   │   └── training.py         # Training procedures
│   ├── evaluation/             # Evaluation tools
│   │   ├── __init__.py
│   │   └── metrics.py          # Performance metrics
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── data_loader.py      # Data loading utilities
│       ├── wikiart_dataloader.py # WikiArt dataset utilities
│       └── visualization.py    # Visualization tools
├── notebooks/                  # Jupyter notebooks for exploration and analysis
│   └── wikiart_dataset_demo.ipynb # Demo of WikiArt dataset usage
├── results/                    # Results and outputs
├── tests/                      # Unit tests
├── requirements.txt            # Project dependencies
├── setup.py                    # Package installation
└── README.md                   # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/art-style-explorer.git
cd art-style-explorer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## WikiArt Dataset

This project integrates the WikiArt dataset containing over 120,000 artwork images with their styles, artists, and other metadata. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link).

### Dataset Processing

To process the WikiArt dataset and extract edge features:

```bash
# Process the dataset
python src/preprocessing/wikiart_processor.py \
    --csv-path data/wikiart_scraped.csv \
    --output-dir data/processed \
    --extract-features
```

For more details on the dataset and processing, see the [data README](data/README.md).

## Usage

### Using the Processed WikiArt Dataset

```python
from src.utils.wikiart_dataloader import create_dataloaders

# Create dataloaders for model training
dataloaders = create_dataloaders(
    metadata_path='data/processed/train/metadata.csv',
    data_dir='data/processed/train',
    batch_size=32,
    image_size=224,
    use_features=True
)

# Access the dataloaders
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

### Processing Individual Images

```python
from src.preprocessing import edge_detection
from src.features import line_features

# Process an artwork image
image_path = 'path/to/artwork.jpg'
preprocessed, edges = edge_detection.process_artwork(
    image_path,
    edge_method='canny'
)

# Extract line features
line_features_dict = line_features.extract_line_features(edges)
```

## Development Process

1. Data Collection: Integrate the WikiArt dataset with 120k+ artwork images
2. Preprocessing: Implement line extraction and image normalization
3. Feature Engineering: Develop algorithms to analyze composition and line work
4. Model Development: Build and train a custom neural network
5. Evaluation: Test the system's ability to identify stylistically similar artworks
6. Fine-tuning: Optimize the model based on evaluation results

## Required Dependencies

See `requirements.txt` for a complete list of dependencies. Key dependencies include:
- numpy, pandas, matplotlib
- opencv-python for image processing
- torch and torchvision for neural networks
- requests for downloading images

## License

[MIT License](LICENSE) 