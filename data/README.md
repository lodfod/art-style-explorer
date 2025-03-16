# WikiArt Dataset

This directory contains the WikiArt dataset and processed derivatives.

## Dataset Structure

- `wikiart_scraped.csv`: The original WikiArt dataset CSV file containing artwork metadata including style, artist, and image URLs.
- `image_cache/`: Directory containing cached images downloaded from URLs (created during processing).
- `processed/`: Directory containing processed images, edges, and features (created during processing).

## Dataset Format

The `wikiart_scraped.csv` file contains the following columns:

- `Style`: The art style of the artwork (e.g., 'Impressionism', 'Cubism', etc.)
- `Artwork`: The name of the artwork
- `Artist`: The name of the artist
- `Date`: The date the artwork was created
- `Link`: URL link to the artwork image

## Processing the Dataset

To process the dataset, run the `wikiart_processor.py` script:

```bash
python src/preprocessing/wikiart_processor.py \
    --csv-path data/wikiart_scraped.csv \
    --output-dir data/processed \
    --target-size 512 \
    --batch-size 32 \
    --edge-method canny \
    --extract-features
```

This will:
1. Download images from the URLs
2. Preprocess images to a standard size
3. Extract edges using the specified method
4. Calculate line features
5. Split the dataset into train/val/test sets
6. Save processed data to the output directory

## Processed Data Structure

After processing, the `processed/` directory will contain:

- `train/`, `val/`, `test/`: Directories for each data split, each containing:
  - `preprocessed/`: Preprocessed images
  - `edges/`: Edge-detected images
  - `features/`: Numpy files containing extracted line features
  - `metadata.csv`: CSV file mapping artwork IDs to file paths and style information
- `train_metadata_raw.csv`, `val_metadata_raw.csv`, `test_metadata_raw.csv`: Original metadata for each split
- `style_mapping.json`: Mapping between style names and numeric IDs

## Using the Processed Data

You can use the processed data with the provided utility functions:

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

See the `notebooks/wikiart_dataset_demo.ipynb` notebook for a complete example of working with the dataset. 