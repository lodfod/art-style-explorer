# WikiArt Data Loader

This script filters the WikiArt dataset by specified art styles and downloads images for each style.

## Features

- Filters the WikiArt dataset to include only specified art styles
- Downloads up to a maximum number of images per style (default: 300)
- Organizes downloaded images by style in separate directories
- Saves filtered metadata in CSV format
- Supports parallel downloading with configurable workers
- Handles download failures with retries and timeout settings

## Target Art Styles

The script filters for the following art styles:
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

## Usage

```bash
python src/preprocessing/wiki_dataloader.py --csv-path data/wikiart_scraped.csv --output-dir data/filtered
```

### Command Line Arguments

- `--csv-path`: Path to the WikiArt CSV file (required)
- `--output-dir`: Directory to save filtered data and images (default: 'data/filtered')
- `--max-per-style`: Maximum number of images to download per style (default: 300)
- `--num-workers`: Number of workers for parallel downloading (default: 4)
- `--timeout`: Timeout for image download requests in seconds (default: 10)
- `--retry`: Number of retries for failed downloads (default: 3)

## Output

The script creates the following output:

1. `filtered_wikiart.csv`: Contains metadata for all selected artworks
2. `downloaded_wikiart.csv`: Contains metadata for successfully downloaded artworks
3. `failed_downloads.csv`: Contains metadata for artworks that failed to download
4. Style-specific directories containing downloaded images

## Example

```bash
# Download up to 200 images per style with 8 parallel workers
python src/preprocessing/wiki_dataloader.py \
    --csv-path data/wikiart_scraped.csv \
    --output-dir data/filtered \
    --max-per-style 200 \
    --num-workers 8
``` 