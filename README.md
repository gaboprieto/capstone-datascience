# Team Clustering Analysis

This project implements various clustering algorithms to analyze and optimize team formation in academic settings.

## Features

- Multiple clustering algorithms (K-means, Hierarchical, DBSCAN, Spectral)
- Team composition analysis
- Temporal evolution analysis
- Visualization tools for cluster comparison

## Installation 

```
bash
git clone https://github.com/yourusername/team-clustering.git
cd team-clustering

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Usage

```
bash

python main.py --input path/to/data.csv --output path/to/output --grades-file path/to/grades
```

## Project Structure

- `src/`: Source code
  - `clustering/`: Clustering algorithms
  - `visualization/`: Plotting and visualization
  - `utils/`: Data processing utilities
- `tests/`: Unit tests
- `data/`: Input data
- `output/`: Generated outputs

## License

MIT License