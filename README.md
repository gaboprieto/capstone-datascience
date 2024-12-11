# Team Clustering Analysis

This project implements various clustering algorithms to analyze and optimize team formation in academic settings.

## Features

- Multiple clustering algorithms (K-means, Hierarchical, DBSCAN)
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

## Directory Structure Explanation

- `data/`: Contains input data files
  - `student_data.csv`: Student activity and performance data
  - `grades.xlsx`: Student grades data

- `src/`: Source code
  - `clustering/`: Clustering algorithms and metrics
    - `algorithms.py`: Implementation of clustering algorithms
    - `metrics.py`: Evaluation metrics for clustering
  - `utils/`: Utility functions
    - `analysis.py`: Data analysis functions
    - `data_processing.py`: Data preprocessing functions
  - `visualization/`: Visualization code
    - `plots.py`: Plotting functions

- `output/`: Generated output
  - `figures/`: Generated plots and visualizations
  - `results/`: Analysis results and reports

- `tests/`: Unit tests

- Root files:
  - `.gitignore`: Git ignore rules
  - `README.md`: Project documentation
  - `requirements.txt`: Project dependencies
  - `main.py`: Main execution script

## License

MIT License
