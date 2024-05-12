
# Music Recommendation

## Overview
"Music_recommendation" is a project designed to explore and implement data clustering techniques. The project uses Python to analyze datasets, apply clustering algorithms, and visualize the results, focusing on music files to provide tailored song recommendations.

## Contents
- `Final.ipynb`: A Jupyter notebook that contains the exploratory data analysis, clustering algorithm implementation, and visualization of the clustering results.
- `main.py`: A Python script that sets up the data processing pipeline, loads the data, applies clustering algorithms, and saves the output.
- `recommend.py`: A Python script that recommends the song.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Libraries: numpy, pandas, matplotlib, scikit-learn
- Spark Shell
- Kafka

### Installation
1. Clone the repository or download the files to your local machine.
2. Install the required Python packages:

```bash
pip install librosa pymongo sklearn matplotlib scipy
```

### Usage
- To view and run the Jupyter notebook:
```bash
jupyter notebook Final.ipynb
```
- To execute the Python script:
```bash
spark-submit --master local[6] main.py
```

## Methodology
- **Data Processing and Feature Extraction:**
  Audio files are processed using `librosa` to extract MFCCs, spectral centroids, and zero-crossing rates. These features are normalized and stored in MongoDB for efficient access.
- **Clustering:**
  Songs are clustered based on extracted audio features using Euclidean distance metrics to form groups of similar tracks.

## Findings
- The system successfully identifies and groups songs with similar audio characteristics, demonstrating the clustering's effectiveness. Detailed cluster assignments and feature space exploration are provided in the Jupyter notebook.

## Evaluation
- **Performance Metrics:**
  We evaluate the system based on the coherence of clusters with subjective music similarity criteria. (Further metrics and evaluations could be included here.)

## Challenges
- Handling large audio files and ensuring the accurate extraction and normalization of features was computationally demanding.

## Improvements
- Future work could explore more sophisticated clustering techniques and integrate additional features such as tempo and harmony.

## Contributing
Contributions to "Music_Recommendation" are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is open source and available under the [MIT License](LICENSE.md).

## Contact
For questions or feedback, please reach out to [azanshahzad416@gmail.com].
