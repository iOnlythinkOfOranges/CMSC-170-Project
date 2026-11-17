# Next-Day Rainfall Prediction in the Philippines

### CMSC 170 · Group BAWWL
We compare Gaussian Naive Bayes, k-nearest neighbors (KNN), and logistic regression for predicting next-day rainfall in six Philippine locations. Each model uses the previous five days of weather data to predict whether the next day's rainfall will exceed 1.0 mm.

## Setup
From the repository root, install the dependencies. The pipeline was verified with Python 3.14.3.

```sh
python -m pip install -r requirements.txt
```

## Data
The repository includes daily NASA POWER weather data in [`data/raw/`](data/raw/) for Baguio, Cebu City, Davao City, Legazpi City, Metro Manila, and Zamboanga City. Each CSV contains 9,132 daily records from January 1, 2000 through December 31, 2024.

Use the CSVs to reproduce the results. If a CSV is missing, the pipeline downloads the corresponding data from the NASA POWER Daily API and saves it under `data/raw/`. An internet connection is needed for downloads.

## Run the comparison
From the repository root, run:

```sh
python src/main.py
```

This command loads and preprocesses the weather data, splits each location's samples chronologically into training, validation, and test sets (70%, 15%, and 15%), runs all three models, and comparison.

## Expected outputs
The terminal prints the data split sizes for each location, validation scores for Gaussian Naive Bayes and logistic regression, and test scores for KNN. For the plots, see `metrics_comparison.png` and `roc_curves_by_location.png`.

| Figure | What it shows |
| --- | --- |
| [`metrics_comparison.png`](figures/metrics_comparison.png) | Test accuracy, precision, recall, and F1 score for each model and location |
| [`roc_curves_by_location.png`](figures/roc_curves_by_location.png) | Test ROC curves and AUC values for each model and location |
| [`calibration_curves.png`](figures/calibration_curves.png) | Predicted rain probabilities compared with observed rain frequencies |
| [`timeseries_metro_manila.png`](figures/timeseries_metro_manila.png) | Predicted rain probabilities and observed rainy days in Metro Manila |