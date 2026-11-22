# Next-Day Rainfall Prediction in the Philippines

## CMSC 170 · Group BAWWL

We compare Gaussian Naive Bayes, k-nearest neighbors (KNN), and logistic regression for predicting next-day rainfall in six Philippine locations. Each model uses the previous three days of weather data to predict whether the next day's rainfall will exceed 1.0 mm.

[View the project page on GitHub Pages](https://ionlythinkoforanges.github.io/CMSC-170-Project/)

## Setup

We ran the project with Python 3.14.3. From the repository root, install the dependencies.

```sh
python -m pip install -r requirements.txt
```

## Data

The [data folder](data/raw/) contains daily NASA POWER weather data for Baguio, Cebu City, Davao City, Legazpi City, Metro Manila, and Zamboanga City. Each CSV has 9,132 daily records from January 1, 2000 through December 31, 2024.

Use the included CSVs to reproduce our results. If a CSV is missing, the script downloads the data from the NASA POWER Daily API and saves it in `data/raw/`. An internet connection is needed for downloads.

## Final settings

| Setting              | Value                                                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Lookback             | 3 days, with 11 features per day (33 per sample)                                                                       |
| Rain event           | More than 1.0 mm of rain on the following day                                                                          |
| Data split           | 70% training, 15% validation, 15% test, in date order for each location                                                |
| Feature scaling      | Each location's training mean and standard deviation for each feature                                                  |
| Gaussian Naive Bayes | Variance smoothing of `1e-9`                                                                                           |
| KNN                  | 151 neighbors and Euclidean distance (`p=2`)                                                                           |
| Logistic regression  | Learning rate of 0.5 and 2,000 iterations                                                                              |
| Rain or dry cutoff   | For each model and location, the cutoff with the highest validation F1 among values from 0.02 to 0.98 in steps of 0.02 |

## Reproduce the results

Run this from the repository root.

```sh
python src/main.py
```

The script trains one of each model per location. It selects the rain or dry cutoffs on the validation set, then uses them on the test set. The terminal shows each location's metrics, the average across the six locations, and the always-rain baseline.

| Figure                                                          | What it shows                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Metrics comparison](figures/metrics_comparison.png)            | Test accuracy, precision, recall, and F1 for each model and location     |
| [ROC curves](figures/roc_curves_by_location.png)                | Test ROC curves and AUC for each model and location                      |
| [Calibration curves](figures/calibration_curves.png)            | Predicted rain probabilities compared with observed rain frequencies     |
| [Metro Manila time series](figures/timeseries_metro_manila.png) | Rain probabilities and observed rainy days during the first 60 test days |

## Reproduce the tuning

Run this from the repository root.

```sh
python src/tune.py
```

The script compares lookback lengths for Gaussian Naive Bayes and logistic regression. With a three-day lookback, it also compares logistic regression learning rates and iteration counts, and different values of `k` for KNN. We compared settings using mean validation ROC-AUC across the six locations. The script also prints mean validation F1. It fits on training data and scores the settings on validation data.

## Contributions

- **Charleen Airam Casal** implemented Gaussian Naive Bayes and logistic regression and ran the model tuning.
- **Jan Vincent Gloria** worked on project structure, initial data fetching, preprocessing, selecting the study locations, moved the comparison into its own file, added logistic regression to the comparison and plots, and worked on the project page and README.
- **Lance Perus** wrote the preprocessing and chronological split, implemented the evaluation metrics, and prepared the final comparison tables and figures.
- **Max Ivan Sangkal** implemented KNN from scratch and revised it for the model comparison.

Everyone helped run and review the initial results.

## Acknowledgement

During the preparation of this work, the author(s) used the following AI tools:

- **Claude Sonnet 5.5** helped us write and improve the tuning script and to help optimize the knn to load faster.
- **ChatGPT 5.6 Luna** helped us in ideating, testing, debugging, research, code architecture, and MD formatting and drafting.
- **Google AI Overview** helped us with python syntax, debugging, ideating, and research.

After using these tools, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content.
