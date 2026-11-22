# DECLARATION OF USE OF GENERATIVE AI
# During the preparation of this work, the author(s) used Claude Sonnet 5.5 to help write and improve the tuning script.  
# After using this tool, the author(s) reviewed and edited the content as  needed and take(s) full responsibility for the content.

# Tuning script: tries different settings and compares them on the VALIDATION set only
# Run: python src/tune.py

from data.nasa_api import LOCATIONS, get_location_weather
from data.preprocess import preprocess_location
from models.gaussian_naive_bayes import GaussianNaiveBayes
from models.logistic_regression import LogisticRegression, LEARNING_RATE, N_ITERATIONS
from models.knn import predict_knn
from metrics import find_best_threshold, roc_auc_score
from comparison import K_VALUE, P_VALUE
from main import START_DATE, END_DATE, WINDOW_SIZE, RAIN_THRESHOLD

# values to try (edit these lists to try others; use KNN_KS = [] to skip KNN, it is slow)
WINDOW_SIZES = [3, 5, 7, 10, 14]
LR_RATES = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
LR_ITERATIONS = [100, 300, 500, 1000, 2000, 3000]
KNN_KS = [5, 15, 25, 51, 75, 101, 151]


def load_rows():
    # download or load the raw weather once per location
    rows_by_location = {}
    for location, (latitude, longitude) in LOCATIONS.items():
        rows_by_location[location] = get_location_weather(
            location_name=location, latitude=latitude, longitude=longitude,
            start_date=START_DATE, end_date=END_DATE,
        )
    return rows_by_location


def preprocess_all(rows_by_location, window_size):
    return {
        location: preprocess_location(rows, window_size=window_size, rain_threshold=RAIN_THRESHOLD)
        for location, rows in rows_by_location.items()
    }


# each function trains on TRAIN and returns P(rain) for the VALIDATION days
def gnb_val_scores(d):
    return GaussianNaiveBayes().fit(d["X_train"], d["y_train"]).predict_proba(d["X_val"])


def lr_val_scores(d, learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS):
    model = LogisticRegression(learning_rate, n_iterations).fit(d["X_train"], d["y_train"])
    return model.predict_proba(d["X_val"])


def knn_val_scores(d, k):
    scores, _ = predict_knn(d["X_train"], d["y_train"], d["X_val"], k, P_VALUE)
    return scores


def average_scores(data, get_scores):
    # average validation AUC and best-threshold F1 across all locations
    aucs, f1s = [], []
    for d in data.values():
        scores = get_scores(d)
        aucs.append(roc_auc_score(d["y_val"], scores))
        f1s.append(find_best_threshold(d["y_val"], scores)[1])
    return sum(aucs) / len(aucs), sum(f1s) / len(f1s)


def show(label, result):
    auc, f1 = result
    print(f"  {label:34s} val AUC = {auc:.4f} | val F1 = {f1:.4f}")


def main():
    rows_by_location = load_rows()

    print("\n[1] Window size (default settings for each model)")
    for window in WINDOW_SIZES:
        data = preprocess_all(rows_by_location, window)
        show(f"GNB, window = {window} days", average_scores(data, gnb_val_scores))
        show(f"LR,  window = {window} days", average_scores(data, lr_val_scores))

    # the next sections use WINDOW_SIZE from main.py
    data = preprocess_all(rows_by_location, WINDOW_SIZE)

    print(f"\n[2] LR learning rate and iterations (window = {WINDOW_SIZE} days)")
    for rate in LR_RATES:
        for iterations in LR_ITERATIONS:
            show(f"LR, rate = {rate}, iters = {iterations}",
                 average_scores(data, lambda d: lr_val_scores(d, rate, iterations)))

    print(f"\n[3] KNN k (window = {WINDOW_SIZE} days, p = {P_VALUE})")
    for k in KNN_KS:
        show(f"KNN, k = {k}", average_scores(data, lambda d: knn_val_scores(d, k)))


if __name__ == "__main__":
    main()