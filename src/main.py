from data.nasa_api import LOCATIONS, get_location_weather
from data.preprocess import preprocess_location
from models.gaussian_naive_bayes import GaussianNaiveBayes
from metrics import evaluate_predictions, find_best_threshold, roc_auc_score

START_DATE = "20000101"
END_DATE = "20241231"

def run_gaussian_naive_bayes(preprocessed_data):
    """Train on TRAIN, tune the threshold on VALIDATION, print initial results.
    The test set is NOT used here (it is saved for the final evaluation)."""
    print("\n" + "=" * 78)
    print("Gaussian Naive Bayes: initial results (VALIDATION set)")
    print("=" * 78)
    print(f"{'Location':16s} {'Thr':>5s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s} "
          f"{'Acc':>7s} {'AUC':>7s} {'AlwaysDry Acc':>14s}")

    for location, data in preprocessed_data.items():
        try:
            model = GaussianNaiveBayes()                    # make a fresh model
            model.fit(data["X_train"], data["y_train"])     # learn from training data only

            # P(rain) for each validation day, then pick the threshold with the best F1.
            val_scores = model.predict_proba(data["X_val"])
            best_threshold, _ = find_best_threshold(data["y_val"], val_scores)

            # Final yes/no predictions using that threshold, then score them.
            val_pred = model.predict(data["X_val"], threshold=best_threshold)
            result = evaluate_predictions(data["y_val"], val_pred)
            auc = roc_auc_score(data["y_val"], val_scores)

            # Baseline: accuracy if we always said "dry".
            always_dry_acc = 1 - sum(data["y_val"]) / len(data["y_val"])

            print(f"{location:16s} {best_threshold:5.2f} {result['f1']:7.3f} "
                  f"{result['precision']:7.3f} {result['recall']:7.3f} "
                  f"{result['accuracy']:7.3f} {auc:7.3f} {always_dry_acc:14.3f}")
        except ValueError as error:
            print(f"[Warning] Gaussian Naive Bayes failed for {location}: {error}")


def main():
    print("=" * 65)
    print("CMSC 170 - Group BAWWL: Weather Pipeline")
    print("=" * 65)

    # Ingest and preprocess data for each location
    preprocessed_data = {}

    for location, (latitude, longitude) in LOCATIONS.items():
        print(f"\nProcessing {location}...")
        try:
            rows = get_location_weather(
                location_name=location,
                latitude=latitude,
                longitude=longitude,
                start_date=START_DATE,
                end_date=END_DATE,
            )
            # 5-day lookback window with 1.0 mm rain threshold
            data = preprocess_location(rows, window_size=5, rain_threshold=1.0)
            preprocessed_data[location] = data
            print(f"  Train: {len(data['X_train'])} samples | Val: {len(data['X_val'])} | Test: {len(data['X_test'])}")
            print(f"  Features per sample: {data['feature_count']} (5 days x 11 daily features)")
        except Exception as error:
            print(f"[Warning] Failed to process {location}: {error}")

    # Pipeline summary
    print("\n" + "=" * 65)
    print("Preprocessing Summary (Ready for Model Training)")
    print("=" * 65)
    for location, data in preprocessed_data.items():
        rain_rate = sum(data["y_train"]) / len(data["y_train"]) * 100
        print(f"- {location:16s}: {len(data['X_train']):4d} train | {len(data['X_test']):4d} test | Train Rain: {rain_rate:5.1f}%")

    # Model results
    run_gaussian_naive_bayes(preprocessed_data)

    # Next steps:
    # 1. KNN and Logistic Regression results (same format as above)
    # 2. Compare all three on validation, then evaluate once on the test set
    # 3. ROC curves and comparison tables with Matplotlib


if __name__ == "__main__":
    main()
