# Invoked by main then it does the following:
# 1. KNN, Gaussian NaiveBayes and Logistic Regression results 
# 2. Compare all three on validation, then evaluate once on the test set
# 3. Pass to plot.py for comparison tables with Matplotlib

from models.gaussian_naive_bayes import GaussianNaiveBayes
from metrics import evaluate_predictions, find_best_threshold, roc_auc_score
from models.knn import predict_knn

def run_gaussian_naive_bayes(preprocessed_data):
    """Train on TRAIN, tune the threshold on VALIDATION, print initial results.
    The test set is NOT used here (it is saved for the final evaluation)."""
    print("\n" + "=" * 78)
    print("Gaussian Naive Bayes: initial results (VALIDATION set)")
    print("=" * 78)

    # Train one model per location (TRAIN data only)
    models = {}
    for location, data in preprocessed_data.items():
        try:
            models[location] = GaussianNaiveBayes().fit(data["X_train"], data["y_train"])
        except ValueError as error:
            print(f"[Warning] Gaussian Naive Bayes failed for {location}: {error}")

    # [1] Raw output: what fit() learned
    print("\n[1] Learned priors (index 0 = Dry, 1 = Rain)")
    for location, model in models.items():
        print(f"  {location:16s} P(Dry)={model.priors_[0]:.3f}  P(Rain)={model.priors_[1]:.3f}")

    # [2] Raw output: first 10 validation days of the first location
    first = next(iter(models))
    data = preprocessed_data[first]
    probs = models[first].predict_proba(data["X_val"][:10])
    print(f"\n[2] Sample predictions, {first} (first 10 validation days)")
    print(f"  {'Day':>3s} {'P(rain)':>8s} {'Pred@0.5':>9s} {'Actual':>7s}")
    for i, p in enumerate(probs):
        print(f"  {i:3d} {p:8.3f} {int(p >= 0.5):9d} {data['y_val'][i]:7d}")

    # [3] Metrics vs. the always-rain baseline
    print("\n[3] Metrics (threshold tuned on validation F1)")
    print(f"  {'Location':16s} {'Thr':>5s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Acc':>6s} "
          f"{'AUC':>6s} | {'AlwaysRain F1':>13s} {'AlwaysRain Acc':>14s}")
    results = []

    for location, model in models.items():
        data = preprocessed_data[location]
        y_val = data["y_val"]
        val_scores = model.predict_proba(data["X_val"])
        best_threshold, _ = find_best_threshold(y_val, val_scores)

        val_pred = model.predict(data["X_val"], threshold=best_threshold)
        r = evaluate_predictions(y_val, val_pred)
        auc = roc_auc_score(y_val, val_scores)

        # Baseline: always predict the majority class (rain)
        base = evaluate_predictions(y_val, [1] * len(y_val))

        print(f"  {location:16s} {best_threshold:5.2f} {r['f1']:6.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['accuracy']:6.3f} {auc:6.3f} | "
              f"{base['f1']:13.3f} {base['accuracy']:14.3f}")
        results.append((location, r))

    # [4] Confusion matrix counts at the tuned threshold
    print("\n[4] Confusion matrix counts")
    print(f"  {'Location':16s} {'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s}")
    for location, r in results:
        print(f"  {location:16s} {r['tp']:5d} {r['fp']:5d} {r['tn']:5d} {r['fn']:5d}")

    return results


def run_knn(data):
    k_value, p_value = 15, 2
    print("K-nearest neighbour output")
    print("k-value: ", k_value, "\t", "p-value: ", p_value)

    results = {}
    all_labels = []
    all_predictions = []
    for location, city_data in data.items():
        X_train, y_train = city_data["X_train"], city_data["y_train"]
        X_test, y_test = city_data["X_test"], city_data["y_test"]
        print(location, "train:", len(X_train), "-> test:", len(X_test))
        _, predictions = predict_knn(X_train, y_train, X_test, k_value, p_value)
        results[location] = evaluate_predictions(y_test, predictions)
        all_labels.extend(y_test)
        all_predictions.extend(predictions)

    results["OVERALL"] = evaluate_predictions(all_labels, all_predictions)
    for location, result in results.items():
        print("\n" + location)
        print(
            "  Confusion matrix [[TN FP] [FN TP]]:",
            [[result["tn"], result["fp"]], [result["fn"], result["tp"]]],
        )
        print("  Accuracy :", round(result["accuracy"], 4))
        print("  Precision:", round(result["precision"], 4))
        print("  Recall   :", round(result["recall"], 4))
        print("  F1-score :", round(result["f1"], 4))

    return results

def run_compare(preprocessed_data):
    gnb_result = run_gaussian_naive_bayes(preprocessed_data)
    knn_result = run_knn(preprocessed_data)