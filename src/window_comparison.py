# Sliding Window Lookback Comparison Suite (3, 5, 7 Days)
#
# GOAL: Evaluate how temporal lookback depth affects next-day rainfall forecasting.
#       Compares 3-day (33 features), 5-day (55 features), and 7-day (77 features)
#       sliding windows across CART Decision Tree, Logistic Regression, KNN, and GNB.
#
# References:
#   - WMO No. 8 Guide to Meteorological Instruments
#   - Lorenz (1969) Atmospheric Predictability / Analog Forecasting

import sys
import os
import argparse
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.nasa_api import LOCATIONS, get_location_weather
from data.preprocess import preprocess_location
from models.decision_tree import CARTDecisionTree
from models.logistic_regression import LogisticRegression
from models.knn import predict_knn
from models.gaussian_naive_bayes import GaussianNaiveBayes
from metrics import evaluate_predictions, find_best_threshold, roc_auc_score
from plot import plot_window_size_comparison

START_DATE = "20000101"
END_DATE = "20241231"


def run_window_size_experiments(
    window_sizes=(3, 5, 7),
    locations=None,
    generate_plot=True,
    verbose=True,
):
    """
    Run experiments across multiple sliding window sizes (e.g. 3, 5, 7 days).

    Parameters:
    -----------
    window_sizes : tuple or list of int, default=(3, 5, 7)
        Window sizes in days.
    locations : dict or list, optional
        Subset of locations to evaluate. Defaults to all project LOCATIONS.
    generate_plot : bool, default=True
        Whether to generate and save 'figures/window_size_comparison.png'.
    verbose : bool, default=True
        Whether to print progress and evaluation tables.

    Returns:
    --------
    summary_results : dict
        Mapping window_size -> model_name -> dict of metrics.
    """
    if locations is None:
        target_locations = LOCATIONS
    elif isinstance(locations, list):
        target_locations = {loc: LOCATIONS[loc] for loc in locations if loc in LOCATIONS}
    else:
        target_locations = locations

    # Cache raw daily records in memory so we don't reload CSVs multiple times
    if verbose:
        print("=" * 80)
        print("CMSC 170: Sliding Window Size Comparison Experiment (3, 5, 7 Days)")
        print("=" * 80)
        print(f"Loading raw meteorological records for {len(target_locations)} stations...")

    raw_weather = {}
    for loc, (lat, lon) in target_locations.items():
        raw_weather[loc] = get_location_weather(
            location_name=loc,
            latitude=lat,
            longitude=lon,
            start_date=START_DATE,
            end_date=END_DATE,
        )

    model_names = [
        "Always-Rain Baseline",
        "Gaussian Naive Bayes",
        "KNN (k=15)",
        "Logistic Regression",
        "CART Decision Tree",
    ]

    summary_results = {}
    detailed_by_location = {}

    for w in sorted(window_sizes):
        if verbose:
            print("\n" + "=" * 80)
            print(f"Evaluating Window Size W = {w} Days ({w * 11} features per sample)")
            print("=" * 80)

        # Preprocess each location for window size w
        preprocessed = {}
        for loc, rows in raw_weather.items():
            preprocessed[loc] = preprocess_location(rows, window_size=w, rain_threshold=1.0)

        # Aggregators for nationwide test performance
        nationwide = {m: {"y_true": [], "y_pred": [], "scores": []} for m in model_names}
        loc_results_w = {}

        for loc, data in preprocessed.items():
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]

            # 1. Baseline
            base_pred = [1] * len(y_test)
            base_scores = [sum(y_train) / len(y_train)] * len(y_test)
            nationwide["Always-Rain Baseline"]["y_true"].extend(y_test)
            nationwide["Always-Rain Baseline"]["y_pred"].extend(base_pred)
            nationwide["Always-Rain Baseline"]["scores"].extend(base_scores)

            # 2. Gaussian Naive Bayes
            gnb = GaussianNaiveBayes().fit(X_train, y_train)
            gnb_val = gnb.predict_proba(X_val)
            gnb_thr, _ = find_best_threshold(y_val, gnb_val)
            gnb_scores = gnb.predict_proba(X_test)
            gnb_pred = [1 if s >= gnb_thr else 0 for s in gnb_scores]
            nationwide["Gaussian Naive Bayes"]["y_true"].extend(y_test)
            nationwide["Gaussian Naive Bayes"]["y_pred"].extend(gnb_pred)
            nationwide["Gaussian Naive Bayes"]["scores"].extend(gnb_scores)

            # 3. KNN (k=15, p=2)
            knn_val_scores, _ = predict_knn(X_train, y_train, X_val, 15, 2)
            knn_scores, knn_pred = predict_knn(X_train, y_train, X_test, 15, 2)
            nationwide["KNN (k=15)"]["y_true"].extend(y_test)
            nationwide["KNN (k=15)"]["y_pred"].extend(knn_pred)
            nationwide["KNN (k=15)"]["scores"].extend(knn_scores)

            # 4. Logistic Regression
            lr = LogisticRegression().fit(X_train, y_train)
            lr_val = lr.predict_proba(X_val)
            lr_thr, _ = find_best_threshold(y_val, lr_val)
            lr_scores = lr.predict_proba(X_test)
            lr_pred = [1 if s >= lr_thr else 0 for s in lr_scores]
            nationwide["Logistic Regression"]["y_true"].extend(y_test)
            nationwide["Logistic Regression"]["y_pred"].extend(lr_pred)
            nationwide["Logistic Regression"]["scores"].extend(lr_scores)

            # 5. CART Decision Tree
            cart = CARTDecisionTree(
                max_depth=5, min_samples_split=50, min_samples_leaf=25
            ).fit(X_train, y_train)
            cart_val = cart.predict_proba(X_val)
            cart_thr, _ = find_best_threshold(y_val, cart_val)
            cart_scores = cart.predict_proba(X_test)
            cart_pred = [1 if s >= cart_thr else 0 for s in cart_scores]
            nationwide["CART Decision Tree"]["y_true"].extend(y_test)
            nationwide["CART Decision Tree"]["y_pred"].extend(cart_pred)
            nationwide["CART Decision Tree"]["scores"].extend(cart_scores)

        # Compute nationwide summary for window size w
        summary_results[w] = {}
        for m in model_names:
            yt = nationwide[m]["y_true"]
            yp = nationwide[m]["y_pred"]
            sc = nationwide[m]["scores"]
            eval_res = evaluate_predictions(yt, yp)
            auc = roc_auc_score(yt, sc) if m != "Always-Rain Baseline" else 0.50
            brier = float(np.mean((np.array(yt) - np.array(sc)) ** 2))
            summary_results[w][m] = {
                "accuracy": eval_res["accuracy"],
                "precision": eval_res["precision"],
                "recall": eval_res["recall"],
                "f1": eval_res["f1"],
                "auc": auc,
                "brier": brier,
            }

        if verbose:
            print(f"{'Model':<24s} {'Accuracy':<10s} {'Precision':<11s} {'Recall':<10s} {'F1-Score':<10s} {'ROC-AUC':<10s} {'Brier(MSE)':<12s}")
            print("-" * 87)
            for m in model_names:
                met = summary_results[w][m]
                print(
                    f"{m:<24s} {met['accuracy']:>8.4f}   {met['precision']:>8.4f}    "
                    f"{met['recall']:>8.4f}   {met['f1']:>8.4f}   {met['auc']:>8.4f}   {met['brier']:>10.4f}"
                )

    # Consolidated Comparison Matrix
    if verbose:
        print("\n" + "=" * 80)
        print("CONSOLIDATED COMPARISON ACROSS WINDOW SIZES (3, 5, 7 DAYS)")
        print("=" * 80)
        print(f"{'Model':<24s} | {'W=3 F1':<7s} {'W=5 F1':<7s} {'W=7 F1':<7s} | {'W=3 Acc':<8s} {'W=5 Acc':<8s} {'W=7 Acc':<8s} | {'W=3 AUC':<8s} {'W=5 AUC':<8s} {'W=7 AUC':<8s}")
        print("-" * 105)
        for m in model_names:
            f1_3 = summary_results.get(3, {}).get(m, {}).get("f1", 0.0)
            f1_5 = summary_results.get(5, {}).get(m, {}).get("f1", 0.0)
            f1_7 = summary_results.get(7, {}).get(m, {}).get("f1", 0.0)

            acc_3 = summary_results.get(3, {}).get(m, {}).get("accuracy", 0.0)
            acc_5 = summary_results.get(5, {}).get(m, {}).get("accuracy", 0.0)
            acc_7 = summary_results.get(7, {}).get(m, {}).get("accuracy", 0.0)

            auc_3 = summary_results.get(3, {}).get(m, {}).get("auc", 0.0)
            auc_5 = summary_results.get(5, {}).get(m, {}).get("auc", 0.0)
            auc_7 = summary_results.get(7, {}).get(m, {}).get("auc", 0.0)

            print(
                f"{m:<24s} | {f1_3:6.4f}  {f1_5:6.4f}  {f1_7:6.4f}  | "
                f"{acc_3:7.4f}  {acc_5:7.4f}  {acc_7:7.4f}  | "
                f"{auc_3:7.4f}  {auc_5:7.4f}  {auc_7:7.4f}"
            )

    # Generate Matplotlib visualization
    if generate_plot:
        plot_window_size_comparison(summary_results)

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliding Window Size Experiments (3, 5, 7 Days)")
    parser.add_argument(
        "--windows",
        type=str,
        default="3,5,7",
        help="Comma-separated window sizes (e.g. '3,5,7')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run on core 6 synoptic stations only for quick verification",
    )
    args = parser.parse_args()

    w_sizes = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    loc_subset = [
        "Metro Manila", "Baguio", "Legazpi City", "Cebu City", "Davao City", "Zamboanga City"
    ] if args.quick else None

    run_window_size_experiments(window_sizes=w_sizes, locations=loc_subset)
