# Invoked by main then it does the following:
# 1. KNN, Gaussian NaiveBayes and Logistic Regression results 
# 2. Compare all three on validation, then evaluate once on the test set
# 3. Pass to plot.py for comparison tables with Matplotlib

from models.gaussian_naive_bayes import GaussianNaiveBayes
from models.knn import predict_knn
from models.logistic_regression import LogisticRegression
from metrics import evaluate_predictions, find_best_threshold, roc_auc_score
K_VALUE = 151   # KNN: number of neighbours (tune)
P_VALUE = 2    # KNN: distance type (2 = Euclidean)

def run_gaussian_naive_bayes(preprocessed_data):
    """Return validation and test rain scores by location and print validation metrics."""
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

    if not models:
        return {}

    scores_by_location = {}
    for location, model in models.items():
        data = preprocessed_data[location]
        X_val, X_test = data["X_val"], data["X_test"]
        scores_by_location[location] = {
            "val_scores": model.predict_proba(X_val) if len(X_val) else [],
            "test_scores": model.predict_proba(X_test) if len(X_test) else [],
        }

    # [1] Raw output: what fit() learned
    print("\n[1] Learned priors (index 0 = Dry, 1 = Rain)")
    for location, model in models.items():
        print(f"  {location:16s} P(Dry)={model.priors_[0]:.3f}  P(Rain)={model.priors_[1]:.3f}")

    # [2] Raw output: first 10 validation days of the first location
    first = next(iter(models))
    data = preprocessed_data[first]
    probs = scores_by_location[first]["val_scores"][:10]
    print(f"\n[2] Sample predictions, {first} (first 10 validation days)")
    print(f"  {'Day':>3s} {'P(rain)':>8s} {'Pred@0.5':>9s} {'Actual':>7s}")
    for i, p in enumerate(probs):
        print(f"  {i:3d} {p:8.3f} {int(p >= 0.5):9d} {data['y_val'][i]:7d}")

    # [3] Metrics vs. the always-rain baseline
    print("\n[3] Metrics (threshold tuned on validation F1)")
    print(f"  {'Location':16s} {'Thr':>5s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Acc':>6s} "
          f"{'AUC':>6s} | {'AlwaysRain F1':>13s} {'AlwaysRain Acc':>14s}")
    results = []

    for location, scores in scores_by_location.items():
        y_val = preprocessed_data[location]["y_val"]
        val_scores = scores["val_scores"]
        best_threshold, _ = find_best_threshold(y_val, val_scores)

        val_pred = [1 if score >= best_threshold else 0 for score in val_scores]
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

    return scores_by_location


def run_knn(preprocessed_data):
    print("\n" + "=" * 78)
    print("K-Nearest Neighbors: initial results (VALIDATION set)")
    print("=" * 78)

    k_value, p_value = K_VALUE, P_VALUE

    scores_by_location = {}
    for location, city_data in preprocessed_data.items():
        X_train, y_train = city_data["X_train"], city_data["y_train"]
        X_val, X_test = city_data["X_val"], city_data["X_test"]
        val_scores, _ = predict_knn(X_train, y_train, X_val, k_value, p_value)
        test_scores, _ = predict_knn(X_train, y_train, X_test, k_value, p_value)
        scores_by_location[location] = {
            "val_scores": val_scores,
            "test_scores": test_scores,
        }

    # [1] Raw output: first 10 validation days of the first location
    first = next(iter(scores_by_location))
    probs = scores_by_location[first]["val_scores"][:10]
    print(f"\n[1] Sample predictions, {first} (first 10 validation days, k={k_value}, p={p_value})")
    print(f"  {'Day':>3s} {'P(rain)':>8s} {'Pred@0.5':>9s} {'Actual':>7s}")
    for i, p in enumerate(probs):
        print(f"  {i:3d} {p:8.3f} {int(p >= 0.5):9d} {preprocessed_data[first]['y_val'][i]:7d}")

    # [2] Metrics vs. the always-rain baseline
    print("\n[2] Metrics (threshold tuned on validation F1)")
    print(f"  {'Location':16s} {'Thr':>5s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Acc':>6s} "
          f"{'AUC':>6s} | {'AlwaysRain F1':>13s} {'AlwaysRain Acc':>14s}")
    results = []

    for location, scores in scores_by_location.items():
        y_val = preprocessed_data[location]["y_val"]
        val_scores = scores["val_scores"]
        best_threshold, _ = find_best_threshold(y_val, val_scores)

        val_pred = [1 if score >= best_threshold else 0 for score in val_scores]
        r = evaluate_predictions(y_val, val_pred)
        auc = roc_auc_score(y_val, val_scores)

        # Baseline: always predict the majority class (rain)
        base = evaluate_predictions(y_val, [1] * len(y_val))

        print(f"  {location:16s} {best_threshold:5.2f} {r['f1']:6.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['accuracy']:6.3f} {auc:6.3f} | "
              f"{base['f1']:13.3f} {base['accuracy']:14.3f}")
        results.append((location, r))

    # [3] Confusion matrix counts at the tuned threshold
    print("\n[3] Confusion matrix counts")
    print(f"  {'Location':16s} {'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s}")
    for location, r in results:
        print(f"  {location:16s} {r['tp']:5d} {r['fp']:5d} {r['tn']:5d} {r['fn']:5d}")

    return scores_by_location


def run_logistic_regression(preprocessed_data):
    print("\n" + "=" * 78)
    print("Logistic Regression: initial results (VALIDATION set)")
    print("=" * 78)

    # Train one model per location (TRAIN data only)
    models = {}
    for location, data in preprocessed_data.items():
        try:
            models[location] = LogisticRegression().fit(data["X_train"], data["y_train"])
        except ValueError as error:
            print(f"[Warning] Logistic Regression failed for {location}: {error}")

    if not models:
        return {}

    scores_by_location = {}
    for location, model in models.items():
        data = preprocessed_data[location]
        X_val, X_test = data["X_val"], data["X_test"]
        scores_by_location[location] = {
            "val_scores": model.predict_proba(X_val) if len(X_val) else [],
            "test_scores": model.predict_proba(X_test) if len(X_test) else [],
        }

    # [1] Raw output: first 10 validation days of the first location
    first = next(iter(models))
    data = preprocessed_data[first]
    probs = scores_by_location[first]["val_scores"][:10]
    print(f"\n[1] Sample predictions, {first} (first 10 validation days)")
    print(f"  {'Day':>3s} {'P(rain)':>8s} {'Pred@0.5':>9s} {'Actual':>7s}")
    for i, p in enumerate(probs):
        print(f"  {i:3d} {p:8.3f} {int(p >= 0.5):9d} {data['y_val'][i]:7d}")

    # [2] Metrics vs. the always-rain baseline
    print("\n[2] Metrics (threshold tuned on validation F1)")
    print(f"  {'Location':16s} {'Thr':>5s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Acc':>6s} "
          f"{'AUC':>6s} | {'AlwaysRain F1':>13s} {'AlwaysRain Acc':>14s}")
    results = []

    for location, scores in scores_by_location.items():
        y_val = preprocessed_data[location]["y_val"]
        val_scores = scores["val_scores"]
        best_threshold, _ = find_best_threshold(y_val, val_scores)

        val_pred = [1 if score >= best_threshold else 0 for score in val_scores]
        r = evaluate_predictions(y_val, val_pred)
        auc = roc_auc_score(y_val, val_scores)

        # Baseline: always predict the majority class (rain)
        base = evaluate_predictions(y_val, [1] * len(y_val))

        print(f"  {location:16s} {best_threshold:5.2f} {r['f1']:6.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['accuracy']:6.3f} {auc:6.3f} | "
              f"{base['f1']:13.3f} {base['accuracy']:14.3f}")
        results.append((location, r))

    # [3] Confusion matrix counts at the tuned threshold
    print("\n[3] Confusion matrix counts")
    print(f"  {'Location':16s} {'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s}")
    for location, r in results:
        print(f"  {location:16s} {r['tp']:5d} {r['fp']:5d} {r['tn']:5d} {r['fn']:5d}")

    return scores_by_location


def evaluate_test_set(preprocessed_data, gnb_result, knn_result, lr_result):
    """Evaluate all three models on the held-out TEST set using validation-tuned thresholds."""
    print("\n" + "=" * 78)
    print("Final Model Comparison on Held-Out TEST Set (threshold tuned on validation F1)")
    print("=" * 78)

    model_specs = [
        ("Gaussian Naive Bayes", gnb_result),
        (f"K-Nearest Neighbors (k={K_VALUE})", knn_result),
        ("Logistic Regression", lr_result),
    ]

    overall_rows = []
    for model_name, model_scores in model_specs:
        if not model_scores:
            continue

        print(f"\n[{model_name}]")
        print(f"  {'Location':16s} {'Thr':>5s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Acc':>6s} "
              f"{'AUC':>6s} | {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s}")

        f1_list, prec_list, rec_list, acc_list, auc_list = [], [], [], [], []
        pooled_y, pooled_pred, pooled_scores = [], [], []

        for location, data in preprocessed_data.items():
            if location not in model_scores:
                continue
            y_val = data["y_val"]
            y_test = data["y_test"]
            val_scores = model_scores[location]["val_scores"]
            test_scores = model_scores[location]["test_scores"]

            best_thr, _ = find_best_threshold(y_val, val_scores)
            test_pred = [1 if s >= best_thr else 0 for s in test_scores]
            r = evaluate_predictions(y_test, test_pred)
            auc = roc_auc_score(y_test, test_scores)

            print(f"  {location:16s} {best_thr:5.2f} {r['f1']:6.3f} {r['precision']:6.3f} "
                  f"{r['recall']:6.3f} {r['accuracy']:6.3f} {auc:6.3f} | "
                  f"{r['tp']:4d} {r['fp']:4d} {r['tn']:4d} {r['fn']:4d}")

            f1_list.append(r["f1"])
            prec_list.append(r["precision"])
            rec_list.append(r["recall"])
            acc_list.append(r["accuracy"])
            auc_list.append(auc)
            pooled_y.extend(y_test)
            pooled_pred.extend(test_pred)
            pooled_scores.extend(test_scores)

        if f1_list:
            macro_f1 = sum(f1_list) / len(f1_list)
            macro_prec = sum(prec_list) / len(prec_list)
            macro_rec = sum(rec_list) / len(rec_list)
            macro_acc = sum(acc_list) / len(acc_list)
            macro_auc = sum(auc_list) / len(auc_list)
            pooled_r = evaluate_predictions(pooled_y, pooled_pred)
            pooled_auc = roc_auc_score(pooled_y, pooled_scores)
            print(f"  {'Macro Average':16s} {'-':>5s} {macro_f1:6.3f} {macro_prec:6.3f} "
                  f"{macro_rec:6.3f} {macro_acc:6.3f} {macro_auc:6.3f} | "
                  f"{pooled_r['tp']:4d} {pooled_r['fp']:4d} {pooled_r['tn']:4d} {pooled_r['fn']:4d}")
            overall_rows.append((model_name, macro_f1, macro_prec, macro_rec, macro_acc, macro_auc, pooled_r, pooled_auc))

    if overall_rows:
        print("\n[Overall Test Set Summary Across All 6 Locations (Macro Average)]")
        print(f"  {'Model':28s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Acc':>6s} {'AUC':>6s}")
        for model_name, mf1, mprec, mrec, macc, mauc, _, _ in overall_rows:
            print(f"  {model_name:28s} {mf1:6.3f} {mprec:6.3f} {mrec:6.3f} {macc:6.3f} {mauc:6.3f}")


def run_compare(preprocessed_data):
    gnb_result = run_gaussian_naive_bayes(preprocessed_data)
    knn_result = run_knn(preprocessed_data)
    lr_result = run_logistic_regression(preprocessed_data)

    evaluate_test_set(preprocessed_data, gnb_result, knn_result, lr_result)

    # Generate Matplotlib comparison figures
    try:
        from plot import generate_all_plots
        generate_all_plots(preprocessed_data, gnb_result, knn_result, lr_result)
    except Exception as e:
        print(f"[Warning] Failed to generate comparison plots: {e}")