# Confusion matrix counts: TP, FP, TN, FN
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, fp, tn, fn


# Overall accuracy: (TP + TN) / Total
def accuracy(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0 else 0.0


# Precision: TP / (TP + FP)
def precision(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


# Recall (Sensitivity): TP / (TP + FN)
def recall(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


# F1-score: harmonic mean of Precision and Recall
def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    denom = prec + rec
    return 2.0 * (prec * rec) / denom if denom > 0 else 0.0


# Evaluate predictions and return all core metrics
def evaluate_predictions(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


# Sweep thresholds on validation set to maximize F1-score
def find_best_threshold(y_true, y_scores, start=0.10, end=0.90, step=0.05):
    best_threshold = 0.50
    best_f1 = -1.0

    current = start
    while current <= end + 1e-9:
        y_pred = [1 if s >= current else 0 for s in y_scores]
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = round(current, 3)
        current += step

    return best_threshold, best_f1


# Generate ROC curve coordinates: FPR and TPR across thresholds
def roc_curve(y_true, y_scores, num_points=101):
    thresholds = [i / (num_points - 1) for i in range(num_points)]
    thresholds.reverse()

    fpr_list = []
    tpr_list = []

    for tau in thresholds:
        y_pred = [1 if s >= tau else 0 for s in y_scores]
        tp, fp, tn, fn = confusion_matrix(y_true, y_pred)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    # Sort monotonically by FPR
    points = sorted(zip(fpr_list, tpr_list, thresholds), key=lambda p: (p[0], p[1]))
    return [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]


# Compute Area Under ROC Curve via trapezoidal integration
def roc_auc_score(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0

    return max(0.0, min(1.0, auc))
