# References:
# - Powers (2011) Evaluation Metrics: https://www.researchgate.net/publication/228529307
# - Fawcett (2006) ROC Analysis: https://doi.org/10.1016/j.patrec.2005.10.010

# Ref: Powers (2011); binary confusion matrix with Rain (1) as positive class
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, fp, tn, fn


# Ref: Powers (2011); Accuracy = (TP + TN) / Total
def accuracy(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0 else 0.0


# Ref: Powers (2011); Precision = TP / (TP + FP)
def precision(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


# Ref: Powers (2011); Recall (Sensitivity) = TP / (TP + FN)
def recall(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


# Ref: Powers (2011); F1-score harmonic mean: balances Precision and Recall
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


# Ref: Empirical threshold calibration on validation set to maximize F1 under class imbalance
def find_best_threshold(y_true, y_scores, start=0.02, end=0.98, step=0.02):
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


# Ref: Fawcett (2006); ROC analysis mapping TPR vs FPR across decision thresholds
def roc_curve(y_true, y_scores, num_points=101):
    unique_scores = sorted(set(y_scores), reverse=True)
    if num_points and num_points > 1:
        grid_thresholds = [i / (num_points - 1) for i in range(num_points)]
        thresholds = sorted(set(unique_scores + grid_thresholds), reverse=True)
    else:
        thresholds = unique_scores

    max_score = max(y_scores) if y_scores else 1.0
    thresholds = [max_score + 1.0] + thresholds

    fpr_list = []
    tpr_list = []
    used_thresholds = []

    for tau in thresholds:
        y_pred = [1 if s >= tau else 0 for s in y_scores]
        tp, fp, tn, fn = confusion_matrix(y_true, y_pred)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        used_thresholds.append(tau)

    # Sort monotonically by FPR, then TPR
    points = sorted(zip(fpr_list, tpr_list, used_thresholds), key=lambda p: (p[0], p[1]))
    return [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]


# Ref: Composite trapezoidal numerical integration rule for ROC-AUC
def roc_auc_score(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0

    return max(0.0, min(1.0, auc))
