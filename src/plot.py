# References:
# - Fawcett (2006) ROC Analysis: https://doi.org/10.1016/j.patrec.2005.10.010
# - Wilks (2011) Statistical Methods in Atmospheric Sciences (Reliability Diagrams)
# - Matplotlib Agg backend: https://matplotlib.org/stable/users/explain/figure/backends.html

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np

from metrics import evaluate_predictions, find_best_threshold, roc_curve, roc_auc_score

DEFAULT_FIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)


LR_COLOR = "#238b57"


def plot_roc_curves(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir=DEFAULT_FIG_DIR):
    """Plot ROC curves for each model across all locations."""
    os.makedirs(output_dir, exist_ok=True)
    locations = list(preprocessed_data.keys())
    num_locs = len(locations)

    cols = 3
    rows = (num_locs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for idx, loc in enumerate(locations):
        ax = axes[idx]
        y_test = preprocessed_data[loc]["y_test"]

        # GNB ROC
        gnb_test = gnb_scores[loc]["test_scores"]
        gnb_fpr, gnb_tpr, _ = roc_curve(y_test, gnb_test)
        gnb_auc = roc_auc_score(y_test, gnb_test)

        # KNN ROC
        knn_test = knn_scores[loc]["test_scores"]
        knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_test)
        knn_auc = roc_auc_score(y_test, knn_test)

        ax.plot(gnb_fpr, gnb_tpr, color="#2b5c8f", lw=2, label=f"GNB (AUC = {gnb_auc:.3f})")
        ax.plot(knn_fpr, knn_tpr, color="#e65c00", lw=2, label=f"KNN (AUC = {knn_auc:.3f})")
        if lr_scores and loc in lr_scores:
            lr_test = lr_scores[loc]["test_scores"]
            lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_test)
            lr_auc = roc_auc_score(y_test, lr_test)
            ax.plot(lr_fpr, lr_tpr, color=LR_COLOR, lw=2, label=f"LR (AUC = {lr_auc:.3f})")

        ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Chance")

        ax.set_title(loc, fontsize=12, fontweight="bold")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

        if idx % cols == 0:
            ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
        if idx >= num_locs - cols:
            ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)

    # Hide unused subplots if any
    for j in range(num_locs, len(axes)):
        axes[j].axis("off")

    fig.suptitle("ROC Curves by Philippine Location (Test Set: 2021-2024)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "roc_curves_by_location.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot saved] ROC curves saved to: {out_path}")
    return out_path


def plot_metrics_comparison(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir=DEFAULT_FIG_DIR):
    """Plot test F1, accuracy, precision, and recall for each model."""
    os.makedirs(output_dir, exist_ok=True)
    locations = list(preprocessed_data.keys())

    metrics_names = ["Accuracy", "F1-score", "Precision", "Recall"]
    gnb_vals = {m: [] for m in metrics_names}
    knn_vals = {m: [] for m in metrics_names}
    lr_vals = {m: [] for m in metrics_names}
    lr_positions = []

    for idx, loc in enumerate(locations):
        y_test = preprocessed_data[loc]["y_test"]

        # GNB evaluation with tuned threshold from val
        y_val = preprocessed_data[loc]["y_val"]
        gnb_val = gnb_scores[loc]["val_scores"]
        best_thr, _ = find_best_threshold(y_val, gnb_val)
        gnb_pred = [1 if s >= best_thr else 0 for s in gnb_scores[loc]["test_scores"]]
        gnb_res = evaluate_predictions(y_test, gnb_pred)

        gnb_vals["Accuracy"].append(gnb_res["accuracy"])
        gnb_vals["F1-score"].append(gnb_res["f1"])
        gnb_vals["Precision"].append(gnb_res["precision"])
        gnb_vals["Recall"].append(gnb_res["recall"])

        # KNN evaluation (0.5 cutoff on voted predictions)
        knn_pred = [1 if s >= 0.5 else 0 for s in knn_scores[loc]["test_scores"]]
        knn_res = evaluate_predictions(y_test, knn_pred)

        knn_vals["Accuracy"].append(knn_res["accuracy"])
        knn_vals["F1-score"].append(knn_res["f1"])
        knn_vals["Precision"].append(knn_res["precision"])
        knn_vals["Recall"].append(knn_res["recall"])

        if lr_scores and loc in lr_scores:
            # Apply the validation F1 threshold to the held-out test scores.
            lr = lr_scores[loc]
            lr_thr, _ = find_best_threshold(y_val, lr["val_scores"])
            lr_pred = [1 if s >= lr_thr else 0 for s in lr["test_scores"]]
            lr_res = evaluate_predictions(y_test, lr_pred)
            lr_positions.append(idx)
            lr_vals["Accuracy"].append(lr_res["accuracy"])
            lr_vals["F1-score"].append(lr_res["f1"])
            lr_vals["Precision"].append(lr_res["precision"])
            lr_vals["Recall"].append(lr_res["recall"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    x = np.arange(len(locations))
    width = 0.25 if lr_positions else 0.35
    gnb_x = x - width if lr_positions else x - width / 2
    knn_x = x if lr_positions else x + width / 2

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        bar_groups = [
            ax.bar(gnb_x, gnb_vals[metric], width, label="Gaussian Naive Bayes", color="#2b5c8f"),
            ax.bar(knn_x, knn_vals[metric], width, label="KNN (k=15)", color="#e65c00"),
        ]
        if lr_positions:
            bar_groups.append(ax.bar(
                x[lr_positions] + width, lr_vals[metric], width,
                label="Logistic Regression", color=LR_COLOR,
            ))

        ax.set_title(f"Test Set {metric}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(locations, rotation=20, ha="right", fontsize=9)
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="lower right")

        # Value labels above bars
        for bars in bar_groups:
            for r in bars:
                h = r.get_height()
                ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Performance Comparison Across Philippine Stations", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot saved] Metrics comparison saved to: {out_path}")
    return out_path


def plot_calibration_curves(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir=DEFAULT_FIG_DIR, num_bins=10):
    """Plot reliability / calibration curves comparing predicted probabilities to empirical rain frequency."""
    os.makedirs(output_dir, exist_ok=True)
    all_y = []
    all_gnb = []
    all_knn = []
    all_lr = []
    all_lr_y = []

    for loc, data in preprocessed_data.items():
        all_y.extend(data["y_test"])
        all_gnb.extend(gnb_scores[loc]["test_scores"])
        all_knn.extend(knn_scores[loc]["test_scores"])
        if lr_scores and loc in lr_scores:
            all_lr.extend(lr_scores[loc]["test_scores"])
            all_lr_y.extend(data["y_test"])

    all_y = np.array(all_y)
    all_gnb = np.array(all_gnb)
    all_knn = np.array(all_knn)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    
    def compute_calibration(probs, labels):
        bin_centers = []
        true_freqs = []
        bin_counts = []
        for i in range(num_bins):
            low, high = bins[i], bins[i+1]
            mask = (probs >= low) & (probs <= high) if i == num_bins - 1 else (probs >= low) & (probs < high)
            if np.sum(mask) > 0:
                bin_centers.append(np.mean(probs[mask]))
                true_freqs.append(np.mean(labels[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_centers.append((low + high) / 2)
                true_freqs.append(np.nan)
                bin_counts.append(0)
        return np.array(bin_centers), np.array(true_freqs), np.array(bin_counts)

    gnb_x, gnb_y_calib, _ = compute_calibration(all_gnb, all_y)
    knn_x, knn_y_calib, _ = compute_calibration(all_knn, all_y)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration (y = x)")
    ax.plot(gnb_x, gnb_y_calib, "s-", color="#2b5c8f", lw=2, ms=6, label="Gaussian Naive Bayes")
    ax.plot(knn_x, knn_y_calib, "o-", color="#e65c00", lw=2, ms=6, label="KNN (k=15)")
    if all_lr:
        lr_x, lr_y_calib, _ = compute_calibration(np.array(all_lr), np.array(all_lr_y))
        ax.plot(lr_x, lr_y_calib, "^-", color=LR_COLOR, lw=2, ms=6, label="Logistic Regression")

    ax.set_title("Probability Calibration Diagram (Reliability Curve)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Forecast Probability P(Rain)", fontsize=11)
    ax.set_ylabel("Empirical Observed Rain Frequency", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "calibration_curves.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot saved] Calibration curves saved to: {out_path}")
    return out_path


def plot_sample_timeseries(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir=DEFAULT_FIG_DIR, location="Metro Manila", days=60):
    """Plot sample timeline of predicted rain probability vs actual rain occurrence."""
    os.makedirs(output_dir, exist_ok=True)
    if location not in preprocessed_data:
        return None

    y_test = preprocessed_data[location]["y_test"][:days]
    dates = preprocessed_data[location]["dates_test"][:days]
    gnb_p = gnb_scores[location]["test_scores"][:days]
    knn_p = knn_scores[location]["test_scores"][:days]
    x = np.arange(len(y_test))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Highlight rainy days as light blue background bands
    for idx, actual in enumerate(y_test):
        if actual == 1:
            ax.axvspan(idx - 0.45, idx + 0.45, color="#70b5ff", alpha=0.35, lw=0, label="Observed Rain (>1mm)" if idx == 0 else "")

    ax.plot(x, gnb_p, color="#2b5c8f", lw=2, marker="s", ms=4, label="GNB P(Rain)")
    ax.plot(x, knn_p, color="#e65c00", lw=2, marker="o", ms=4, label="KNN P(Rain)")
    if lr_scores and location in lr_scores:
        lr_p = lr_scores[location]["test_scores"][:days]
        ax.plot(x, lr_p, color=LR_COLOR, lw=2, marker="^", ms=4, label="Logistic Regression P(Rain)")
    ax.axhline(0.5, color="red", linestyle=":", lw=1.2, label="0.5 Decision Boundary")

    # Format x-axis with date labels
    step = max(1, len(x) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(dates[i]) for i in x[::step]], rotation=25, ha="right", fontsize=9)

    ax.set_title(f"Rain Probability Forecast vs Actual Events: {location} (First {days} Test Days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Rain Probability", fontsize=10)
    ax.set_xlabel("Date (YYYYMMDD)", fontsize=10)
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle=":", alpha=0.5)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"timeseries_{location.lower().replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot saved] Sample timeseries saved to: {out_path}")
    return out_path


def generate_all_plots(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir=DEFAULT_FIG_DIR):
    """Generate the full visualization suite."""
    print("\n" + "=" * 65)
    print("Generating Matplotlib Visualizations...")
    print("=" * 65)
    plot_roc_curves(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir)
    plot_metrics_comparison(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir)
    plot_calibration_curves(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir)
    plot_sample_timeseries(preprocessed_data, gnb_scores, knn_scores, lr_scores, output_dir)
    print(f"All figures generated successfully in: {output_dir}")
