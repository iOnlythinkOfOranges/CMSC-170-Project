# References:
# - WMO No. 8 (Rain Measurement): https://library.wmo.int/records/item/35616-guide-to-instruments-and-methods-of-observation
# - Directional Statistics (Mardia & Jupp): https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316979
# - Wilks (2011) Statistical Methods in Atmospheric Sciences
# - Technical Documentation: docs/exploratory_data_analysis.md

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import math
import argparse
import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from data.nasa_api import get_location_weather, LOCATIONS, PARAMETERS
from data.preprocess import clean_rows, preprocess_location

DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)

# Core 6 benchmark locations representing Coronas Climate Types I - IV
CORE_LOCATIONS = [
    "Metro Manila",    # Coronas Type I (Lowland)
    "Baguio",          # Coronas Type I (Highland, ~1500m)
    "Legazpi City",    # Coronas Type II (No dry season, winter peak)
    "Cebu City",       # Coronas Type III (Intermediate)
    "Davao City",      # Coronas Type IV (Uniform rainfall)
    "Zamboanga City",  # Coronas Type III (Southwestern Mindanao)
]

# Physical metadata and units for raw parameters
PARAMETER_METADATA = {
    "T2M": {"name": "Air Temperature", "unit": "°C", "short": "T2M"},
    "T2M_MAX": {"name": "Max Temperature", "unit": "°C", "short": "T2M_MAX"},
    "T2M_MIN": {"name": "Min Temperature", "unit": "°C", "short": "T2M_MIN"},
    "PRECTOTCORR": {"name": "Precipitation", "unit": "mm/day", "short": "Rain"},
    "RH2M": {"name": "Relative Humidity", "unit": "%", "short": "RH2M"},
    "PS": {"name": "Surface Pressure", "unit": "kPa", "short": "PS"},
    "WS10M": {"name": "Wind Speed", "unit": "m/s", "short": "WS10M"},
    "WD10M": {"name": "Wind Direction", "unit": "degrees", "short": "WD10M"},
    "T2MDEW": {"name": "Dew Point", "unit": "°C", "short": "T2MDEW"},
    "ALLSKY_SFC_SW_DWN": {"name": "Solar Insolation", "unit": "MJ/m²/day", "short": "Insolation"},
}

CORONAS_TYPES = {
    "Metro Manila": "Type I (Dry/Wet)",
    "Baguio": "Type I (Highland)",
    "Legazpi City": "Type II (No Dry, Winter Peak)",
    "Cebu City": "Type III (Intermediate)",
    "Davao City": "Type IV (Uniform)",
    "Zamboanga City": "Type III (Intermediate)",
}

COLOR_PALETTE = {
    "Metro Manila": "#2b5c8f",
    "Baguio": "#e65c00",
    "Legazpi City": "#238b57",
    "Cebu City": "#7b2cbf",
    "Davao City": "#c0392b",
    "Zamboanga City": "#16a085",
}


# =====================================================================
# 1. Mathematical & Statistical Utilities (from scratch)
# =====================================================================

def compute_fisher_pearson_skewness(values):
    """Compute the adjusted Fisher-Pearson standardized skewness coefficient g1."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 3:
        return 0.0

    mean_val = np.mean(arr)
    diff = arr - mean_val
    m2 = np.mean(diff ** 2)
    m3 = np.mean(diff ** 3)

    if m2 < 1e-12:
        return 0.0

    g1 = m3 / (m2 ** 1.5)
    return float(g1)


def compute_raw_descriptive_stats(rows, parameters=PARAMETERS):
    """Compute empirical distribution statistics for raw meteorological features."""
    cleaned = clean_rows(rows)
    stats = {}

    for param in parameters:
        vals = [r[param] for r in cleaned if r.get(param) is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_v = float(np.min(arr))
        q25_v = float(np.percentile(arr, 25))
        med_v = float(np.median(arr))
        q75_v = float(np.percentile(arr, 75))
        max_v = float(np.max(arr))
        skew_v = compute_fisher_pearson_skewness(arr)

        stats[param] = {
            "count": len(arr),
            "mean": mean_v,
            "std": std_v,
            "min": min_v,
            "q25": q25_v,
            "median": med_v,
            "q75": q75_v,
            "max": max_v,
            "skewness": skew_v,
        }
    return stats


def compute_correlation_matrix(matrix):
    """Compute pairwise Pearson correlation matrix from scratch or via np.corrcoef."""
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}")
    return np.corrcoef(X, rowvar=False)


def compute_autocorrelation(series, max_lag=5):
    """Compute empirical autocorrelation function rho(k) across lags 0 to max_lag."""
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n <= max_lag:
        return np.ones(max_lag + 1)

    s_centered = s - np.mean(s)
    var = np.sum(s_centered ** 2)
    if var < 1e-12:
        return np.ones(max_lag + 1)

    res = [1.0]
    for k in range(1, max_lag + 1):
        c_k = np.sum(s_centered[:n - k] * s_centered[k:]) / var
        res.append(float(c_k))
    return np.array(res)


def compute_monthly_seasonality(rows):
    """Aggregate daily precipitation by calendar month to analyze climate seasonality."""
    monthly_precip = {m: [] for m in range(1, 13)}
    monthly_rain_days = {m: 0 for m in range(1, 13)}

    for row in rows:
        date_str = str(row.get("date", ""))
        p_val = row.get("PRECTOTCORR")
        if p_val is None or len(date_str) < 6:
            continue
        try:
            m = int(date_str[4:6])
            val = float(p_val)
        except (ValueError, TypeError):
            continue

        monthly_precip[m].append(val)
        if val > 1.0:
            monthly_rain_days[m] += 1

    monthly_stats = {}
    for m in range(1, 13):
        vals = monthly_precip[m]
        count = len(vals)
        monthly_stats[m] = {
            "mean_precip": float(np.mean(vals)) if count > 0 else 0.0,
            "rain_frequency": (monthly_rain_days[m] / count * 100.0) if count > 0 else 0.0,
            "total_days": count,
        }
    return monthly_stats


def compute_class_imbalance(rows, rain_threshold=1.0):
    """Compute binary rainfall target class balance from raw observations."""
    rain_days = 0
    dry_days = 0
    for r in rows:
        p = r.get("PRECTOTCORR")
        if p is None:
            continue
        try:
            val = float(p)
        except (ValueError, TypeError):
            continue
        if val > rain_threshold:
            rain_days += 1
        else:
            dry_days += 1

    total = rain_days + dry_days
    rain_pct = (rain_days / total * 100.0) if total > 0 else 0.0
    dry_pct = (dry_days / total * 100.0) if total > 0 else 0.0
    return {
        "total_days": total,
        "rain_days": rain_days,
        "dry_days": dry_days,
        "rain_pct": rain_pct,
        "dry_pct": dry_pct,
    }


# =====================================================================
# 2. Visualization Functions (4 Target Figures)
# =====================================================================

def plot_raw_feature_distributions(manila_rows, baguio_rows, output_path):
    """Figure 1: Side-by-side boxplots showing units/ranges and elevation contrast."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cleaned_m = clean_rows(manila_rows)
    cleaned_b = clean_rows(baguio_rows)

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.suptitle(
        "Initial Raw Meteorological Feature Distributions: Lowland (Metro Manila) vs. Highland (Baguio)\n"
        "25 Years of Daily Observations (2000–2024, N=9,132 Days per Station)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    axes_flat = axes.flatten()

    for idx, param in enumerate(PARAMETERS):
        ax = axes_flat[idx]
        vals_m = [r[param] for r in cleaned_m if r.get(param) is not None]
        vals_b = [r[param] for r in cleaned_b if r.get(param) is not None]

        meta = PARAMETER_METADATA.get(param, {"name": param, "unit": ""})

        bp = ax.boxplot(
            [vals_m, vals_b],
            patch_artist=True,
            widths=0.55,
            tick_labels=["Manila", "Baguio"],
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgecolor="none"),
        )

        bp["boxes"][0].set_facecolor("#2b5c8f")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#e65c00")
        bp["boxes"][1].set_alpha(0.7)

        title_str = f"{param} ({meta['unit']})\n{meta['name']}"
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.tick_params(labelsize=10)

        # Highlight key physical insights
        if param == "PS":
            diff = np.mean(vals_m) - np.mean(vals_b)
            ax.annotate(
                f"Lapse Drop:\n~{diff:.1f} kPa",
                xy=(1.5, min(np.mean(vals_m), np.mean(vals_b))),
                xytext=(1.1, 95.0),
                fontsize=9,
                fontweight="bold",
                color="#b22222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe4e1", alpha=0.8),
            )
        elif param == "PRECTOTCORR":
            max_val = max(np.max(vals_m), np.max(vals_b))
            ax.annotate(
                f"Peak: {max_val:.0f} mm\nSkew: +5.0 to +6.0",
                xy=(2, max_val),
                xytext=(1.2, max_val * 0.75),
                fontsize=8.5,
                fontweight="bold",
                color="#005580",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e6f2ff", alpha=0.8),
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_climate_seasonality_and_imbalance(seasonality_data, imbalance_data, output_path):
    """Figure 2: 12-month precipitation curves across Coronas Types and Class Imbalance."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Left Panel: Monthly Rainfall Trajectories by Climate Type
    for loc, s_dict in seasonality_data.items():
        color = COLOR_PALETTE.get(loc, "#333333")
        ctype = CORONAS_TYPES.get(loc, "")
        means = [s_dict[m]["mean_precip"] for m in months]

        linestyle = "-"
        if "Highland" in ctype:
            linestyle = "--"
        elif "Type II" in ctype:
            linestyle = "-."
        elif "Type IV" in ctype:
            linestyle = ":"

        ax1.plot(
            months,
            means,
            label=f"{loc} ({ctype})",
            color=color,
            linewidth=2.4,
            linestyle=linestyle,
            marker="o",
            markersize=5,
        )

    ax1.set_title("Climatological Seasonality Across Philippine Coronas Types\n(25-Year Daily Means, 2000–2024)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Calendar Month", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Daily Precipitation (mm/day)", fontsize=11, fontweight="bold")
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    # Highlight Habagat vs Amihan zones
    ax1.axvspan(6.5, 9.5, color="#ffeb3b", alpha=0.18, label="Habagat (SW Monsoon)")
    ax1.axvspan(10.5, 12.5, color="#03a9f4", alpha=0.15, label="Amihan Peak (Type II)")

    # Right Panel: Class Imbalance Stacked Horizontal Bars
    loc_names = list(imbalance_data.keys())
    # Sort by rain percentage descending
    loc_names.sort(key=lambda x: imbalance_data[x]["rain_pct"], reverse=True)

    y_pos = np.arange(len(loc_names))
    rain_pcts = [imbalance_data[loc]["rain_pct"] for loc in loc_names]
    dry_pcts = [imbalance_data[loc]["dry_pct"] for loc in loc_names]

    ax2.barh(y_pos, rain_pcts, color="#2b5c8f", alpha=0.85, label="Rainy Day (> 1.0 mm)")
    ax2.barh(y_pos, dry_pcts, left=rain_pcts, color="#d9534f", alpha=0.65, label="Dry Day (<= 1.0 mm)")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(loc_names, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Percentage of 25-Year Observations (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Target Class Imbalance by Synoptic Station\n(Binary Threshold: PRECTOTCORR > 1.0 mm/day)", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="x")

    for i, loc in enumerate(loc_names):
        r_pct = rain_pcts[i]
        d_pct = dry_pcts[i]
        r_cnt = imbalance_data[loc]["rain_days"]
        d_cnt = imbalance_data[loc]["dry_days"]
        ax2.text(r_pct / 2, i, f"{r_pct:.1f}%\n({r_cnt:,}d)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax2.text(r_pct + (d_pct / 2), i, f"{d_pct:.1f}%\n({d_cnt:,}d)", ha="center", va="center", color="black", fontweight="bold", fontsize=9)

    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_correlations(corr_matrix, feature_names, output_path, title="Pearson Correlation Matrix of Raw Meteorological Features"):
    """Figure 3: Pearson correlation heatmap explaining GNB saturation and physical coupling."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9.5))

    cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient (r)", fontsize=11, fontweight="bold")

    n_features = len(feature_names)
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(feature_names, fontsize=10, fontweight="bold")

    # Add numeric correlation values inside cells
    for i in range(n_features):
        for j in range(n_features):
            val = corr_matrix[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")

    ax.set_title(
        f"{title}\n"
        "(Physical Multicollinearity Explains Naive Bayes Likelihood Saturation)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_preprocessing_transformation(raw_rows, preprocessed_dict, output_path):
    """Figure 4: 4-quadrant before/after preprocessing transformation diagnostics."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cleaned = clean_rows(raw_rows)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(
        "Before vs. After: Meteorological Data Preprocessing Transformations\n"
        "From Raw Physical Observations to Normalized Sliding-Window Representations",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # -------------------------------------------------------------
    # Quadrant 1 (Top-Left): Scale Equalization
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    raw_sample_params = ["PS", "RH2M", "WD10M", "T2M", "WS10M", "PRECTOTCORR"]
    raw_matrix = []
    for p in raw_sample_params:
        vals = [r[p] for r in cleaned if r.get(p) is not None]
        raw_matrix.append(vals)

    # Boxplot of raw disparate scales
    bp1 = ax1.boxplot(
        raw_matrix,
        patch_artist=True,
        tick_labels=raw_sample_params,
        widths=0.45,
        flierprops=dict(markersize=2, alpha=0.3),
    )
    for b in bp1["boxes"]:
        b.set_facecolor("#d9534f")
        b.set_alpha(0.65)

    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_title("1. Scale Equalization: Raw Scales (SymLog) vs. Standardized Z-Scores", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Raw Physical Scale (SymLog)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Inset or twin axis showing Z-scores after standardization
    ax1_twin = ax1.twinx()
    z_scores_sample = [preprocessed_dict["X_train"][:, idx * 5] for idx in range(len(raw_sample_params))]
    bp1_z = ax1_twin.boxplot(
        z_scores_sample,
        patch_artist=True,
        positions=np.arange(1, len(raw_sample_params) + 1) + 0.25,
        widths=0.3,
        showfliers=False,
    )
    for b in bp1_z["boxes"]:
        b.set_facecolor("#2b5c8f")
        b.set_alpha(0.7)
    ax1_twin.set_ylabel("Standardized Z-Score (Mean=0, Std=1)", fontsize=10, fontweight="bold", color="#2b5c8f")
    ax1_twin.set_ylim(-4, 4)
    ax1.set_xticks(np.arange(1, len(raw_sample_params) + 1) + 0.12)
    ax1.set_xticklabels(raw_sample_params, fontweight="bold")

    # -------------------------------------------------------------
    # Quadrant 2 (Top-Right): Wind Direction Vectorization
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    angles_deg = np.linspace(0, 360, 361)
    angles_rad = np.radians(angles_deg)
    sin_vals = np.sin(angles_rad)
    cos_vals = np.cos(angles_rad)

    # Plot circular unit circle
    ax2.plot(cos_vals, sin_vals, color="#7b2cbf", lw=2, linestyle="--", label="Unit Circle (Continuous)")
    ax2.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax2.axvline(0, color="gray", lw=0.8, linestyle=":")

    # Highlight near-north discontinuity in raw vs continuous in vector
    d1 = 1.0
    d359 = 359.0
    r1, r359 = np.radians(d1), np.radians(d359)
    x1, y1 = np.cos(r1), np.sin(r1)
    x359, y359 = np.cos(r359), np.sin(r359)

    ax2.scatter([x1, x359], [y1, y359], color=["#238b57", "#c0392b"], s=80, zorder=5)
    ax2.annotate(f"1° (x={x1:.3f}, y={y1:.3f})", xy=(x1, y1), xytext=(x1 - 0.45, y1 + 0.15),
                 fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#238b57"))
    ax2.annotate(f"359° (x={x359:.3f}, y={y359:.3f})\nEuclidean Δ = 0.035", xy=(x359, y359), xytext=(x359 - 0.55, y359 - 0.25),
                 fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#c0392b"))

    ax2.set_title("2. Wind Direction: Discontinuous Degrees [0°, 360°] -> [cos(θ), sin(θ)]", fontsize=12, fontweight="bold")
    ax2.set_xlabel("cos(Wind Direction)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("sin(Wind Direction)", fontsize=10, fontweight="bold")
    ax2.set_xlim(-1.25, 1.25)
    ax2.set_ylim(-1.25, 1.25)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9.5)

    # -------------------------------------------------------------
    # Quadrant 3 (Bottom-Left): Multi-Day Synoptic Memory (Autocorrelation)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    lags = np.arange(0, 6)
    track_params = ["PS", "T2M", "RH2M", "PRECTOTCORR"]
    param_colors = {"PS": "#238b57", "T2M": "#d9534f", "RH2M": "#2b5c8f", "PRECTOTCORR": "#7b2cbf"}

    for p in track_params:
        vals = [r[p] for r in cleaned if r.get(p) is not None]
        ac = compute_autocorrelation(vals, max_lag=5)
        ax3.plot(lags, ac, marker="o", lw=2.2, label=f"{p} ({PARAMETER_METADATA[p]['name']})", color=param_colors[p])

    ax3.axhline(0, color="black", lw=0.8, linestyle="--")
    ax3.set_title("3. Multi-Day Atmospheric Autocorrelation (Synoptic Memory vs. Episodic Rain)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Lag k (Days Lookback)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Autocorrelation Coefficient ρ(k)", fontsize=10, fontweight="bold")
    ax3.set_xticks(lags)
    ax3.set_ylim(-0.1, 1.05)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=9.5)

    # -------------------------------------------------------------
    # Quadrant 4 (Bottom-Right): Leak-Proof Chronological Partitions
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    splits = ["Train (2000–2017)", "Val (2018–2020)", "Test (2021–2024)"]
    counts = [
        len(preprocessed_dict["X_train"]),
        len(preprocessed_dict["X_val"]),
        len(preprocessed_dict["X_test"]),
    ]
    years = [18, 3, 4]
    bar_colors = ["#2b5c8f", "#e65c00", "#238b57"]

    bars = ax4.bar(splits, counts, color=bar_colors, alpha=0.8, width=0.55)
    ax4.set_title("4. Leakage-Proof Partitions with 5-Day Boundary Purges", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Number of Processed Daily Samples", fontsize=10, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax4.set_ylim(0, 7500)

    for bar, yr, cnt in zip(bars, years, counts):
        height = bar.get_height()
        pct = (cnt / sum(counts)) * 100
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 150,
            f"{cnt:,} samples\n({pct:.1f}% | {yr} yrs)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    ax4.annotate(
        "Purged: 5 days at Val start\nPurged: 5 days at Test start\nZero Lookback Leakage",
        xy=(1, counts[1]),
        xytext=(0.7, 4500),
        fontsize=9,
        fontweight="bold",
        color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe4e1", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="#8b0000"),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# =====================================================================
# 3. Execution Pipeline & CLI Integration
# =====================================================================

def generate_all_eda_plots(output_dir=DEFAULT_OUTPUT_DIR, locations=None):
    """Generate all 4 EDA figures specified in docs/exploratory_data_analysis.md."""
    os.makedirs(output_dir, exist_ok=True)
    target_locs = locations if locations else CORE_LOCATIONS

    print(f"\n[EDA] Loading raw and preprocessed data for {len(target_locs)} stations...")
    raw_data_by_loc = {}
    preprocessed_by_loc = {}
    seasonality_by_loc = {}
    imbalance_by_loc = {}

    for loc in target_locs:
        coords = LOCATIONS.get(loc)
        if not coords:
            continue
        lat, lon = coords
        rows = get_location_weather(loc, lat, lon, "20000101", "20241231")
        raw_data_by_loc[loc] = rows

        # Compute seasonality and class imbalance
        seasonality_by_loc[loc] = compute_monthly_seasonality(rows)
        imbalance_by_loc[loc] = compute_class_imbalance(rows, rain_threshold=1.0)

        # Preprocess with 5-day sliding window
        p_data = preprocess_location(rows, window_size=5, rain_threshold=1.0)
        # Ensure numpy arrays for vectorized plotting
        p_data["X_train"] = np.array(p_data["X_train"])
        p_data["X_val"] = np.array(p_data["X_val"])
        p_data["X_test"] = np.array(p_data["X_test"])
        preprocessed_by_loc[loc] = p_data

    generated_figures = {}

    # 1. Raw Feature Distributions (Metro Manila vs. Baguio)
    fig1_path = os.path.join(output_dir, "eda_raw_feature_distributions.png")
    print(f"[EDA] Generating Figure 1: {fig1_path}...")
    m_rows = raw_data_by_loc.get("Metro Manila", raw_data_by_loc[target_locs[0]])
    b_rows = raw_data_by_loc.get("Baguio", raw_data_by_loc[target_locs[-1]])
    plot_raw_feature_distributions(m_rows, b_rows, fig1_path)
    generated_figures["raw_feature_distributions"] = fig1_path

    # 2. Climate Seasonality & Class Imbalance
    fig2_path = os.path.join(output_dir, "eda_climate_seasonality_and_imbalance.png")
    print(f"[EDA] Generating Figure 2: {fig2_path}...")
    plot_climate_seasonality_and_imbalance(seasonality_by_loc, imbalance_by_loc, fig2_path)
    generated_figures["climate_seasonality_and_imbalance"] = fig2_path

    # 3. Feature Correlations
    fig3_path = os.path.join(output_dir, "eda_feature_correlations.png")
    print(f"[EDA] Generating Figure 3: {fig3_path}...")
    cleaned_m = clean_rows(m_rows)
    raw_feature_matrix = np.array([[r[p] for p in PARAMETERS] for r in cleaned_m])
    corr_mat = compute_correlation_matrix(raw_feature_matrix)
    plot_feature_correlations(corr_mat, list(PARAMETERS), fig3_path)
    generated_figures["feature_correlations"] = fig3_path

    # 4. Preprocessing Transformations
    fig4_path = os.path.join(output_dir, "eda_preprocessing_transformation.png")
    print(f"[EDA] Generating Figure 4: {fig4_path}...")
    m_prep = preprocessed_by_loc.get("Metro Manila", preprocessed_by_loc[target_locs[0]])
    plot_preprocessing_transformation(m_rows, m_prep, fig4_path)
    generated_figures["preprocessing_transformation"] = fig4_path

    print("[EDA] All 4 publication-quality figures successfully generated!")
    return generated_figures


def run_eda(output_dir=DEFAULT_OUTPUT_DIR, locations=None, print_tables=True):
    """Run full EDA pipeline, print descriptive summary tables, and generate plots."""
    target_locs = locations if locations else CORE_LOCATIONS
    print("=" * 75)
    print("CMSC 170 — Exploratory Data Analysis (EDA): Raw vs. Preprocessed Data")
    print("=" * 75)

    if print_tables:
        print("\n[Part I: Raw Meteorological Feature Summary]")
        for loc in ["Metro Manila", "Baguio"]:
            if loc not in LOCATIONS:
                continue
            lat, lon = LOCATIONS[loc]
            rows = get_location_weather(loc, lat, lon, "20000101", "20241231")
            stats = compute_raw_descriptive_stats(rows)
            print(f"\n--- {loc} (2000–2024, N={len(rows)} days) ---")
            print(f"{'Parameter':18s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Median':>8s} {'Max':>8s} {'Skewness':>9s}")
            print("-" * 75)
            for param in PARAMETERS:
                s = stats.get(param, {})
                print(
                    f"{param:18s} {s.get('mean', 0.0):8.2f} {s.get('std', 0.0):8.2f} "
                    f"{s.get('min', 0.0):8.2f} {s.get('median', 0.0):8.2f} {s.get('max', 0.0):8.2f} "
                    f"{s.get('skewness', 0.0):+9.2f}"
                )

        print("\n[Part II: Target Class Imbalance (Rain > 1.0 mm/day)]")
        print(f"{'Location':18s} {'Coronas Type':22s} {'Rain Days':>10s} {'Dry Days':>10s} {'Rain %':>8s}")
        print("-" * 75)
        for loc in target_locs:
            lat, lon = LOCATIONS[loc]
            rows = get_location_weather(loc, lat, lon, "20000101", "20241231")
            imb = compute_class_imbalance(rows)
            ctype = CORONAS_TYPES.get(loc, "Unknown")
            print(
                f"{loc:18s} {ctype:22s} {imb['rain_days']:10,d} {imb['dry_days']:10,d} {imb['rain_pct']:7.1f}%"
            )

    return generate_all_eda_plots(output_dir=output_dir, locations=target_locs)


def main():
    parser = argparse.ArgumentParser(description="CMSC 170 — Exploratory Data Analysis (EDA)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated figures (default: figures/)",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip printing descriptive statistic tables to stdout",
    )
    args = parser.parse_args()
    run_eda(output_dir=args.output_dir, print_tables=not args.no_tables)


if __name__ == "__main__":
    main()
