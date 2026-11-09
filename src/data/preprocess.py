# References:
# - WMO No. 8 (Rain Measurement): https://library.wmo.int/records/item/35616-guide-to-instruments-and-methods-of-observation
# - Directional Statistics (Mardia & Jupp): https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316979
# - Technical Documentation: docs/code_decisions_and_implementation.md
import math
from data.nasa_api import PARAMETERS

# Ref: NASA POWER data dictionary uses -999.0 as missing observation sentinel
MISSING_VALUE = -999.0

# Ref: 10 NASA variables expanded to 11 via trigonometric wind decomposition
DAILY_FEATURE_NAMES = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "PS",
    "WS10M",
    "WD10M_sin",
    "WD10M_cos",
    "T2MDEW",
    "ALLSKY_SFC_SW_DWN",
)


# Parse raw string rows into numeric values and map -999.0 to None
def clean_rows(rows):
    cleaned_rows = []

    for row in rows:
        cleaned_row = {}

        if "date" in row:
            cleaned_row["date"] = row["date"]

        if "YEAR" in row and row["YEAR"] is not None:
            cleaned_row["YEAR"] = int(row["YEAR"])
        elif "date" in row:
            cleaned_row["YEAR"] = int(str(row["date"])[:4])

        if "DOY" in row and row["DOY"] is not None:
            cleaned_row["DOY"] = int(row["DOY"])

        for parameter in PARAMETERS:
            val_raw = row.get(parameter)
            if val_raw is None or val_raw == "":
                cleaned_row[parameter] = None
                continue

            value = float(val_raw)
            if value == MISSING_VALUE:
                cleaned_row[parameter] = None
            else:
                cleaned_row[parameter] = value

        cleaned_rows.append(cleaned_row)

    return cleaned_rows


# Forward-fill and backward-fill missing None values
def impute_missing(rows):
    if not rows:
        return []

    imputed = [dict(r) for r in rows]

    for param in PARAMETERS:
        # Forward fill
        last_val = None
        for r in imputed:
            if r.get(param) is not None:
                last_val = r[param]
            elif last_val is not None:
                r[param] = last_val

        # Backward fill if leading entries are None
        first_val = None
        for r in imputed:
            if r.get(param) is not None:
                first_val = r[param]
                break

        if first_val is not None:
            for r in imputed:
                if r.get(param) is None:
                    r[param] = first_val
                else:
                    break

    return imputed


# Ref: Circular statistics (Mardia & Jupp): decompose wind degrees into sin/cos to remove 0/360 discontinuity
def extract_daily_features(row):
    wd_deg = row["WD10M"]
    wd_rad = math.radians(wd_deg)

    return [
        row["T2M"],
        row["T2M_MAX"],
        row["T2M_MIN"],
        row["PRECTOTCORR"],
        row["RH2M"],
        row["PS"],
        row["WS10M"],
        math.sin(wd_rad),
        math.cos(wd_rad),
        row["T2MDEW"],
        row["ALLSKY_SFC_SW_DWN"],
    ]


# Ref: Lookback W=5 days captures synoptic systems (Milestone 1); WMO/PAGASA > 1.0 mm separates measurable rain from trace
def create_sliding_windows(rows, window_size=5, rain_threshold=1.0):
    daily_vectors = [extract_daily_features(r) for r in rows]

    X = []
    y = []
    target_dates = []

    # Window covers past days; target is next day (i + 1)
    for i in range(window_size - 1, len(rows) - 1):
        sample = []
        for w in range(i - window_size + 1, i + 1):
            sample.extend(daily_vectors[w])
        X.append(sample)

        # Ref: WMO standard: 1 if tomorrow precipitation > 1.0 mm, else 0
        target_row = rows[i + 1]
        tomorrow_rain = 1 if target_row["PRECTOTCORR"] > rain_threshold else 0
        y.append(tomorrow_rain)
        target_dates.append(target_row.get("date"))

    return X, y, target_dates


# Ref: Chronological 70/15/15 split without shuffling prevents temporal data leakage
def train_val_test_split(X, y, target_dates, train_ratio=0.70, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = X[:train_end], y[:train_end], target_dates[:train_end]
    val = X[train_end:val_end], y[train_end:val_end], target_dates[train_end:val_end]
    test = X[val_end:], y[val_end:], target_dates[val_end:]

    return train, val, test


# Ref: Z-score standardization: fit mean and std strictly on train set to prevent test leakage
def standardize_features(X_train, X_val, X_test):
    n_train = len(X_train)
    num_features = len(X_train[0])

    # Compute mean for each feature on train set
    means = [
        sum(X_train[i][j] for i in range(n_train)) / n_train
        for j in range(num_features)
    ]

    # Compute std for each feature on train set
    stds = []
    for j in range(num_features):
        variance = sum((X_train[i][j] - means[j]) ** 2 for i in range(n_train)) / n_train
        std = math.sqrt(variance)
        stds.append(std if std > 1e-8 else 1.0)

    # Scale dataset using train statistics
    def scale(dataset):
        return [
            [(row[j] - means[j]) / stds[j] for j in range(num_features)]
            for row in dataset
        ]

    return scale(X_train), scale(X_val), scale(X_test), (means, stds)


# Complete end-to-end preprocessing pipeline for one location
def preprocess_location(rows, window_size=5, rain_threshold=1.0):
    cleaned = clean_rows(rows)
    imputed = impute_missing(cleaned)
    X, y, target_dates = create_sliding_windows(
        imputed, window_size=window_size, rain_threshold=rain_threshold
    )

    train, val, test = train_val_test_split(X, y, target_dates)
    X_train, y_train, dates_train = train
    X_val, y_val, dates_val = val
    X_test, y_test, dates_test = test
    X_train_s, X_val_s, X_test_s, (means, stds) = standardize_features(X_train, X_val, X_test)

    return {
        "X_train": X_train_s,
        "y_train": y_train,
        "dates_train": dates_train,
        "X_val": X_val_s,
        "y_val": y_val,
        "dates_val": dates_val,
        "X_test": X_test_s,
        "y_test": y_test,
        "dates_test": dates_test,
        "means": means,
        "stds": stds,
        "window_size": window_size,
        "feature_count": len(X_train_s[0]),
    }
