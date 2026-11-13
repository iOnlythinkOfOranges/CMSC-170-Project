import argparse
import sys
from data.nasa_api import LOCATIONS, get_location_weather
from data.preprocess import preprocess_location
from comparison import run_compare
from window_comparison import run_window_size_experiments
from eda import run_eda

START_DATE = "20000101"
END_DATE = "20241231"


def main():
    parser = argparse.ArgumentParser(description="CMSC 170 - Weather Pipeline")
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Sliding window lookback size in days (e.g. 3, 5, 7; default: 5)",
    )
    parser.add_argument(
        "--compare-windows",
        action="store_true",
        help="Run comprehensive benchmark comparing window sizes 3, 5, and 7 days",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="3,5,7",
        help="Comma-separated window sizes for --compare-windows (default: '3,5,7')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run on core 6 synoptic stations only for fast execution",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run Exploratory Data Analysis (EDA) on raw vs. preprocessed data and generate diagnostic plots",
    )
    args = parser.parse_args()

    if args.eda:
        loc_subset = [
            "Metro Manila", "Baguio", "Legazpi City", "Cebu City", "Davao City", "Zamboanga City"
        ] if args.quick else None
        run_eda(locations=loc_subset)
        return

    if args.compare_windows:
        w_sizes = [int(x.strip()) for x in args.window_sizes.split(",") if x.strip()]
        loc_subset = [
            "Metro Manila", "Baguio", "Legazpi City", "Cebu City", "Davao City", "Zamboanga City"
        ] if args.quick else None
        run_window_size_experiments(window_sizes=w_sizes, locations=loc_subset)
        return

    print("=" * 65)
    print("CMSC 170 - Group BAWWL: Weather Pipeline")
    print(f"Sliding Window Size: {args.window_size} Days ({args.window_size * 11} features per sample)")
    print("=" * 65)

    # Ingest and preprocess data for each location
    preprocessed_data = {}

    target_locs = {
        k: LOCATIONS[k]
        for k in ["Metro Manila", "Baguio", "Legazpi City", "Cebu City", "Davao City", "Zamboanga City"]
    } if args.quick else LOCATIONS

    for location, (latitude, longitude) in target_locs.items():
        print(f"\nProcessing {location}...")
        try:
            rows = get_location_weather(
                location_name=location,
                latitude=latitude,
                longitude=longitude,
                start_date=START_DATE,
                end_date=END_DATE,
            )
            # Lookback window with 1.0 mm rain threshold
            data = preprocess_location(rows, window_size=args.window_size, rain_threshold=1.0)
            preprocessed_data[location] = data
            print(f"  Train: {len(data['X_train'])} samples | Val: {len(data['X_val'])} | Test: {len(data['X_test'])}")
            print(f"  Features per sample: {data['feature_count']} ({args.window_size} days x 11 daily features)")
        except Exception as error:
            print(f"[Warning] Failed to process {location}: {error}")

    # Pipeline summary
    print("\n" + "=" * 65)
    print("Preprocessing Summary (Ready for Model Training)")
    print("=" * 65)
    for location, data in preprocessed_data.items():
        rain_rate = sum(data["y_train"]) / len(data["y_train"]) * 100
        print(f"- {location:16s}: {len(data['X_train']):4d} train | {len(data['X_test']):4d} test | Train Rain: {rain_rate:5.1f}%")

    # Model results and compare
    run_compare(preprocessed_data)


if __name__ == "__main__":
    main()
