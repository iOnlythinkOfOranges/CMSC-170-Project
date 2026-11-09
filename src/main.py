from data.nasa_api import LOCATIONS, get_location_weather
from data.preprocess import preprocess_location
from comparison import run_compare
START_DATE = "20000101"
END_DATE = "20241231"

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

    # Model results and compare
    run_compare(preprocessed_data)


if __name__ == "__main__":
    main()
