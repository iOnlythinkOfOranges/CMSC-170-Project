from data.nasa_api import LOCATIONS, get_location_weather
from data.preprocess import clean_rows

START_DATE = "20000101"
END_DATE = "20241231"


def main():
    weather_by_location = {}

    print("\nNASA POWER Weather Data Ingestion (CMSC 170 - Group BAWWL)")
    print("=" * 65)

    # Ingest data for each location
    for location, (latitude, longitude) in LOCATIONS.items():
        print(f"\nProcessing {location} ({latitude}, {longitude})...")
        try:
            weather_by_location[location] = get_location_weather(
                location_name=location,
                latitude=latitude,
                longitude=longitude,
                start_date=START_DATE,
                end_date=END_DATE,
            )
        except Exception as error:
            print(f"Failed to ingest {location}: {error}")

    # Print summary of ingested data
    print("\n" + "=" * 65)
    print("Ingestion Summary")
    print("=" * 65)
    for location, rows in weather_by_location.items():
        cleaned = clean_rows(rows[:3])
        sample_date = cleaned[0].get("date", "N/A") if cleaned else "N/A"
        print(f"- {location:16s}: {len(rows):5d} daily records (starts {sample_date})")



if __name__ == "__main__":
    main()
