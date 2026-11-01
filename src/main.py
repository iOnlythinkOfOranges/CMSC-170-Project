from data.nasa_api import LOCATIONS, get_weather
from data.preprocess import clean_rows
from datetime import date

START_DATE = "20000101"
END_DATE = date.today().strftime("%Y%m%d")

def main():
    weather_by_location = {}

    for location, (latitude, longitude) in LOCATIONS.items():
        print(f"Getting weather data for {location}...")

        weather_by_location[location] = get_weather(
            latitude,
            longitude,
            START_DATE,
            END_DATE,
        )

    for location, rows in weather_by_location.items():
        print(f"{location}: {len(rows)} daily records")
        # for row in rows:
        #     print(row)

        # Next steps:
        # 1. Clean the rows and handle missing values
        # 2. Build weather windows and next-day rain labels
        # 3. Pass the prepared data to models


if __name__ == "__main__":
    main()