import requests
import csv 
import io

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = (
    "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M",
    "PS", "WS10M", "WD10M", "T2MDEW", "ALLSKY_SFC_SW_DWN",
)

LOCATIONS = {
    "Manila": (14.5995, 120.9842),
    "Baguio": (16.4023, 120.5960),
    "Cebu City": (10.3157, 123.8854),
    "Davao City": (7.1907, 125.4553),
    "Zamboanga City": (6.9214, 122.0790),
}


# Return daily weather rows for one location
def get_weather(latitude, longitude, start_date, end_date):
    query_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start": start_date,
        "end": end_date,
        "community": "AG",
        "parameters": ",".join(PARAMETERS),
        "format": "CSV",
    }

    response = requests.get(URL, params=query_params, timeout=60)
    response.raise_for_status()

    csv_file = io.StringIO(response.text)

    # NASA places metadata before the actual CSV table
    for line in csv_file:
        if line.strip() == "-END HEADER-":
            break
    else:
        raise ValueError("NASA's CSV header was not found.")

    return list(csv.DictReader(csv_file))


# For testing
# Start with one location and two days to check that if it works
if __name__ == "__main__":
    latitude, longitude = LOCATIONS["Manila"]

    try:
        rows = get_weather(latitude, longitude, "20260101", "20260102")

        for row in rows:
            print(row)

    except requests.RequestException as error:
        print(f"NASA POWER request failed: {error}")
        if error.response is not None:
            print(error.response.text)