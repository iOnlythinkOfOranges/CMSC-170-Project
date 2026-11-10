# References:
# - NASA POWER API: https://power.larc.nasa.gov/
# - PAGASA Synoptic Stations: https://bagong.pagasa.dost.gov.ph/climate/climate-monitoring

import os
import csv
import io
import requests
from datetime import datetime

# Ref: NASA POWER Agroclimatology (AG) Daily API (NASA Langley Research Center)
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Ref: NASA POWER meteorology parameters (MERRA-2 / GMAO assimilation)
PARAMETERS = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "PS",
    "WS10M",
    "WD10M",
    "T2MDEW",
    "ALLSKY_SFC_SW_DWN",
)

# Coordinates reference PAGASA Synoptic Weather Stations across Coronas Climate Types I-IV
LOCATIONS = {
    # Ref: PAGASA Science Garden Synoptic Station, Quezon City (Coronas Type I)
    "Metro Manila": (14.6400, 121.0400),
    # Ref: PAGASA Mactan / Cebu Complex Synoptic Station (Coronas Type III)
    "Cebu City": (10.3157, 123.8854),
    # Ref: PAGASA Davao Synoptic Station, F. Bangoy Airport (Coronas Type IV)
    "Davao City": (7.1907, 125.4553),
    # Ref: PAGASA Legazpi Synoptic Station, Albay (Coronas Type II)
    "Legazpi City": (13.1391, 123.7438),
    # Ref: PAGASA Baguio Synoptic Station, Benguet (Coronas Type I Highland)
    "Baguio": (16.4023, 120.5960),
    # Ref: PAGASA Zamboanga Synoptic Station, Airport (Coronas Type III)
    "Zamboanga City": (6.9214, 122.0790),
}

DEFAULT_RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
)

LOCATION_ALIASES = {
    "metro manila": ["metro_manila.csv", "manila.csv"],
    "manila": ["manila.csv", "metro_manila.csv"],
    "cebu city": ["cebu_city.csv", "cebu.csv"],
    "cebu": ["cebu.csv", "cebu_city.csv"],
    "davao city": ["davao_city.csv", "davao.csv"],
    "davao": ["davao.csv", "davao_city.csv"],
    "legazpi city": ["legazpi_city.csv", "legazpi.csv"],
    "legazpi": ["legazpi.csv", "legazpi_city.csv"],
    "baguio": ["baguio.csv"],
    "zamboanga city": ["zamboanga_city.csv", "zamboanga.csv"],
    "zamboanga": ["zamboanga.csv", "zamboanga_city.csv"],
}


# Make sure date, YEAR, and DOY are all present
def _normalize_row(row):
    normalized = dict(row)

    has_date = bool(normalized.get("date"))
    has_year = bool(normalized.get("YEAR"))
    has_doy = bool(normalized.get("DOY"))

    if has_date and has_year and has_doy:
        return normalized

    if has_date:
        date_str = str(normalized["date"]).strip()
        if len(date_str) == 8 and date_str.isdigit():
            try:
                dt = datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                normalized["YEAR"] = str(dt.year)
                normalized["DOY"] = str(dt.timetuple().tm_yday)
            except ValueError:
                pass
        else:
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                normalized["YEAR"] = str(dt.year)
                normalized["DOY"] = str(dt.timetuple().tm_yday)
            except ValueError:
                pass
    elif has_year and has_doy:
        try:
            year = int(normalized["YEAR"])
            doy = int(normalized["DOY"])
            dt = datetime.strptime(f"{year} {doy:03d}", "%Y %j")
            normalized["date"] = dt.strftime("%Y%m%d")
        except ValueError:
            pass

    return normalized


# Save rows to a CSV file
def save_weather_to_csv(rows, file_path):
    if not rows:
        raise ValueError("Cannot save empty weather rows.")

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    fieldnames = ["date", "YEAR", "DOY"] + [p for p in PARAMETERS if p in rows[0]]

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))


# Filter rows by inclusive date range (YYYYMMDD)
def _filter_by_date(rows, start_date=None, end_date=None):
    if not start_date and not end_date:
        return rows
    filtered = []
    for row in rows:
        d = str(row.get("date", ""))
        if start_date and d < str(start_date):
            continue
        if end_date and d > str(end_date):
            continue
        filtered.append(row)
    return filtered


# Load rows from local CSV
def load_weather_from_csv(file_path, start_date=None, end_date=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Weather CSV file not found: {file_path}")

    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [_normalize_row(row) for row in reader]

    if start_date or end_date:
        rows = _filter_by_date(rows, start_date, end_date)

    return rows


# Check if local CSV already exists for this location
def find_cached_file(location_name, cache_dir=DEFAULT_RAW_DIR):
    clean_name = location_name.strip().lower()
    candidates = LOCATION_ALIASES.get(
        clean_name,
        [f"{clean_name.replace(' ', '_')}.csv"],
    )

    for filename in candidates:
        full_path = os.path.join(cache_dir, filename)
        if os.path.isfile(full_path):
            return full_path

    return None


# Match lat/lon to a known location name
def _find_location_name_by_coords(lat, lon):
    for name, coords in LOCATIONS.items():
        if abs(coords[0] - lat) < 0.1 and abs(coords[1] - lon) < 0.1:
            return name
    return None


# Fetch from NASA POWER API, with cache fallback
def get_weather(
    latitude,
    longitude,
    start_date="20000101",
    end_date="20241231",
    community="AG",
    timeout=60,
    cache_dir=DEFAULT_RAW_DIR,
    force_download=False,
    location_name=None,
):
    loc_name = location_name or _find_location_name_by_coords(latitude, longitude)
    if loc_name and not force_download:
        cached_path = find_cached_file(loc_name, cache_dir)
        if cached_path:
            return load_weather_from_csv(cached_path)

    query_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start": start_date,
        "end": end_date,
        "community": community,
        "parameters": ",".join(PARAMETERS),
        "format": "CSV",
    }

    headers = {"User-Agent": "CMSC170-GroupBAWWL/1.0"}
    try:
        response = requests.get(URL, params=query_params, headers=headers, timeout=timeout)
        response.raise_for_status()
    except Exception as err:
        # Fall back to local cache if offline or request fails
        if loc_name:
            cached_path = find_cached_file(loc_name, cache_dir)
            if cached_path:
                print(f"Network failed ({err}). Loaded cached data from {cached_path}")
                return load_weather_from_csv(cached_path)
        raise

    csv_file = io.StringIO(response.text)

    # Ref: NASA POWER CSV specification: tabular records begin after -END HEADER-
    for line in csv_file:
        if line.strip() == "-END HEADER-":
            break
    else:
        raise ValueError("NASA CSV header boundary '-END HEADER-' was not found.")

    reader = csv.DictReader(csv_file)
    rows = [_normalize_row(row) for row in reader]

    if not rows:
        raise ValueError(f"No records returned for coords ({latitude}, {longitude}).")

    # Cache fetched records
    if loc_name:
        os.makedirs(cache_dir, exist_ok=True)
        slug = loc_name.strip().lower().replace(" ", "_")
        target_path = os.path.join(cache_dir, f"{slug}.csv")
        save_weather_to_csv(rows, target_path)
        print(f"[Cache saved] Saved {len(rows)} records to {target_path}")

    return rows


# Get weather for a location: loads from cache if present, else downloads
def get_location_weather(
    location_name,
    latitude=None,
    longitude=None,
    start_date="20000101",
    end_date="20241231",
    cache_dir=DEFAULT_RAW_DIR,
    force_download=False,
):
    cached_path = find_cached_file(location_name, cache_dir)

    if cached_path and not force_download:
        rows = load_weather_from_csv(cached_path)
        print(f"Loaded {len(rows)} daily records for '{location_name}' from {cached_path}")
        return rows

    if latitude is None or longitude is None:
        if location_name in LOCATIONS:
            latitude, longitude = LOCATIONS[location_name]
        else:
            raise ValueError(f"Unknown location '{location_name}'. Coordinates required.")

    print(f"[NASA API] Fetching '{location_name}' ({latitude}, {longitude}) from {start_date} to {end_date}...")
    return get_weather(
        latitude,
        longitude,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        force_download=force_download,
        location_name=location_name,
    )


# Ingest data for all locations
def get_all_locations_weather(
    locations=None,
    start_date="20000101",
    end_date="20241231",
    cache_dir=DEFAULT_RAW_DIR,
    force_download=False,
):
    if locations is None:
        locations = LOCATIONS

    weather_by_location = {}
    for loc_name, coords in locations.items():
        lat, lon = coords
        try:
            rows = get_location_weather(
                location_name=loc_name,
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            weather_by_location[loc_name] = rows
        except Exception as err:
            print(f"[Warning] Failed to ingest data for {loc_name}: {err}")

    return weather_by_location


# Test run
if __name__ == "__main__":
    data = get_all_locations_weather()
    for loc, rows in data.items():
        print(f"{loc}: {len(rows)} records")
