import requests
import csv 
import io

URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"

query_params = {
    "start": 20261031,
    "end": 20261101,
    "latitude-min": 90.0,
    "latitude-max": 90.0,
    "longitude-min": 180.0,
    "longitude-max": 180.0,
    "community": "ag",
    "parameters": "T2M",
    "format": "csv"
}

response = requests.get(URL, params=query_params)

csv_file = io.StringIO(response)
reader = csv.DictReader(csv_file)
for row in reader:
    print(row)

