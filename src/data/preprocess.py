import os
import sys

from data.nasa_api import PARAMETERS


MISSING_VALUE = -999.0


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
