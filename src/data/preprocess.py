from data.nasa_api import PARAMETERS

MISSING_VALUE = -999.0


def clean_rows(rows):
    cleaned_rows = []

    for row in rows:
        cleaned_row = {
            "YEAR": int(row["YEAR"]),
            "DOY": int(row["DOY"]),
        }

        for parameter in PARAMETERS:
            value = float(row[parameter])

            if value == MISSING_VALUE:
                cleaned_row[parameter] = None
            else:
                cleaned_row[parameter] = value

        cleaned_rows.append(cleaned_row)

    return cleaned_rows