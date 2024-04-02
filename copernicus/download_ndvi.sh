#!/bin/bash

BASE_URL="https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/"

for year in {1982..2013}; do
    YEAR_URL="${BASE_URL}${year}/"
    FILES=$(curl -s "$YEAR_URL" | grep -oE 'href="[^"]+\.nc"' | cut -d'"' -f2)

    for file in $FILES; do
        DOWNLOAD_URL="${YEAR_URL}${file}"
        FILE_NAME=$(basename "$file")

        echo "Downloading: $DOWNLOAD_URL"
        wget "$DOWNLOAD_URL" -O "$FILE_NAME"
    done

    # After downloading, create a zip file for the year's data
    echo "Zipping data for year $year"
    zip -r "${year}_data.zip" *.nc

    # Clean up the individual downloaded files
    rm -f *.nc
done

echo "Download and zip complete!"