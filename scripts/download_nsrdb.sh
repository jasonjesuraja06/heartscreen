#!/usr/bin/env bash
# Download the MIT-BIH Normal Sinus Rhythm Database (18 records, ~24 hours each,
# 128 Hz) into data/nsrdb/ for healthy-cohort false-alarm measurement.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/data/nsrdb"
mkdir -p "$dest"

if [ ! -f "$dest/RECORDS" ]; then
    curl -fL --retry 3 -o "$dest/nsrdb.zip" \
        "https://physionet.org/content/nsrdb/get-zip/1.0.0/"
    unzip -q -o "$dest/nsrdb.zip" -d "$dest"
    inner="$(find "$dest" -maxdepth 2 -name RECORDS -exec dirname {} \; | head -1)"
    if [ "$inner" != "$dest" ]; then
        mv "$inner"/* "$dest/"
        rmdir "$inner" 2>/dev/null || true
    fi
    rm "$dest/nsrdb.zip"
fi

echo "records: $(wc -l < "$dest/RECORDS" | tr -d ' ')"
