#!/usr/bin/env bash
# Download the MIT-BIH Long-Term AF Database (84 records, 6 to 26 hours each,
# 128 Hz, rhythm-annotated) into data/ltafdb/.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/data/ltafdb"
mkdir -p "$dest"

if [ ! -f "$dest/RECORDS" ]; then
    curl -fL --retry 3 -o "$dest/ltafdb.zip" \
        "https://physionet.org/content/ltafdb/get-zip/1.0.0/"
    unzip -q -o "$dest/ltafdb.zip" -d "$dest"
    inner="$(find "$dest" -maxdepth 2 -name RECORDS -exec dirname {} \; | head -1)"
    if [ -z "$inner" ]; then
        echo "RECORDS not found under $dest after unzip" >&2
        exit 1
    fi
    if [ "$inner" != "$dest" ]; then
        mv "$inner"/* "$dest/"
        rmdir "$inner" 2>/dev/null || true
    fi
    rm "$dest/ltafdb.zip"
fi

echo "records: $(wc -l < "$dest/RECORDS" | tr -d ' ')"
