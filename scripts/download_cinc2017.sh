#!/usr/bin/env bash
# Download the PhysioNet/CinC 2017 public training set (8,528 records) and the
# revised v3 reference labels into data/cinc2017/.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/data/cinc2017"
mkdir -p "$dest"

base="https://physionet.org/files/challenge-2017/1.0.0"

if [ ! -f "$dest/training2017/REFERENCE.csv" ]; then
    curl -fL --retry 3 -o "$dest/training2017.zip" "$base/training2017.zip"
    unzip -q -o "$dest/training2017.zip" -d "$dest"
    rm "$dest/training2017.zip"
fi

# REFERENCE-v3 is the post-challenge label revision; the loader prefers it.
curl -fL --retry 3 -o "$dest/REFERENCE-v3.csv" "$base/REFERENCE-v3.csv"

echo "records: $(ls "$dest/training2017"/*.hea | wc -l | tr -d ' ')"
