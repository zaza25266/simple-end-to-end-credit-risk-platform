#!/bin/bash
# scripts/download_data.sh

set -e # Exit immediately if any command fails

# Technical Guardrail: Fail loudly if Kaggle credentials are missing
if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "$KAGGLE_USERNAME" ]; then
    echo "ERROR: Kaggle API credentials not found at ~/.kaggle/kaggle.json or env variables." >&2
    exit 1
fi

echo "Downloading 'Give Me Some Credit' dataset from Kaggle..."

# Target download directory
OUTPUT_DIR="data/raw"

# Download using the official Kaggle CLI directly into our raw vault
kaggle competitions download -c GiveMeSomeCredit -p $OUTPUT_DIR

# Unzip and clean up the archive wrapper
echo "Extracting data files..."
unzip -o "$OUTPUT_DIR/GiveMeSomeCredit.zip" -d $OUTPUT_DIR
rm "$OUTPUT_DIR/GiveMeSomeCredit.zip"

echo "Data acquisition successfully completed inside $OUTPUT_DIR/"