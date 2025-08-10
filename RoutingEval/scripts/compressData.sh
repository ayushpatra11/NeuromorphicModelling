#!/bin/bash

# Script to compress ../data and ../logs into a timestamped tar.gz archive in ../compressed

# Create ../compressed directory if it doesn't exist
if [ ! -d "../compressed" ]; then
    echo "Creating ../compressed directory..."
    mkdir ../compressed
fi

# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
tarfile="../compressed/backup_${timestamp}.tar.gz"

echo "Compressing ../data and ../logs into $tarfile ..."
tar -czvf "$tarfile" ../data ../logs

echo "Compression complete."