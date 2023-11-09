#!/bin/bash

# Check if a directory path is provided as an argument
if [ -z "$1" ]; then
  echo "Please provide a directory path as an argument"
  exit 1
fi

# Check if the provided directory path exists
if [ ! -d "$1" ]; then
  echo "The provided directory path does not exist"
  exit 1
fi

# Convert SVG files to PNGs using ImageMagick
for file in "$1"/*.svg; do
  if [ -f "$file" ]; then
    filename="${file%.*}"
    convert "$file" "${filename}.png"
    echo "Converted $file to ${filename}.png"
  fi
done
