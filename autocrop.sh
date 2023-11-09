#!/bin/bash


# Loop through all files in the specified directory
for file in "$@"
do
  # Check if the file is an image
  if [[ $file == *.jpg || $file == *.png ]]
  then
    # Crop the image using the `convert` command from the ImageMagick suite
    echo "Cropping $file"
    convert $file -trim +repage $file
  fi
done
