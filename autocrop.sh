#!/bin/bash


# Loop through all files in the specified directory
for file in "$@"
do
  # Check if the file is an image
  if [[ $file == *.jpeg || $file == *.png ]]
  then
    # Crop the image using the `convert` command from the ImageMagick suite
    echo "Cropping $file"
    convert $file -trim +repage $file
  fi
done


# Frontiers

# mark images
# convert input_image -pointsize 50 -fill black -annotate +20+50 '(A)' output_image && xdg-open output_image

# merge images
# convert \( 16.jpeg -resize x500 \) \( 17.jpeg -resize x500 -splice 10x0 \)  +append 1617.jpeg