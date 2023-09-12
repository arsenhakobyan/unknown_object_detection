#!/usr/bin/env python3

from osgeo import gdal
import numpy as np
import sys
import os
from PIL import Image

# Define input and output file paths
input_tiff_path = sys.argv[1]
output_tiff_path = os.path.join(os.path.dirname(input_tiff_path), "resized_" + os.path.basename(input_tiff_path))

scale = (float)(sys.argv[2])
# Open the input GeoTIFF file
input_dataset = gdal.Open(input_tiff_path, gdal.GA_ReadOnly)

input_width = input_dataset.RasterXSize
input_height = input_dataset.RasterYSize


# Calculate the new dimensions (half the size)
new_width = input_width // scale
new_height = input_height // scale

# Perform the translation and resizing
gdal.Translate(output_tiff_path, input_dataset, width=new_width, height=new_height)

# Close the input dataset
input_dataset = None

# Get the input raster band
print("Resized and saved the GeoTIFF map in", output_tiff_path)

