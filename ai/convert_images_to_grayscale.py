"""Convert all dataset images to grayscale format.

This script processes all images from the dataset (train, test, validation splits)
and converts them to grayscale format, saving outputs to the designated directory.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_GRAYSCALE_PATH
from helpers.image_to_grayscale import image_to_grayscale
from helpers.convert_images import convert_images

# Import path constants for dataset images
from constants.paths import INPUT_IMAGE_FOLDER_PATHS

print("Starting grayscale conversion process...")

# Process all images from the dataset using the image_to_grayscale function
# The convert_images helper iterates through all specified input folders
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_GRAYSCALE_PATH,
    function=image_to_grayscale,
)

print(f"Grayscale conversion completed. Results saved to: {OUTPUT_IMAGE_FOLDER_GRAYSCALE_PATH}")
