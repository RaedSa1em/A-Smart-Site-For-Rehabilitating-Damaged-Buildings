"""Apply grayscale thresholding to all dataset images.

This script processes all images from the dataset (train, test, validation splits),
converts them to grayscale, and applies thresholding to create binary images
that highlight features based on intensity levels.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_GRAYSCALE_THRESHOLD_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_grayscale_threshold import apply_thresholds
from helpers.convert_images import convert_images

print("Starting grayscale thresholding process...")

# Convert images to grayscale and apply thresholding
# This creates binary images that emphasize structural differences in the original images
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_GRAYSCALE_THRESHOLD_PATH,
    function=apply_thresholds,
)

print(f"Grayscale thresholding completed. Results saved to: {OUTPUT_IMAGE_FOLDER_GRAYSCALE_THRESHOLD_PATH}")
