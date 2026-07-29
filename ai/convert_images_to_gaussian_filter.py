"""Apply Gaussian blur filtering to all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and applies Gaussian blur filtering for smoothing and noise reduction.
Useful as preprocessing for feature detection algorithms.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_GAUSSIAN_FILTER_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_filter_by_gaussian import image_to_filter_by_gaussian
from helpers.convert_images import convert_images

print("Starting Gaussian filter application...")

# Apply Gaussian blur filtering to all images in the dataset
# This smooths images and reduces noise while preserving edges
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_GAUSSIAN_FILTER_PATH,
    function=image_to_filter_by_gaussian,
)

print(f"Gaussian filtering completed. Results saved to: {OUTPUT_IMAGE_FOLDER_GAUSSIAN_FILTER_PATH}")
