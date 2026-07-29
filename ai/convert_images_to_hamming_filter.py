"""Apply Hamming window filtering to all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and applies Hamming window filtering to improve frequency domain analysis
and reduce spectral leakage.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_HAMMING_FILTER_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_filter_by_hamming import image_to_filter_by_hamming
from helpers.convert_images import convert_images

print("Starting Hamming filter application...")

# Apply Hamming window filtering to all images
# This tapers image values for improved frequency analysis
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_HAMMING_FILTER_PATH,
    function=image_to_filter_by_hamming,
)

print(f"Hamming filtering completed. Results saved to: {OUTPUT_IMAGE_FOLDER_HAMMING_FILTER_PATH}")
