"""Convert all dataset images to binary (black and white) format.

This script processes all images from the dataset (train, test, validation splits)
and converts them to binary images using multiple threshold values to explore
different binarization effects.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_BLACK_AND_WHITE_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_black_and_white import image_to_black_and_white
from helpers.convert_images import convert_images

print("Starting binary (black and white) conversion...")

# Convert all images to binary using multiple threshold values
# This creates high-contrast black and white representations
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_BLACK_AND_WHITE_PATH,
    function=image_to_black_and_white,
)

print(f"Binary conversion completed. Results saved to: {OUTPUT_IMAGE_FOLDER_BLACK_AND_WHITE_PATH}")
