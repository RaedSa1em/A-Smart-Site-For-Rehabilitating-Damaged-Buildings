"""Apply Otsu's thresholding to all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and applies Otsu's binarization method, which automatically calculates the optimal
threshold value based on image histogram analysis.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_OTSU_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_otsu import image_to_otsu
from helpers.convert_images import convert_images

print("Starting Otsu thresholding conversion process...")

# Apply Otsu's binarization to all images
# This method automatically determines the optimal threshold value from the image histogram
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_OTSU_PATH,
    function=image_to_otsu,
)

print(f"Otsu thresholding completed. Results saved to: {OUTPUT_IMAGE_FOLDER_OTSU_PATH}")
