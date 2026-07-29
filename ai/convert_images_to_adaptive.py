"""Apply adaptive thresholding to all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and applies adaptive thresholding, which dynamically adjusts the threshold value
for different regions of the image for better results in variable lighting conditions.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_ADAPTIVE_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_adaptive import image_to_adaptive
from helpers.convert_images import convert_images

print("Starting adaptive thresholding conversion process...")

# Apply adaptive thresholding to all images in the dataset
# This technique adjusts threshold values locally, improving results on images with uneven lighting
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_ADAPTIVE_PATH,
    function=image_to_adaptive,
)

print(f"Adaptive thresholding completed. Results saved to: {OUTPUT_IMAGE_FOLDER_ADAPTIVE_PATH}")
