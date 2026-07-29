"""Extract and visualize SURF features from all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and extracts Speeded Up Robust Features (SURF) keypoints and descriptors,
which provide a faster and more robust alternative to SIFT.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_SURF_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_surf import image_to_surf
from helpers.convert_images import convert_images

print("Starting SURF feature extraction...")

# Extract SURF features from all images in the dataset
# SURF is faster than SIFT while maintaining robustness to scale, rotation, and lighting changes
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_SURF_PATH,
    function=image_to_surf,
)

print(f"SURF feature extraction completed. Results saved to: {OUTPUT_IMAGE_FOLDER_SURF_PATH}")
