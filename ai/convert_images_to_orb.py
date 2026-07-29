"""Extract and visualize ORB features from all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and extracts Oriented FAST and Rotated BRIEF (ORB) keypoints and descriptors,
which are fast and efficient alternatives to SIFT with rotation invariance.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_ORB_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_orb import image_to_orb
from helpers.convert_images import convert_images

print("Starting ORB feature extraction...")

# Extract ORB features from all images in the dataset
# ORB is a fast, free alternative to SIFT that provides rotation invariant features
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_ORB_PATH,
    function=image_to_orb,
)

print(f"ORB feature extraction completed. Results saved to: {OUTPUT_IMAGE_FOLDER_ORB_PATH}")
