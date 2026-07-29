"""Extract and visualize SIFT features from all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and extracts Scale-Invariant Feature Transform (SIFT) keypoints and descriptors,
which are robust to scale, rotation, and illumination changes.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_SIFT_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_sift import image_to_sift
from helpers.convert_images import convert_images

print("Starting SIFT feature extraction...")

# Extract SIFT features from all images in the dataset
# SIFT is scale and rotation invariant, making it useful for feature matching and object recognition
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_SIFT_PATH,
    function=image_to_sift,
)

print(f"SIFT feature extraction completed. Results saved to: {OUTPUT_IMAGE_FOLDER_SIFT_PATH}")
