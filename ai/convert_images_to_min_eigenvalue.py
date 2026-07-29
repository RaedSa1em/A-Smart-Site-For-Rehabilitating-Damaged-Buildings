"""Extract and visualize minimum eigenvalue corner features from all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and extracts corner points using the minimum eigenvalue method, which is useful for
structure analysis and feature-based image matching.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_MIN_EIG_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_min_eigenvalue import image_to_min_eigenvalue
from helpers.convert_images import convert_images

print("Starting minimum eigenvalue feature extraction...")

# Extract minimum eigenvalue corners from all images
# This method detects corner points by analyzing the eigenvalues of the image gradient matrix
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_MIN_EIG_PATH,
    function=image_to_min_eigenvalue,
)

print(f"Minimum eigenvalue feature extraction completed. Results saved to: {OUTPUT_IMAGE_FOLDER_MIN_EIG_PATH}")
