"""Apply FAST corner detection and feature extraction to all dataset images.

This script processes all images from the dataset (train, test, validation splits)
and applies the FAST (Features from Accelerated Segment Test) corner detection algorithm,
which identifies corner points efficiently for use in feature matching and tracking.
"""

from constants.paths import OUTPUT_IMAGE_FOLDER_FAST_DETECTION_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.image_to_fast_object_detection import image_to_fast_object_detection
from helpers.convert_images import convert_images

print("Starting FAST corner detection...")

# Apply FAST corner detection to all images
# FAST is computationally efficient and identifies corners by examining pixels in a circular pattern
convert_images(
    output_folder_path=OUTPUT_IMAGE_FOLDER_FAST_DETECTION_PATH,
    function=image_to_fast_object_detection,
)

print(f"FAST corner detection completed. Results saved to: {OUTPUT_IMAGE_FOLDER_FAST_DETECTION_PATH}")
