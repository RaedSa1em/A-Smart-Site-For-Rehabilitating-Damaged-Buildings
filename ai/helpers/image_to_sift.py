"""Extract SIFT (Scale-Invariant Feature Transform) keypoints and descriptors.

SIFT features are robust to scale, rotation, and illumination changes,
making them useful for image matching and object recognition.
"""

import cv2
from helpers.image_to_grayscale import image_to_grayscale


def image_to_sift(input_image):
    """
    Extract and visualize SIFT features.
    
    Detects keypoints and computes SIFT descriptors, then visualizes
    keypoints on the image.
    
    Args:
        input_image: Input image array
        
    Returns:
        Image with SIFT keypoints drawn on it
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Convert to grayscale for feature detection
    gray_image = image_to_grayscale(input_image)

    # Validate image was loaded successfully
    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Draw keypoints on grayscale image for visualization
    image_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, None)

    return image_with_keypoints
