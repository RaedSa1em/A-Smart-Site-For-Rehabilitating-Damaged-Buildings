"""Extract minimum eigenvalue (Shi-Tomasi) corner features.

Uses the Shi-Tomasi corner detector to identify robust corner points
useful for feature matching and structure analysis.
"""

import cv2
from helpers.image_to_grayscale import image_to_grayscale


def image_to_min_eigenvalue(input_image):
    """
    Extract and visualize minimum eigenvalue corners.
    
    Detects corner points using minimum eigenvalue method (Shi-Tomasi)
    and visualizes them on the image.
    
    Args:
        input_image: Input image array
        
    Returns:
        Image with corner keypoints drawn on it
    """
    # Initialize minimum eigenvalue corner detector (Shi-Tomasi)
    # maxCorners=500: detect up to 500 corners
    # qualityLevel=0.01: corners above 1% of max quality
    # minDistance=10: minimum distance between corners
    # useHarrisDetector=False: use Shi-Tomasi (not Harris)
    min_eig_detector = cv2.GFTTDetector_create(
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=10,
        useHarrisDetector=False,
    )

    # Convert to grayscale for corner detection
    gray_image = image_to_grayscale(input_image)

    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Detect corner keypoints
    keypoints = min_eig_detector.detect(gray_image, None)

    # Draw keypoints on grayscale image for visualization
    image_with_keypoints = cv2.drawKeypoints(
        gray_image, keypoints, None, color=(0, 255, 0)
    )

    return image_with_keypoints
