"""Extract ORB (Oriented FAST and Rotated BRIEF) features.

ORB provides a fast, free alternative to SIFT with rotation invariance,
suitable for real-time applications and embedded systems.
"""

import cv2
from helpers.image_to_grayscale import image_to_grayscale


def image_to_orb(input_image):
    """
    Extract and visualize ORB features.
    
    Detects keypoints and computes ORB descriptors, then visualizes
    keypoints on the image.
    
    Args:
        input_image: Input image array
        
    Returns:
        Image with ORB keypoints drawn on it
    """
    # Convert to grayscale for feature detection
    gray_image = image_to_grayscale(input_image)

    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Initialize ORB detector with 1000 keypoints
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    # Draw keypoints with rich visualization
    output_image = cv2.drawKeypoints(
        gray_image,
        keypoints,
        None,
        (0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return output_image
