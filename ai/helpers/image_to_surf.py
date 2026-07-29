"""Extract SURF (Speeded Up Robust Features) keypoints and descriptors.

SURF provides a faster and more robust alternative to SIFT while
maintaining invariance to scale, rotation, and lighting changes.
"""

import cv2
from helpers.image_to_grayscale import image_to_grayscale


def image_to_surf(input_image):
    """
    Extract and visualize SURF features.
    
    Detects keypoints and computes SURF descriptors, then visualizes
    keypoints on the image.
    
    Args:
        input_image: Input image array
        
    Returns:
        Image with SURF keypoints drawn on it
    """
    # Convert to grayscale for feature detection
    gray_image = image_to_grayscale(input_image)

    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Initialize SURF detector with Hessian threshold of 400
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
    except AttributeError:
        print("Error: SURF not available - install opencv-contrib-python")
        return gray_image

    # Detect keypoints and compute descriptors
    keypoints, descriptors = surf.detectAndCompute(gray_image, None)

    # Draw keypoints with rich visualization
    output_image = cv2.drawKeypoints(
        gray_image,
        keypoints,
        None,
        (255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return output_image
