"""Convert images to grayscale format.

Converts color images from BGR (OpenCV default) to grayscale representation
for use in grayscale-based image processing algorithms.
"""

import cv2


def image_to_grayscale(input_image):
    """
    Convert image to grayscale.
    
    Args:
        input_image: Input image array in BGR format
        
    Returns:
        Grayscale image array, or None if input is invalid
    """
    if input_image is None:
        print("Error: Failed to load image")
        return None

    # Convert from BGR color space to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    return gray_image
