"""Apply Otsu's automatic thresholding to images.

Otsu's method automatically calculates the optimal threshold value
based on image histogram analysis for binary conversion.
"""

import cv2
from helpers.image_to_grayscale import image_to_grayscale


def image_to_otsu(input_image):
    """
    Apply Otsu's automatic thresholding.
    
    Calculates optimal threshold value automatically from histogram
    to create binary image with maximum between-class variance.
    
    Args:
        input_image: Input image array
        
    Returns:
        Otsu thresholded binary image
    """
    # Convert to grayscale for thresholding
    gray_image = image_to_grayscale(input_image)

    # Validate image was loaded successfully
    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Apply Otsu's thresholding - automatically determines optimal threshold
    _, otsu_thresh = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return otsu_thresh
