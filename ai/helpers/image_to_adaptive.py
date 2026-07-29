"""Apply adaptive thresholding to images.

Uses adaptive thresholding to handle varying lighting conditions
by computing threshold values locally for each image region.
"""

import cv2
from helpers.image_to_grayscale import image_to_grayscale


def image_to_adaptive(input_image):
    """
    Apply adaptive thresholding to image.
    
    Computes threshold values based on local image regions,
    improving results in images with uneven lighting.
    
    Args:
        input_image: Input image array
        
    Returns:
        Adaptively thresholded binary image
    """
    # Convert to grayscale for thresholding
    gray_image = image_to_grayscale(input_image)

    # Validate image was loaded successfully
    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Apply adaptive threshold using Gaussian weighted sum
    # 11x11 neighborhood with constant subtraction of 2
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return adaptive_thresh
