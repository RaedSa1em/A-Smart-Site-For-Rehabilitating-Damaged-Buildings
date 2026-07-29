"""Gaussian blur filtering for image smoothing and noise reduction.

Applies a Gaussian blur filter to grayscale images to smooth and denoise them.
Useful for preprocessing before feature detection and other image analysis tasks.
"""

import cv2
from helpers.image_to_grayscale import image_to_grayscale

# Gaussian kernel size and standard deviation for blur effect
KERNEL_SIZE = (5, 5)  # Width x Height of the filter window
SIGMA = 1.5  # Standard deviation controlling blur spread


def image_to_filter_by_gaussian(input_image):
    """
    Apply Gaussian blur filter to an image.
    
    Converts image to grayscale and applies Gaussian blur for smoothing
    and noise reduction.
    
    Args:
        input_image: Input image array
        
    Returns:
        Filtered image array after Gaussian blur application
    """
    # Convert to grayscale for filtering
    gray_image = image_to_grayscale(input_image)

    # Validate image was loaded successfully
    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Apply Gaussian blur - spreads pixel values across kernel for smoothing
    gaussian_filtered = cv2.GaussianBlur(gray_image, KERNEL_SIZE, SIGMA)

    return gaussian_filtered
