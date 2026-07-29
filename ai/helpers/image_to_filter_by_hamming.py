"""Hamming window filtering for frequency domain analysis.

Applies a Hamming window to images to improve frequency analysis
and reduce spectral leakage in signal processing operations.
"""

import cv2
import numpy as np
from helpers.image_to_grayscale import image_to_grayscale


def image_to_filter_by_hamming(input_image):
    """
    Apply Hamming window filter to an image.
    
    Converts image to grayscale and applies a Hamming window for
    improved frequency domain analysis.
    
    Args:
        input_image: Input image array
        
    Returns:
        Filtered image array after Hamming window application
    """
    # Convert to grayscale for filtering
    gray_image = image_to_grayscale(input_image)

    # Validate image was loaded successfully
    if gray_image is None:
        print("Error: Failed to load image")
        return None

    # Get image dimensions
    height, width = gray_image.shape

    # Create 2D Hamming window matching image dimensions
    # Outer product of 1D Hamming windows creates 2D window
    hamming_window = np.outer(np.hamming(height), np.hamming(width))

    # Apply Hamming window by element-wise multiplication
    # This tapers the image values near edges for better frequency analysis
    filtered_image = np.multiply(gray_image.astype(np.float32), hamming_window)

    # Convert back to uint8 for image compatibility
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image
