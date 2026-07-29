"""Apply grayscale thresholding with multiple threshold values.

Converts grayscale images to binary format using various threshold values
to explore different binarization effects.
"""

import cv2


def apply_thresholds(input_image):
    """
    Apply grayscale thresholding at multiple levels.
    
    Converts image to grayscale and applies binary thresholding
    at different threshold values.
    
    Args:
        input_image: Input image array
        
    Returns:
        List of tuples (binary_image, threshold_value)
    """
    if input_image is None:
        print("Error: Failed to load image")
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Threshold values to test
    thresholds = [50, 100, 150, 200]
    results = []

    # Apply each threshold value
    for threshold_val in thresholds:
        # Apply binary threshold - creates high contrast black and white image
        _, th_img = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        # Store result with threshold value for reference
        results.append((th_img, threshold_val))

    return results
