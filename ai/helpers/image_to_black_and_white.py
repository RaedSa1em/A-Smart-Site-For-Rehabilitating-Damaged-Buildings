"""Binary image conversion with multiple threshold values.

Converts images to binary (black and white) format using various
threshold values to explore different binarization effects.
"""

import cv2


def image_to_black_and_white(input_image):
    """
    Convert image to binary (black and white) using multiple threshold values.
    
    Converts image to grayscale and applies binary thresholding at different
    levels to create multiple binary representations.
    
    Args:
        input_image: Input image array
        
    Returns:
        List of tuples containing (binary_image, threshold_value)
    """
    # Validate image was loaded successfully
    if input_image is None:
        print("Error: Failed to load image")
        return None

    binary_images = []  # Store results
    
    # Threshold values to test for binarization
    threshold_values = [50, 100, 150, 200]

    # Apply each threshold value to create binary versions
    for threshold_val in threshold_values:
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold - pixels above threshold become white (255)
        # Pixels below become black (0)
        _, binary_image = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Store binary image with its threshold value for reference
        binary_images.append((binary_image, threshold_val))

    return binary_images
