"""Detect corners using FAST (Features from Accelerated Segment Test).

FAST is a computationally efficient corner detection algorithm that
identifies corners by examining pixels in a circular pattern.
"""

import cv2


def image_to_fast_object_detection(input_image):
    """
    Apply FAST corner detection.
    
    Detects corner features using FAST algorithm and visualizes
    them on the original image.
    
    Args:
        input_image: Input image array
        
    Returns:
        Image with FAST corner keypoints drawn on it
    """
    if input_image is None:
        print("Error: Failed to load image")
        return None

    # Initialize FAST corner detector
    fast = cv2.FastFeatureDetector_create()

    # Detect corners directly on the color image
    keypoints = fast.detect(input_image, None)

    # Draw keypoints on original image in green
    fast_image = cv2.drawKeypoints(input_image, keypoints, None, color=(0, 255, 0))

    return fast_image
