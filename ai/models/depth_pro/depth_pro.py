"""Depth Estimation with Depth Pro Model.

This module provides depth estimation using the Depth Pro model.
It processes images from the dataset and generates depth maps.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import depth_pro
from constants.paths import OUTPUT_IMAGE_FOLDER_DEPTH_PRO_PATH, INPUT_IMAGE_FOLDER_PATHS
from helpers.convert_images import convert_images


def image_to_depth_pro(image):
    """
    Estimate depth using the Depth Pro model.
    
    This function takes an image and generates a depth map showing the relative
    distance of objects from the camera. The depth values are normalized to 0-255
    for visualization purposes.
    
    Args:
        image: Input image (BGR format from OpenCV)
    
    Returns:
        Depth map image in BGR format for visualization, or None if processing fails
    """
    if image is None:
        return None
    
    try:
        # Load model and preprocessing transform (cached after first load)
        if not hasattr(image_to_depth_pro, 'model'):
            image_to_depth_pro.model, image_to_depth_pro.transform = depth_pro.create_model_and_transforms()
            image_to_depth_pro.model.eval()
        
        # Convert BGR to RGB for the model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Load and preprocess the image
        image_data, _, f_px = depth_pro.load_rgb(pil_image)
        image_tensor = image_to_depth_pro.transform(image_data)
        
        # Run depth inference
        with torch.no_grad():
            prediction = image_to_depth_pro.model.infer(image_tensor, f_px=f_px)
        
        # Extract depth map
        depth_map = prediction["depth"]  # Depth in meters
        
        # Normalize depth to 0-255 range for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        
        # Convert to 3-channel BGR for consistency
        depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        
        return depth_bgr
    
    except Exception as e:
        print(f"Error in depth estimation: {str(e)}")
        return None


# Note: This script is standalone and should be run separately, not via main.py
# Usage: python models/depth_pro/depth_pro.py
if __name__ == "__main__":
    print("--- Starting Depth Pro Processing ---")
    print("Loading Depth Pro model...")
    
    # Process images using the convert_images helper
    print("Processing dataset images for depth estimation...")
    convert_images(
        output_folder_path=OUTPUT_IMAGE_FOLDER_DEPTH_PRO_PATH,
        function=image_to_depth_pro,
    )
    
    print(f"Depth estimation completed. Results saved to: {OUTPUT_IMAGE_FOLDER_DEPTH_PRO_PATH}")
