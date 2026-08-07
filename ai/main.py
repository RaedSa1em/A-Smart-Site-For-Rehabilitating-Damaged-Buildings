"""
Main execution script for running all image processing and object detection algorithms.

This script orchestrates the execution of all processing pipelines:
1. Image preprocessing (building isolation, adaptive, grayscale, Otsu thresholding, etc.)
2. Feature extraction (SIFT, SURF, ORB, etc.)
3. Object detection (RT-DETR, Fast R-CNN)

All results are saved to the outputs directory with their respective subdirectories.
"""

import os
import sys

print("=" * 80)
print("STARTING BUILDING DAMAGE DETECTION PIPELINE")
print("=" * 80)

# Image processing algorithms
processing_scripts = [
    ("Building Isolation", "python ai/convert_images_to_isolation.py"),
    ("Adaptive Thresholding", "python ai/convert_images_to_adaptive.py"),
    ("Grayscale Conversion", "python ai/convert_images_to_grayscale.py"),
    ("Otsu Thresholding", "python ai/convert_images_to_otsu.py"),
    ("SIFT Feature Extraction", "python ai/convert_images_to_sift.py"),
    ("Grayscale Threshold", "python ai/convert_images_to_grayscale_threshold.py"),
    ("FAST Corner Detection", "python ai/convert_images_to_fast_object_detection.py"),
    ("ORB Feature Extraction", "python ai/convert_images_to_orb.py"),
    ("Min Eigenvalue Features", "python ai/convert_images_to_min_eigenvalue.py"),
    ("SURF Feature Extraction", "python ai/convert_images_to_surf.py"),
    ("Gaussian Filter", "python ai/convert_images_to_gaussian_filter.py"),
    ("Hamming Filter", "python ai/convert_images_to_hamming_filter.py"),
    ("Black and White Conversion", "python ai/convert_images_to_black_and_white.py"),
]

# Object detection algorithms
detection_scripts = [
    ("RT-DETR Object Detection", "python ai/RT_DETR.py"),
    ("Fast R-CNN Detection", "python ai/Fast_R_CNN.py"),
]

# Run all image processing scripts
print("\n--- PHASE 1: IMAGE PROCESSING ---")
for script_name, command in processing_scripts:
    print(f"\nExecuting: {script_name}")
    result = os.system(command)
    if result != 0:
        print(f"Warning: {script_name} completed with exit code {result}")

# Run all object detection scripts
print("\n--- PHASE 2: OBJECT DETECTION ---")
for script_name, command in detection_scripts:
    print(f"\nExecuting: {script_name}")
    result = os.system(command)
    if result != 0:
        print(f"Warning: {script_name} completed with exit code {result}")

print("\n" + "=" * 80)
print("ALL PROCESSING PIPELINES COMPLETED")
print("Results are saved in the 'outputs' directory")
print("=" * 80)