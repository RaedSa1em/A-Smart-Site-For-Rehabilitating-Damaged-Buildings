"""Real-Time Object Detection with RT-DETR Model.

This script uses the RT-DETR (Real-Time Detection Transformer) model to perform
object detection on all images in the dataset (train, test, validation splits).
It generates annotated images with bounding boxes and saves detection results.
"""

import os
from constants.paths import (
    OUTPUT_IMAGE_FOLDER_RT_DETR_PATH,
    TRAIN_IMAGES,
    TEST_IMAGES,
    VALID_IMAGES,
)
from ultralytics import RTDETR

print("--- Loading RT-DETR Model ---")
# Initialize the pre-trained RT-DETR model with the provided weights
model = RTDETR("rtdetr-x.pt")

# Create the output directory structure if it doesn't exist
os.makedirs(OUTPUT_IMAGE_FOLDER_RT_DETR_PATH, exist_ok=True)

# Define all dataset splits to process
dataset_splits = {
    "train": TRAIN_IMAGES,
    "test": TEST_IMAGES,
    "valid": VALID_IMAGES,
}

print("Starting object detection on all dataset splits...")

# Process each dataset split (train, test, validation)
for split_name, image_folder in dataset_splits.items():
    if os.path.exists(image_folder):
        print(f"Processing {split_name} split...")

        # Run detection on all images in the current split
        model.predict(
            source=image_folder,
            conf=0.25,  # Confidence threshold for object detection
            save=True,  # Save annotated images with bounding boxes
            save_txt=True,  # Save detection coordinates in text format
            project=OUTPUT_IMAGE_FOLDER_RT_DETR_PATH,
            name=split_name,  # Create separate folder for each split
            exist_ok=True,  # Allow reprocessing of the same folder
            imgsz=1024,  # Input image size for the model
        )
        print(f"Completed {split_name} split detection")
    else:
        print(f"Warning: {split_name} split folder not found at {image_folder}")

print(
    f"Object detection completed successfully. Results saved to: {OUTPUT_IMAGE_FOLDER_RT_DETR_PATH}"
)