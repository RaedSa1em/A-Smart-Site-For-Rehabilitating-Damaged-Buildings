"""Fast R-CNN Object Detection on Dataset Images.

This script uses a pre-trained Fast R-CNN model (ResNet50-FPN backbone) to perform
object detection on all images in the dataset (train, test, validation splits).
It draws bounding boxes around detected objects and saves the annotated images.
"""

import os
import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from constants.paths import (
    OUTPUT_IMAGE_FOLDER_FAST_R_CNN_PATH,
    TRAIN_IMAGES,
    TEST_IMAGES,
    VALID_IMAGES,
)

print("--- Loading Fast R-CNN Model ---")
# Load pre-trained Fast R-CNN model with ResNet50-FPN backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Ensure the output directory exists
os.makedirs(OUTPUT_IMAGE_FOLDER_FAST_R_CNN_PATH, exist_ok=True)

# Define all dataset splits to process
dataset_splits = {
    "train": TRAIN_IMAGES,
    "test": TEST_IMAGES,
    "valid": VALID_IMAGES,
}

# Verify that dataset paths exist
for split_name, image_folder in dataset_splits.items():
    if not os.path.exists(image_folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {image_folder}\n"
            f"Please ensure the inputs/dataset/{split_name}/images folder exists "
            f"and you are running this script from the 'ai' directory or with the correct working path."
        )


def process_image(image_path, output_path):
    """
    Perform object detection on a single image and save the result with bounding boxes.
    
    Args:
        image_path: Path to the input image
        output_path: Path where the annotated image will be saved
    """
    # Load image and convert to RGB format
    image = Image.open(image_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert PIL image to tensor format for the model
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # Run object detection inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract detection results
    boxes = predictions[0]["boxes"].cpu().numpy().astype(int)
    scores = predictions[0]["scores"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()

    # Filter detections by confidence threshold
    conf_threshold = 0.5
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Draw bounding boxes and labels on the image
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        # Draw rectangle around detected object
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Add label and confidence score
        text = f"{label}: {score:.2f}"
        cv2.putText(
            image_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    # Save annotated image
    cv2.imwrite(output_path, image_cv)


print("Starting object detection on all dataset splits...")

# Process each dataset split
for split_name, image_folder in dataset_splits.items():
    print(f"Processing {split_name} split...")
    output_base_dir = os.path.join(OUTPUT_IMAGE_FOLDER_FAST_R_CNN_PATH, split_name)
    os.makedirs(output_base_dir, exist_ok=True)

    # Iterate through all images in the current split
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_image_path = os.path.join(image_folder, image_file)
            output_image_path = os.path.join(output_base_dir, image_file)
            
            # Process and save the image with detections
            process_image(input_image_path, output_image_path)

    print(f"Completed {split_name} split detection")

print(
    f"Object detection completed successfully. Results saved to: {OUTPUT_IMAGE_FOLDER_FAST_R_CNN_PATH}"
)
