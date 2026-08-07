"""Isolate the building from the background in every input image.

This script separates the damaged building from sky, ground and surrounding
scenery, which is a prerequisite for the later stages of the pipeline: facade
parsing should not see neighbouring blocks, and the point cloud builder in
images_to_3d.py expects a background that has already been removed.

Four results are written per image:
  *_cutout.png    transparent cut-out, the archival result
  *_white_bg.jpg  white backdrop, ready for images_to_3d.py
  *_mask.jpg      the alpha matte on its own
  *_overlay.jpg   green tint over the kept region, for visual verification

Sample images at the repository root are processed alongside the dataset
splits, so the algorithm can be checked by eye before a full run.
"""

import os

from constants.paths import (
    OUTPUT_IMAGE_FOLDER_ISOLATION_PATH,
    INPUT_IMAGE_FOLDER_PATHS,
    SAMPLE_IMAGES_DIR,
)
from helpers.image_to_isolated import image_to_isolated
from helpers.convert_images import convert_images

print("Starting building isolation process...")

# Sample images first, so a visual check is available early in the run
candidate_folders = [SAMPLE_IMAGES_DIR] + INPUT_IMAGE_FOLDER_PATHS
input_folders = [folder for folder in candidate_folders if os.path.isdir(folder)]

if not input_folders:
    print("Error: no input folders found. Add images to the 'images' folder or the dataset.")
else:
    for folder in input_folders:
        print(f"  Input folder: {folder}")

    # Isolation runs at the original resolution: the square resize used by the
    # other filters distorts the aspect ratio, which would carry through into
    # the cut-out and the 3D reconstruction
    convert_images(
        output_folder_path=OUTPUT_IMAGE_FOLDER_ISOLATION_PATH,
        function=image_to_isolated,
        use_diff_sizes=False,
        input_folder_paths=input_folders,
    )

    print(f"Building isolation completed. Results saved to: {OUTPUT_IMAGE_FOLDER_ISOLATION_PATH}")
