"""Image processing and batch conversion utility module.

This module provides functionality for batch processing images through various
transformation functions, with support for resizing, processing, and saving results.
"""

import os
import cv2
import numpy as np
from helpers.get_all_files_in_a_folder import get_all_files_in_a_folder
from constants.paths import INPUT_IMAGE_FOLDER_PATHS

# Standard image sizes for processing
SIZES = [1000]


def remove_extension(filename):
    """
    Remove file extension from a filename.
    
    Args:
        filename: The filename with extension
        
    Returns:
        Filename without extension
    """
    return os.path.splitext(filename)[0]


def resize_image(image, size):
    """
    Resize an image to a square format.
    
    Args:
        image: Input image array
        size: Target size (width and height in pixels)
        
    Returns:
        Resized image array
    """
    return cv2.resize(image, (size, size))


def process_image(image, function):
    """
    Apply a processing function to an image.
    
    Args:
        image: Input image array
        function: Processing function to apply
        
    Returns:
        List of tuples containing processed images and optional metadata (e.g., threshold values)
    """
    processed_output = function(image)
    return (
        processed_output
        if isinstance(processed_output, list)
        else [(processed_output, None)]
    )


def build_metadata_suffix(metadata):
    """
    Build the filename suffix that describes a processed variant.

    Numeric metadata is a threshold value (e.g. 150 -> "_threshold_150"),
    while textual metadata names a variant produced by the same algorithm
    (e.g. "cutout" -> "_cutout").

    Args:
        metadata: Threshold value, variant name, or None

    Returns:
        Filename suffix string (empty when there is no metadata)
    """
    if metadata is None:
        return ""

    if isinstance(metadata, (int, float)):
        return f"_threshold_{metadata}"

    return f"_{metadata}"


def save_images(images, output_path_base, size):
    """
    Save processed images with appropriate naming and directory structure.

    Args:
        images: List of tuples (processed_image, metadata)
        output_path_base: Base path for output files
        size: Size suffix for the output filename
    """
    for processed_image, metadata in images:
        if processed_image is None:
            continue

        # Build filename with optional threshold / variant metadata
        metadata_suffix = build_metadata_suffix(metadata)
        size_suffix = f"_{size}x{size}" if size != "Original" else "_original"

        # Images carrying an alpha channel must be written as PNG, since
        # JPEG cannot store transparency
        has_alpha = processed_image.ndim == 3 and processed_image.shape[2] == 4
        extension = ".png" if has_alpha else ".jpg"

        # Construct final filename
        final_output_path = f"{output_path_base}{size_suffix}{metadata_suffix}{extension}"

        # Create output subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        # Save the processed image
        cv2.imwrite(final_output_path, processed_image)


def convert_images(
    output_folder_path, function, use_diff_sizes=True, input_folder_paths=None
):
    """
    Main batch image processing pipeline.

    Processes all images from the dataset (train, test, validation splits),
    applies the specified transformation function, and saves results with
    proper directory structure and naming conventions.

    Args:
        output_folder_path: Directory where processed images will be saved
        function: Processing function to apply to each image
        use_diff_sizes: Whether to resize images before processing (default: True)
        input_folder_paths: Folders to read images from. Defaults to the
            dataset splits, but can be overridden to run an algorithm on a
            small set of sample images instead.
    """
    if input_folder_paths is None:
        input_folder_paths = INPUT_IMAGE_FOLDER_PATHS

    for input_folder_path in input_folder_paths:
        # Get all image files from the current input folder
        # Includes recursive search through subfolders
        images_rel_paths = get_all_files_in_a_folder(input_folder_path)

        # Process each image found
        for rel_path in images_rel_paths:
            input_image_path = os.path.join(input_folder_path, rel_path)

            # Maintain folder structure in output (e.g., train/image_name)
            output_image_base = os.path.join(
                output_folder_path, remove_extension(rel_path)
            )

            # Read image from disk
            input_image = cv2.imread(input_image_path)
            if input_image is None:
                print(f"Warning: Could not read image {input_image_path}")
                continue

            # Process with optional resizing
            if use_diff_sizes:
                for size in SIZES:
                    # Resize image to standard size
                    resized_image = resize_image(input_image, size)
                    # Apply processing function
                    processed_images = process_image(resized_image, function)
                    # Save processed results
                    save_images(processed_images, output_image_base, size)
            else:
                # Process image at original size
                processed_images = process_image(input_image, function)
                save_images(processed_images, output_image_base, "Original")

    print(f"Processing complete! Results saved to: {output_folder_path}")