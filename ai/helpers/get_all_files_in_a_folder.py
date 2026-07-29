"""Utility module for recursively finding all images in a directory tree."""

import os


def get_all_files_in_a_folder(base_path):
    """
    Recursively find all images in a directory and its subdirectories.
    
    Returns relative paths to all image files found, maintaining folder structure.
    This is useful for batch processing datasets with nested folder hierarchies
    (e.g., train/images, test/images, etc.)
    
    Args:
        base_path: Root directory to search for images
        
    Returns:
        Sorted list of relative paths to all image files found
    """
    if os.path.exists(base_path) and os.path.isdir(base_path):
        relative_files_list = []

        # Walk through the directory tree recursively
        for root, dirs, files in os.walk(base_path):
            for file in files:
                # Filter for image file extensions
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    # Construct full file path
                    full_path = os.path.join(root, file)
                    # Get relative path from base directory (preserves folder structure)
                    rel_path = os.path.relpath(full_path, base_path)
                    relative_files_list.append(rel_path)

        # Sort for consistent processing order
        relative_files_list.sort()
        return relative_files_list
    else:
        print(f"Error: Path '{base_path}' does not exist or is not a directory.")
        return []