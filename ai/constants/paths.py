import os

# --- Input Configuration ---
# Base folder containing all datasets
# Get the absolute path to the ai/inputs directory
AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_BASE_DIR = os.path.join(AI_DIR, "inputs")

# Dataset directory with YAML config and train/test/valid splits
DATASET_DIR = os.path.join(INPUT_BASE_DIR, "dataset")
DATASET_CONFIG = os.path.join(DATASET_DIR, "data.yaml")

# Dataset splits - train, test, validation
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
VALID_DIR = os.path.join(DATASET_DIR, "valid")

# Image and label directories for each split
TRAIN_IMAGES = os.path.join(TRAIN_DIR, "images")
TRAIN_LABELS = os.path.join(TRAIN_DIR, "labels")

TEST_IMAGES = os.path.join(TEST_DIR, "images")
TEST_LABELS = os.path.join(TEST_DIR, "labels")

VALID_IMAGES = os.path.join(VALID_DIR, "images")
VALID_LABELS = os.path.join(VALID_DIR, "labels")

# For backward compatibility - list of all image paths
INPUT_IMAGE_FOLDER_PATHS = [TRAIN_IMAGES, TEST_IMAGES, VALID_IMAGES]

# For backward compatibility - list of all label paths
INPUT_LABEL_FOLDER_PATHS = [TRAIN_LABELS, TEST_LABELS, VALID_LABELS]

# --- Output Configuration ---
# Base output directory - in ai/outputs
OUTPUT_BASE_DIR = os.path.join(AI_DIR, "outputs")

# Image processing outputs
OUTPUT_IMAGE_FOLDER_GRAYSCALE_THRESHOLD_PATH = os.path.join(OUTPUT_BASE_DIR, "grayscale_threshold")
OUTPUT_IMAGE_FOLDER_GRAYSCALE_PATH = os.path.join(OUTPUT_BASE_DIR, "grayscale")
OUTPUT_IMAGE_FOLDER_BLACK_AND_WHITE_PATH = os.path.join(OUTPUT_BASE_DIR, "black_and_white")
OUTPUT_IMAGE_FOLDER_OTSU_PATH = os.path.join(OUTPUT_BASE_DIR, "otsu")
OUTPUT_IMAGE_FOLDER_ADAPTIVE_PATH = os.path.join(OUTPUT_BASE_DIR, "adaptive")
OUTPUT_IMAGE_FOLDER_SIFT_PATH = os.path.join(OUTPUT_BASE_DIR, "sift")
OUTPUT_IMAGE_FOLDER_ORB_PATH = os.path.join(OUTPUT_BASE_DIR, "orb")
OUTPUT_IMAGE_FOLDER_SURF_PATH = os.path.join(OUTPUT_BASE_DIR, "surf")
OUTPUT_IMAGE_FOLDER_MIN_EIG_PATH = os.path.join(OUTPUT_BASE_DIR, "MIN_EIGENVALUE")
OUTPUT_IMAGE_FOLDER_FAST_DETECTION_PATH = os.path.join(OUTPUT_BASE_DIR, "fast_object_detection")
OUTPUT_IMAGE_FOLDER_GAUSSIAN_FILTER_PATH = os.path.join(OUTPUT_BASE_DIR, "gaussian_filter")
OUTPUT_IMAGE_FOLDER_HAMMING_FILTER_PATH = os.path.join(OUTPUT_BASE_DIR, "hamming_filter")

# Depth estimation output
OUTPUT_IMAGE_FOLDER_DEPTH_PRO_PATH = os.path.join(OUTPUT_BASE_DIR, "depth_pro")

# Model detection outputs
OUTPUT_IMAGE_FOLDER_RT_DETR_PATH = os.path.join(OUTPUT_BASE_DIR, "Rt-DETR")
OUTPUT_IMAGE_FOLDER_FAST_R_CNN_PATH = os.path.join(OUTPUT_BASE_DIR, "Fast-R-CNN")

# Models directory - in ai/models
MODELS_DIR = os.path.join(AI_DIR, "models")
YOLO_MODELS_DIR = os.path.join(MODELS_DIR, "yolo")
DEPTH_PRO_MODELS_DIR = os.path.join(MODELS_DIR, "depth_pro")
DEPTH_PRO_CHECKPOINT = os.path.join(DEPTH_PRO_MODELS_DIR, "checkpoints", "depth_pro.pt")