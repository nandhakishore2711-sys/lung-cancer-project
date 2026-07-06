"""
Shared image preprocessing utilities.

Used by BOTH the training scripts (train_tensorflow.py / train_pytorch.py)
and the Flask inference app (app.py), so that a model always sees images
prepared the exact same way at train time and at prediction time.
"""

import numpy as np
from PIL import Image

# Canonical settings — change here and everything downstream stays in sync.
IMG_SIZE = (224, 224)          # (width, height)
CLASS_NAMES = ["Benign", "Malignant", "Normal"]  # alphabetical == ImageFolder / image_dataset_from_directory default order


def load_and_preprocess_image(path_or_file, img_size=IMG_SIZE):
    """
    Load an image from a path or file-like object, convert to RGB,
    resize, and scale pixel values to [0, 1].

    Returns a float32 numpy array of shape (H, W, 3).
    """
    img = Image.open(path_or_file)
    img = img.convert("RGB")          # CT scans are often grayscale; CNNs below expect 3 channels
    img = img.resize(img_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def preprocess_for_tensorflow(path_or_file, img_size=IMG_SIZE):
    """Returns a batch of shape (1, H, W, 3) ready for a Keras model."""
    arr = load_and_preprocess_image(path_or_file, img_size)
    return np.expand_dims(arr, axis=0)


def preprocess_for_pytorch(path_or_file, img_size=IMG_SIZE):
    """Returns a torch-ready numpy array of shape (1, 3, H, W)."""
    arr = load_and_preprocess_image(path_or_file, img_size)
    arr = np.transpose(arr, (2, 0, 1))     # HWC -> CHW
    return np.expand_dims(arr, axis=0)


def validate_image_file(filename, allowed_extensions=None):
    """Basic filename extension guard used by the upload route."""
    if allowed_extensions is None:
        allowed_extensions = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions
