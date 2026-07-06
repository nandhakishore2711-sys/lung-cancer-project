"""
Central configuration for the Flask app.

Everything here can be overridden with environment variables so you can
switch backends/models without touching code, e.g.:

    export MODEL_BACKEND=pytorch
    export MODEL_ARCHITECTURE=efficientnet
    python app.py
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# "tensorflow" or "pytorch" — which trained model the app should load.
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "tensorflow")

# "custom" or "efficientnet" — must match what you trained.
MODEL_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "custom")

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

TF_MODEL_PATH = os.environ.get(
    "TF_MODEL_PATH", os.path.join(SAVED_MODELS_DIR, f"lung_cnn_tf_{MODEL_ARCHITECTURE}.keras")
)
PT_MODEL_PATH = os.environ.get(
    "PT_MODEL_PATH", os.path.join(SAVED_MODELS_DIR, f"lung_cnn_pt_{MODEL_ARCHITECTURE}.pt")
)
LABELS_PATH = os.environ.get("LABELS_PATH", os.path.join(SAVED_MODELS_DIR, "labels.json"))

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
MAX_UPLOAD_MB = 10

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
